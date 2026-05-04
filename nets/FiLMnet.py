import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from torch.optim import Adam, lr_scheduler
from torch.nn import init,Conv2d,Conv3d,ConvTranspose2d,ConvTranspose3d

# ---------- utility: 3x3 tiling -> 9 channels at 256x256 ----------
def tile3x3_to_channels(img_768):  # img_768: [B, 1, 768, 768] or [B, C, 768, 768] (C=1 expected)
    B, C, H, W = img_768.shape
    assert H == 768 and W == 768 and C == 1, "Expect [B,1,768,768]"
    # split into 3x3 grid of 256x256 and stack as channels -> [B, 9, 256, 256]
    tiles = []
    for ry in range(3):
        for rx in range(3):
            tiles.append(img_768[:, :, ry*256:(ry+1)*256, rx*256:(rx+1)*256])
    x = torch.cat(tiles, dim=1)  # [B, 9, 256, 256]
    return x

# ---------- FiLM module ----------
class FiLM(nn.Module):
    def __init__(self, emb_dim, num_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(emb_dim, 2 * num_channels)  # -> gamma,beta
        )

    def forward(self, h, emb):  # h: [B,C,H,W], emb: [B,emb_dim]
        gb = self.mlp(emb)              # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)
        # gamma = torch.sigmoid(gamma)    # Keep scale reasonable
        
        # Debug: Check if parameters are changing
        if torch.is_grad_enabled():  # Only during training
            print(f"FiLM gamma range: [{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
            print(f"FiLM beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]")
            
        gamma = gamma[:, :, None, None] # [B,C,1,1]
        beta  = beta[:, :, None, None]
        return gamma * h + beta

# ---------- a FiLM-residual block ----------
class FilmResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, groups=8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # self.norm1 = nn.GroupNorm(groups, in_ch)
        if in_ch == 9:
            group = 9
        else:
            group = 8
        self.norm1 = nn.GroupNorm(group, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # constructor FiLM modules
        self.film1 = FiLM(emb_dim, in_ch)
        self.film2 = FiLM(emb_dim, out_ch)
        self.skip = (in_ch != out_ch)
        if self.skip:
            self.proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        # forward pass
        h = self.film1(self.norm1(x), emb)
        h = F.leaky_relu(h, 0.01)
        h = self.conv1(h)
        # forward pass
        h = self.film2(self.norm2(h), emb)
        h = F.leaky_relu(h, 0.01)
        h = self.conv2(h)
        if self.skip:
            x = self.proj(x)
        return x + h

# ---------- PSF encoder (18D vector -> compact embedding) ----------
class PSFEncoder(nn.Module):
    """
    Input: PSF parameters as [B, 18] vectors.
    Strategy: MLP layers to process the 18D input into embedding.
    The 18 parameters could represent various PSF characteristics like:
    - Depth-dependent shift coefficients
    """
    def __init__(self, input_dim=18, emb_dim=256):
        super().__init__()
        # MLP to process 18D input vectors into embeddings
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(emb_dim, emb_dim*2),
            nn.LeakyReLU(0.01),
            nn.Linear(emb_dim*2, emb_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, psf_params):  # [B, 18] - PSF parameter vectors
        # Ensure consistent dtype (float32)
        psf_params = psf_params.float()
        
        # Process through MLP to get embedding
        emb = self.mlp(psf_params)  # [B, emb_dim]
        return emb

# ---------- FiLM-UNet (2D) producing a 3D volume as channels ----------
class FiLMUNet2D(nn.Module):
    """
    In:  image tiles as [B, 9, 256, 256]
    Out: volume [B, 64, 256, 256]  (Z treated as channels)
    FiLM conditioning from psf_emb at every block.
    """
    def __init__(self, psf_emb_dim=256, base_ch=64, z_channels=64, dropout_rate=0.1):
        super().__init__()
        # Encoder
        self.enc1 = FilmResBlock(9, base_ch, psf_emb_dim)
               
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 256->128
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.enc2 = FilmResBlock(base_ch, base_ch*2, psf_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*2, 4, stride=2, padding=1)  # 128->64
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.enc3 = FilmResBlock(base_ch*2, base_ch*4, psf_emb_dim)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*4, 4, stride=2, padding=1)  # 64->32
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # Bottleneck
        self.bot  = FilmResBlock(base_ch*4, base_ch*4, psf_emb_dim)
        self.dropout_bot = nn.Dropout2d(dropout_rate * 1.5)  # Higher dropout in bottleneck

        # Decoder
        # self.up3  = nn.ConvTranspose2d(base_ch*4, base_ch*4, 4, stride=2, padding=1)  # 32->64
        # fist upsample then conv to reduce checkerboard artifacts
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 32->64
        self.dropout_up3 = nn.Dropout2d(dropout_rate)
        self.dec3 = FilmResBlock(base_ch*4 + base_ch*4, base_ch*2, psf_emb_dim)
        
        # self.up2  = nn.ConvTranspose2d(base_ch*2, base_ch*2, 4, stride=2, padding=1)  # 64->128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 64->128
        self.dropout_up2 = nn.Dropout2d(dropout_rate)
        self.dec2 = FilmResBlock(base_ch*2 + base_ch*2, base_ch, psf_emb_dim)
        
        # self.up1  = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)      # 128->256
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128->256
        self.dropout_up1 = nn.Dropout2d(dropout_rate * 0.5)  # Lower dropout near output
        self.dec1 = FilmResBlock(base_ch + base_ch, base_ch, psf_emb_dim)

        # Head to volume channels (no dropout here to preserve final output quality)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, z_channels, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(z_channels, z_channels*2, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(z_channels*2, z_channels, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01))
        # self.pos = nn.Softplus(beta=1.0, threshold=20.0)   
        
    def forward(self, x9, psf_emb):
        # Encoder
        e1 = self.enc1(x9, psf_emb)     # 256
        x  = self.dropout1(self.down1(e1))  # 128
        
        e2 = self.enc2(x, psf_emb)      # 128
        x  = self.dropout2(self.down2(e2))  # 64
        
        e3 = self.enc3(x, psf_emb)      # 64
        x  = self.dropout3(self.down3(e3))  # 32

        # Bottleneck
        x  = self.dropout_bot(self.bot(x, psf_emb))  # 32

        # Decoder
        x  = self.dropout_up3(self.up3(x))  # 64
        x  = torch.cat([x, e3], dim=1)
        x  = self.dec3(x, psf_emb)

        x  = self.dropout_up2(self.up2(x))  # 128
        x  = torch.cat([x, e2], dim=1)
        x  = self.dec2(x, psf_emb)

        x  = self.dropout_up1(self.up1(x))  # 256
        x  = torch.cat([x, e1], dim=1)
        x  = self.dec1(x, psf_emb)

        vol = self.head(x)                 # [B, Z, 256, 256]        
        # vol = self.pos(vol_logits)                # non-negative output
        return vol

# ---------- Wire everything together ----------
class PSFConditionedRecon(nn.Module):
    def __init__(self, psf_input_dim=18, emb_dim=256, base_ch=64, z_channels=64, dropout_rate=0.1):
        super().__init__()
        self.psf_enc = PSFEncoder(input_dim=psf_input_dim, emb_dim=emb_dim)
        self.unet    = FiLMUNet2D(psf_emb_dim=emb_dim, base_ch=base_ch, z_channels=z_channels, dropout_rate=dropout_rate)

    def forward(self, network_input):  
        # Handle dictionary input format from training script
        if isinstance(network_input, dict):
            img768 = network_input['curr_img_stack']      # [B,1,768,768]
            # Support both 'PSF' (new) and 'OTF' (legacy) keys for parameter vectors
            psf_params = network_input.get('PSF', network_input.get('OTF'))  # [B, 18] - PSF parameter vectors
        else:
            # Handle direct tuple/list input (img768, psf_params)
            img768, psf_params = network_input
            
        x9 = tile3x3_to_channels(img768)   # [B,9,256,256]
        psf_emb = self.psf_enc(psf_params)  # [B,emb_dim] - Process 18D vectors through MLP
          # Debug: Check if parameters are changing
        if torch.is_grad_enabled():  # Only during training
            print(f"psf_params shape: {psf_params.shape}")
            print(f"psf_emb: [{psf_emb.min().item():.4f}, {psf_emb.max().item():.4f}]")
            
        vol = self.unet(x9, psf_emb)       # [B,64,256,256]
        return vol

class setupNetwork(nn.Module):
    def __init__(self, args, stats, device):
        super(setupNetwork, self).__init__()
        args, checkpoint_FLFMnet = self.configUNet(args, device)
        net = PSFConditionedRecon(psf_input_dim=18, emb_dim=args.psf_emb_dim, base_ch=args.base_ch, z_channels=args.z_channels, dropout_rate=args.dropout_rate).to(device)
        
        net.apply(self.init_weights)
        print("# Weights initialization function created!")
        
        trainable_params = [{'params': net.parameters()}]
        params = sum([p.numel() for p in net.parameters()])

        optimizer = Adam(trainable_params, lr=args.learning_rate)
        
        # The scheduler waits for 2*20 epochs without improvement before reducing LR by factor 0.5
        lr = args.learning_rate
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        # lr_sched = lr_scheduler.ExponentialLR(optimizer,gamma=0.95, verbose=True)
        scaler = GradScaler()
        print("# Optimizer created", optimizer)
        net.stats = stats

        self.values2return = (net,checkpoint_FLFMnet,optimizer,lr,scaler,lr_sched,params)

    def configUNet(self, args, device):
        # Load previous checkpoints
        if len(args.checkpoint_FLFMnet)>0:
            checkpoint_FLFMnet = torch.load(args.checkpoint_FLFMnet, map_location=device, weights_only=False)
        else:
            checkpoint_FLFMnet = None
        return args,checkpoint_FLFMnet

    def init_weights(self,m):
        if type(m) == Conv2d or type(m) == Conv3d or type(m) == ConvTranspose2d or type(m) == ConvTranspose3d:
            init.xavier_uniform(m.weight)
            
            
# # ---------- Physics loss skeleton (plug your projector) ----------
# def forward_project_lfm(volume, psf_stack):
#     """
#     volume: [B, Z, 256, 256]
#     psf_stack: [B, Z, 768, 768]
#     Return: rendered 2D image [B, 1, 768, 768]

#     NOTE: Replace this with your *actual* FLFM forward model (per-EI & depth).
#     Here we just raise NotImplementedError to emphasize the hook.
#     """
#     raise NotImplementedError("Plug in your FLFM projector here.")

# # ---------- Training step example ----------
# def training_step(model, img768, psf_stack, optimizer, lam=1e-4, reg='hessian'):
#     model.train()
#     vol = model(img768, psf_stack)  # [B,64,256,256]

#     # 1) render with the *same PSF* used for conditioning
#     Ihat = forward_project_lfm(vol, psf_stack)  # [B,1,768,768]

#     # 2) data fidelity (L2 or Poisson NLL if you prefer RL-like)
#     data_loss = F.mse_loss(Ihat, img768)

#     # 3) simple regularizer on the volume (pick what you use today)
#     if reg == 'tv':
#         tv = (vol[:, :, 1:, :] - vol[:, :, :-1, :]).abs().mean() + \
#              (vol[:, :, :, 1:] - vol[:, :, :, :-1]).abs().mean()
#         reg_loss = tv
#     else:  # 'hessian' (lightweight)
#         dx = vol[:, :, 2:, 1:-1] - 2*vol[:, :, 1:-1, 1:-1] + vol[:, :, :-2, 1:-1]
#         dy = vol[:, :, 1:-1, 2:] - 2*vol[:, :, 1:-1, 1:-1] + vol[:, :, 1:-1, :-2]
#         reg_loss = (dx.abs().mean() + dy.abs().mean())

#     loss = data_loss + lam * reg_loss
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return {'loss': float(loss.item()),
#             'data': float(data_loss.item()),
#             'reg': float(reg_loss.item())}
