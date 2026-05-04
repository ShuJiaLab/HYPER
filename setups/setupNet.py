from torch import load
from torch.optim import Adam, lr_scheduler
from torch.nn import init,Conv2d,Conv3d,ConvTranspose2d
from numpy import prod
from torch.cuda.amp import GradScaler
import torch

def get_bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    Create a 2D bilinear kernel for ConvTranspose2d weight initialization.
    The kernel is expanded to (out_channels, in_channels, k, k) shape.
    """
    # Create a 2D bilinear kernel
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = torch.arange(kernel_size, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor)
    kernel_2d = filt[:, None] * filt[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()  # Normalize

    # Initialize the weight tensor
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=torch.float32)
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = kernel_2d
    return weight

def configUNet(args, device):
    # Load previous checkpoints
    if len(args.checkpoint_FLFMnet)>0:
        checkpoint_FLFMnet = load(args.checkpoint_FLFMnet, map_location=device, weights_only=False)
        args_deconv = checkpoint_FLFMnet['args']
        args.unet_depth = args_deconv.unet_depth
        args.unet_wf = args_deconv.unet_wf
    else:
        checkpoint_FLFMnet = None

    unet_settings = {'depth':args.unet_depth, 'wf':args.unet_wf, 'drop_out':args.unet_drop_out}
    args.unet_settings = unet_settings
    print("# Unet settings: ", unet_settings)
    return args,checkpoint_FLFMnet

# def init_weights(m):
#     if type(m) == Conv2d or type(m) == Conv3d or type(m) == ConvTranspose2d:
#         init.xavier_uniform(m.weight)
 
#  to avoid checkerbpard artifacts in ConvTranspose2d, we use bilinear kernel initialization
def init_weights(m):
    if type(m) == Conv2d or type(m) == Conv3d:
        init.xavier_uniform(m.weight)
    elif type(m) == ConvTranspose2d:
        bilinear_kernel = get_bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(bilinear_kernel)

def setupNet(NETWORK_OBJ, dataset, args, stats, device):
    args,checkpoint_FLFMnet = configUNet(args, device)
    net = NETWORK_OBJ(dataset.n_lenslets, args.output_shape, dataset=dataset, 
              use_bias=args.use_bias, unet_settings=args.unet_settings).to(device)
    net.apply(init_weights)
    print("# Weights initialization function created!")

    # trainable_params = [{'params': net.deconv.parameters()}]
    trainable_params = [{'params': net.parameters()}]
    params = sum([prod(p.size()) for p in net.parameters()])

    optimizer = Adam(trainable_params, lr=args.learning_rate)
    lr = args.learning_rate
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()
    print("# Optimizer created", optimizer)
    net.stats = stats

    return net,checkpoint_FLFMnet,optimizer,lr,scaler,lr_sched,params