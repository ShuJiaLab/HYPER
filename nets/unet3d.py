from torch import nn,cat
import torch.nn.functional as F
from torch.amp import autocast #,GradScaler
# from blocks import *

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

class ExtendChannels(nn.Module):
    def __init__(self, NumOfChannels):
        super(ExtendChannels, self).__init__()
        self.NumOfChannels = NumOfChannels

    def forward(self, x):
        return x.repeat(1, self.NumOfChannels, 1, 1, 1)
    
class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class UNet3d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode='upsample',
        drop_out=0,
        use_bias=False,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet3d, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.drop_out = drop_out #Randomly sets 30% of the feature map values to zero Scales remaining values by 1/(1-0.3) = 1.43 to maintain expected output magnitude
        prev_channels = 2**(wf-1)#in_channels
        self.down_path = nn.ModuleList()
        self.first = nn.Sequential(Unsqueeze(1),
                                   ExtendChannels(2**(wf-1)))
        # self.down_path.append(Unsqueeze(1))

        for i in range(depth):
            self.down_path.append(
                UNetConvBlock3d(prev_channels, 2 ** (wf + i), padding, batch_norm, use_bias=use_bias)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock3d(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, use_bias=use_bias)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(nn.Conv3d(prev_channels, 1, kernel_size=1, bias=use_bias),
                    nn.ReLU(),
                    Squeeze(1)#,
                    # nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=use_bias)
        )

    @autocast(device_type="cuda")
    def forward(self, x):
        blocks = []
        x = self.first(x)
        for i, down in enumerate(self.down_path):
            x = down(x)
            # print(x.shape)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool3d(x, 2)
                x = F.dropout3d(x, self.drop_out)
                # print(x.shape)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            x = F.dropout3d(x, self.drop_out)
            # print(x.shape)
            # print("---End of unet3d---")
        return self.last(x)


class UNetConvBlock3d(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, use_bias=False):
        super(UNetConvBlock3d, self).__init__()
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size=3, padding=int(padding), bias=use_bias))
        block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3, padding=int(padding), bias=use_bias))
        block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock3d(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, use_bias=False):
        super(UNetUpBlock3d, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, bias=use_bias)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='trilinear', scale_factor=2),
                nn.Conv3d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock3d(in_size, out_size, padding, batch_norm, use_bias=use_bias)

    def center_crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_z = (layer_depth - target_size[0]) // 2
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        return layer[
            :, :, diff_z : (diff_z + target_size[0]), diff_y : (diff_y + target_size[1]), diff_x : (diff_x + target_size[2])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
    