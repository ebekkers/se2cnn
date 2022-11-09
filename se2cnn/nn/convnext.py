from torch import nn
from se2cnn.nn import SE2ToSE2Conv
import torch.nn.functional as F


class SE2ToSE2ConvNext(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, input_dim, output_dim, num_theta, kernel_size, padding=0):
        super().__init__()

        self.crop = int((2 * padding - (kernel_size - 1))/2)
        self.skip = (input_dim == output_dim)
        self.padding_mode = "replicate"

        self.dwconv = SE2ToSE2Conv(input_dim, input_dim, num_theta, kernel_size=kernel_size, padding=padding, groups=input_dim)
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(input_dim, 4 * input_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * input_dim, output_dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)  # [B, C, num_theta, X, Y]
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, Theta, H, W) -> (N, Theta, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (N, Theta, H, W, C) -> (N, C, Theta, H, W)
        if self.skip:
            if self.crop < 0:
                x = input[...,-self.crop:self.crop,-self.crop:self.crop] + x
            elif self.crop > 0:
                channels = input.shape[1]
                x = F.pad(input.flatten(1, 2), (self.crop, self.crop, self.crop, self.crop),
                          self.padding_mode).unflatten(1, (channels, -1)) + x
            else:
                x = input + x
        return x