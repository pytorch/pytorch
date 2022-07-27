import math

from torch import nn
from torch.nn import init


def _initialize_orthogonal(conv):
    prelu_gain = math.sqrt(2)
    init.orthogonal(conv.weight, gain=prelu_gain)
    if conv.bias is not None:
        conv.bias.data.zero_()


class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.prelu = nn.PReLU(n_filters)
        self.conv2 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_filters)

        # Orthogonal initialisation
        _initialize_orthogonal(self.conv1)
        _initialize_orthogonal(self.conv2)

    def forward(self, x):
        residual = self.prelu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


class UpscaleBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.upscaling_conv = nn.Conv2d(
            n_filters, 4 * n_filters, kernel_size=3, padding=1
        )
        self.upscaling_shuffler = nn.PixelShuffle(2)
        self.upscaling = nn.PReLU(n_filters)
        _initialize_orthogonal(self.upscaling_conv)

    def forward(self, x):
        return self.upscaling(self.upscaling_shuffler(self.upscaling_conv(x)))


class SRResNet(nn.Module):
    def __init__(self, rescale_factor, n_filters, n_blocks):
        super().__init__()
        self.rescale_levels = int(math.log(rescale_factor, 2))
        self.n_filters = n_filters
        self.n_blocks = n_blocks

        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU(n_filters)

        for residual_block_num in range(1, n_blocks + 1):
            residual_block = ResidualBlock(self.n_filters)
            self.add_module(
                "residual_block" + str(residual_block_num),
                nn.Sequential(residual_block),
            )

        self.skip_conv = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )
        self.skip_bn = nn.BatchNorm2d(n_filters)

        for upscale_block_num in range(1, self.rescale_levels + 1):
            upscale_block = UpscaleBlock(self.n_filters)
            self.add_module(
                "upscale_block" + str(upscale_block_num), nn.Sequential(upscale_block)
            )

        self.output_conv = nn.Conv2d(n_filters, 3, kernel_size=9, padding=4)

        # Orthogonal initialisation
        _initialize_orthogonal(self.conv1)
        _initialize_orthogonal(self.skip_conv)
        _initialize_orthogonal(self.output_conv)

    def forward(self, x):
        x_init = self.prelu1(self.conv1(x))
        x = self.residual_block1(x_init)
        for residual_block_num in range(2, self.n_blocks + 1):
            x = getattr(self, "residual_block" + str(residual_block_num))(x)
        x = self.skip_bn(self.skip_conv(x)) + x_init
        for upscale_block_num in range(1, self.rescale_levels + 1):
            x = getattr(self, "upscale_block" + str(upscale_block_num))(x)
        return self.output_conv(x)
