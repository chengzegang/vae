from torch import Tensor, nn


class _ConvNxN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.SiLU(True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Conv3x3(_ConvNxN):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, eps: float = 1e-5
    ):
        super().__init__(in_channels, out_channels, 3, stride, 1, eps)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-5):
        super().__init__()
        self.conv1 = Conv3x3(in_channels, out_channels, eps=eps)
        self.conv2 = Conv3x3(out_channels, out_channels, eps=eps)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.conv3(self.conv2(self.conv1(x)))
