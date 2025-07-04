import torch
from torch import nn

class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 n_channels: list=None, batch_norm: bool=False) -> None:
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if n_channels is None:
            n_channels = [64, 128, 256, 512]

        self.in_conv = DoubleConv(in_channels, n_channels[0], batch_norm=batch_norm)
        self.encoder_1 = DownSample(n_channels[0], n_channels[1], batch_norm=batch_norm)
        self.encoder_2 = DownSample(n_channels[1], n_channels[2], batch_norm=batch_norm)
        self.encoder_3 = DownSample(n_channels[2], n_channels[3], batch_norm=batch_norm)

        self.decoder_1 = UpSample(n_channels[3], n_channels[2], n_channels[2], batch_norm=batch_norm)
        self.decoder_2 = UpSample(n_channels[2], n_channels[1], n_channels[1], batch_norm=batch_norm)
        self.decoder_3 = UpSample(n_channels[1], n_channels[0], n_channels[0], batch_norm=batch_norm)
        self.out_conv = OutConv(n_channels[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.encoder_1(x1)
        x3 = self.encoder_2(x2)
        x4 = self.encoder_3(x3)

        x = self.decoder_1(x4, x3)
        x = self.decoder_2(x, x2)
        x = self.decoder_3(x, x1)
        x = self.out_conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool=False) -> None:
        super(DoubleConv, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        if batch_norm:
            self.conv1.insert(index=1, module=nn.BatchNorm3d(mid_channels))
            self.conv2.insert(index=1, module=nn.BatchNorm3d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool=False) -> None:
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_norm=batch_norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, encoder_channels: int, batch_norm: bool=False) -> None:
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + encoder_channels, out_channels, batch_norm=batch_norm)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        decoder = self.up(decoder)
        x = torch.cat([encoder, decoder], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.out = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensor = torch.randn([8, 1, 96, 96, 96]).to(device)
    model = UNet3D(in_channels=1, out_channels=2, batch_norm=True).to(device)
    print(model)
    output = model(tensor)
    print(output.shape)