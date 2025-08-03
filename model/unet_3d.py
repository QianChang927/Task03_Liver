import torch
from torch import nn

# 类型注解所用库
from torch import Tensor
from torch.nn import ModuleList
from typing import Literal

class UNet3D(nn.Module):
    """
    3D U-Net类，总体框架分为编码部分和解码部分
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_channels: list=None,
            norm_layer: Literal['BatchNorm', 'InstanceNorm', 'None']='BatchNorm'
    ) -> None:
        """
        3D U-Net类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n_channels: 中间层通道数
        :param norm_layer: 归一化层类型
        :return:
        """
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = norm_layer

        if n_channels is None:
            n_channels = [64, 128, 256, 512]
        if len(n_channels) <= 1:
            raise ValueError('n_channels should have at least 2 elements')
        self.n_channels = n_channels

        self.in_conv = DoubleConv(self.in_channels, self.n_channels[0], norm_layer=self.norm_layer)
        self.encoder = self._build_layer(mode='encoder')
        self.decoder = self._build_layer(mode='decoder')
        self.out_conv = OutConv(self.n_channels[0], self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        # 输入层
        x = self.in_conv(x)

        # 编码过程
        features = [x]
        for layer in self.encoder:
            features.append(layer(features[-1]))

        # 解码过程
        x = features[-1]
        for layer, feature in zip(self.decoder, reversed(features[:-1])):
            x = layer(x, feature)

        # 输出层
        x = self.out_conv(x)
        return x

    def _build_layer(self, mode: Literal['encoder', 'decoder']) -> ModuleList:
        """
        构造中间层
        :param mode: 构造模式，选择构造编码器或解码器
        :return: 构造出的中间网络层
        """
        layers = nn.ModuleList()
        n_channels = self.n_channels if mode == 'encoder' else tuple(reversed(self.n_channels))
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            if mode == 'encoder':
                layer = DownSample(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_layer=self.norm_layer
                )
            elif mode == 'decoder':
                layer = UpSample(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    encoder_channels=out_channels,
                    norm_layer=self.norm_layer
                )
            else:
                raise ValueError('mode should be either `encoder` or `decoder`')
            layers.append(layer)
        return layers


class DoubleConv(nn.Module):
    """
    双层卷积，包含两个卷积层
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm_layer: Literal['BatchNorm', 'InstanceNorm', 'None']='BatchNorm'
    ) -> None:
        """
        双层卷积类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param norm_layer: 归一化层类型
        :return:
        """
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

        if norm_layer == 'BatchNorm':
            norm_class = nn.BatchNorm3d
        elif norm_layer == 'InstanceNorm':
            norm_class = nn.InstanceNorm3d
        elif norm_layer == 'None':
            norm_class = None
        else:
            raise NotImplementedError('norm_layer should be in ["BatchNorm", "InstanceNorm", "None"]')

        if norm_layer != 'None':
            self.conv1.insert(index=1, module=norm_class(mid_channels))
            self.conv2.insert(index=1, module=norm_class(out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSample(nn.Module):
    """
    编码器，执行下采样操作
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm_layer: Literal['BatchNorm', 'InstanceNorm', 'None']='BatchNorm'
    ) -> None:
        """
        编码器类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param norm_layer: 归一化层类型
        :return:
        """
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        x = self.down(x)
        return x


class UpSample(nn.Module):
    """
    解码器，执行上采样操作
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            encoder_channels: int,
            norm_layer: Literal['BatchNorm', 'InstanceNorm', 'None']='BatchNorm'
    ) -> None:
        """
        编码器类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param encoder_channels: 传递张量编码器的通道数
        :param norm_layer: 归一化层类型
        :return:
        """
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + encoder_channels, out_channels, norm_layer=norm_layer)

    def forward(self, decoder: Tensor, encoder: Tensor) -> Tensor:
        """
        前向传播
        :param decoder: 传入张量
        :param encoder: 从编码器传来用于连接的张量
        :return: 计算结果
        """
        decoder = self.up(decoder)
        x = torch.cat([encoder, decoder], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """
    输出层，包含一个卷积层
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ) -> None:
        """
        输出层类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :return:
        """
        super(OutConv, self).__init__()
        self.out = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(
        in_channels=1,
        out_channels=2,
        n_channels=[64, 128, 256, 512, 1024],
        norm_layer='None'
    ).to(device)
    print(model)

    tensor = torch.randn([8, 1, 96, 96, 96]).to(device)
    output = model(tensor)
    print(output.shape)