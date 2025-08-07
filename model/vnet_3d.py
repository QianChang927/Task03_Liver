import torch
from torch import nn

# 类型注解所用库
from torch import Tensor
from torch.nn import Module, ModuleList
from typing import Sequence, Literal

class VNet3D(nn.Module):
    """
    3D V-Net类，总体框架分为编码部分和解码部分
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_channels: Sequence[int]=None,
        layer_nums: Sequence[int]=None
    ) -> None:
        """
        3D V-Net类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n_channels: 中间层通道数
        :param layer_nums: 中间卷积层数
        :return:
        """
        super(VNet3D, self).__init__()
        if n_channels is None: n_channels = [16, 32, 64, 128, 256]
        if layer_nums is None: layer_nums = [1, 2, 3, 3, 3]
        assert len(n_channels) == len(layer_nums)
        assert len(n_channels) >= 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.layer_nums = layer_nums

        self.in_conv = ResidualBlock(MultiConv(self.in_channels, self.n_channels[0], self.layer_nums[0]))
        self.encoder = self.__build_layer('encoder')
        self.decoder = self.__build_layer('decoder')
        self.out_conv = nn.Sequential(
            nn.Conv3d(self.n_channels[1], self.out_channels, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
        """
        # 特征缓存
        features = []

        # 输入层
        x = self.in_conv(x)
        features.append(x)

        # 编码器
        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        # 瓶颈层处理：丢弃此层的特征避免错误残差计算
        features.pop()

        # 解码器
        for layer in self.decoder:
            x = layer(x, features.pop())

        # 输出层
        x = self.out_conv(x)

        return x

    def __build_layer(self, mode: Literal['encoder', 'decoder']) -> ModuleList:
        """
        动态生成编码器与解码器
        :return: 生成的编码器/解码器
        """
        layers = nn.ModuleList()
        if mode == 'encoder':
            n_channels = self.n_channels
            layer_nums = self.layer_nums[1:]
            info_zip = zip(n_channels[:len(layer_nums)], n_channels[-len(layer_nums):], layer_nums)
        else:
            # 以n_channels = reversed([16, 32, 64, 128, 256, 256])举例
            # in:   [256,   256,    128,    64]
            # out:  [128,   64,     32,     16]
            n_channels = list(reversed(list(self.n_channels) + [self.n_channels[-1]]))
            layer_nums = list(reversed(self.layer_nums[:-1]))
            info_zip = zip(n_channels[:len(layer_nums)], n_channels[-len(layer_nums):], layer_nums)

        for i, (in_channels, out_channels, layer_num) in enumerate(info_zip):
            if mode == 'encoder':
                layer = Encoder(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    layer_nums = layer_num
                )
            elif mode == 'decoder':
                layer = Decoder(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    encoder_channels=n_channels[i + 2],
                    layer_nums = layer_num
                )
            else:
                raise ValueError('mode should be either `encoder` or `decoder`')
            layers.append(layer)
        return layers


class MultiConv(nn.Module):
    """
    多层卷积级联类
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_nums: int
    ) -> None:
        """
        多层卷积级联类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param layer_nums: 级联卷积层数
        :return:
        """
        super(MultiConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_nums = layer_nums
        self.conv = self.__build_layers()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        for layer in self.conv:
            x = layer(x)
        return x

    def __build_layers(self) -> ModuleList:
        """
        自动生成多层卷积层，当卷积层级联层数大于1时，通道变化为：in_channels -> out_channels -> ... -> out_channels
        :return:
        """
        layers = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.out_channels
        for _ in range(self.layer_nums):
            layers.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
                nn.PReLU()
            ))
            in_channels = out_channels
        # 复现原论文的Caffe实现，最后一个卷积层不加激活，激活由外部的残差块/编码器/解码器完成
        layers[-1].pop(-1)
        return layers


class ResidualBlock(nn.Module):
    """
    残差块，将输入和通过网络的输出相加，需保证输入与输出通道数相同或输入通道数为1
    """
    def __init__(
        self,
        net: Module
    ) -> None:
        """
        残差块构造函数
        :param net: 用于计算的中间网络层
        """
        super(ResidualBlock, self).__init__()
        self.net = net
        self.activate = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.add(x, self.net(x))
        x = self.activate(x)
        return x


class Encoder(nn.Module):
    """
    编码器类
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_nums: int
    ) -> None:
        """
        编码器类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param layer_nums: 级联卷积层数
        :return:
        """
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_nums = layer_nums
        self.encoder = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = ResidualBlock(MultiConv(self.out_channels, self.out_channels, self.layer_nums))

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.encoder(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """
    解码器类，注意该层真实输出通道数为out_channels + encoder_channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: int,
        layer_nums: int
    ) -> None:
        """
        解码器类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 上采样部分的输出通道数
        :param encoder_channels: 传递张量编码器的通道数
        :param layer_nums: 级联卷积层数
        :return:
        """
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.layer_nums = layer_nums
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            nn.PReLU()
        )
        true_out_channels = self.out_channels + self.encoder_channels
        self.conv = MultiConv(true_out_channels, true_out_channels, self.layer_nums)
        self.activate = nn.PReLU()

    def forward(self, x: Tensor, encoder: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :param encoder: 从编码器传来的张量
        :return: 传出张量
        """
        x = self.decoder(x)
        x = torch.cat([encoder, x], dim=1 if len(x.shape) == 5 else 0)
        x = torch.add(x, self.conv(x))
        return self.activate(x)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vnet_3d = VNet3D(
        in_channels=1,
        out_channels=2,
        n_channels=[16, 32, 64, 128, 256],
        layer_nums=[1, 2, 3, 3, 3]
    ).to(device)
    print(vnet_3d)

    tensor: Tensor = torch.randn([8, 1, 128, 128, 64]).to(device)
    output: Tensor = vnet_3d(tensor)
    print(output.shape)