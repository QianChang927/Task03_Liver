from torch import nn

class VNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(VNet3D, self).__init__()