from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    卷积块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding=0,
        activate="mish",
    ):
        """
        初始化模块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param padding: 填充
        """
        super().__init__()
        # 卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act = activate
        # 标准化,将输出的均值控制为0，方差控制为1
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.Mish(True)

    def forward(self, x):
        """
        前馈
        :param x: state
        :return: 卷积后处理后的state
        """
        if self.act == "none":
            return self.batch_norm(self.conv(x))
        if self.act == "relu":
            return F.relu(self.batch_norm(self.conv(x)))
        return self.activate(self.batch_norm(self.conv(x)))
