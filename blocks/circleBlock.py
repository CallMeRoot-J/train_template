import torch
from torch import nn
import torch.nn.functional as F


class CircleConv(nn.Module):
    """
    阶梯卷积层
    """

    def __init__(self, in_channel: int, out_channel: int):
        """
        初始化
        :param kernel_size: 卷积核大小
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((9, out_channel, in_channel)))
        self.bias = nn.Parameter(torch.zeros((out_channel,)))
        nn.init.kaiming_normal_(self.weight)

    def _conv2d_3x_in_5x(self, x: torch.Tensor):
        """
        阶梯形卷积核，参与卷积
        :param x:
        :return:
        """
        w_num, out_channel, in_channel = self.weight.shape
        zero = torch.zeros(
            (out_channel, in_channel),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        kernel_5x5 = [
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            self.weight[0],
            self.weight[1],
            self.weight[2],
            zero,
            zero,
            self.weight[3],
            self.weight[4],
            self.weight[5],
            zero,
            zero,
            self.weight[6],
            self.weight[7],
            self.weight[8],
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        ]
        weight = torch.stack(kernel_5x5, dim=2)
        weight = weight.reshape(out_channel, in_channel, 5, 5)
        return F.conv2d(x, weight, self.bias, padding=0)

    def forward(self, x: torch.Tensor):
        """
        前馈
        :param x:
        :return:
        """
        return self._conv2d_3x_in_5x(x)
