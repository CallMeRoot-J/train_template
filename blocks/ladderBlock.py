import torch
from torch import nn
import torch.nn.functional as F


class LadderConv(nn.Module):
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
        self.weight = nn.Parameter(torch.empty((3, out_channel, in_channel)))
        self.bias = nn.Parameter(torch.zeros((out_channel, )))
        nn.init.kaiming_normal_(self.weight)

    def _conv2d_ladder(self, x: torch.Tensor):
        """
        阶梯形卷积核，参与卷积
        :param x:
        :return:
        """
        w_num, out_channel, in_channel = self.weight.shape
        zero = torch.zeros((out_channel, in_channel),
                           dtype=self.weight.dtype, device=self.weight.device)
        kernel_2x2 = [
            self.weight[0], zero,
            self.weight[1], self.weight[2],
        ]
        weight = torch.stack(kernel_2x2, dim=2)
        weight = weight.reshape(out_channel, in_channel, 2, 2)
        return F.conv2d(x, weight, self.bias, padding=0)

    def forward(self, x: torch.Tensor):
        """
        前馈
        :param x:
        :return:
        """
        return self._conv2d_ladder(x)
