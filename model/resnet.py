import torch
from torch import nn
from blocks.conv2dBlock import ConvBlock
from blocks.resBlock import ResBlock
from blocks.valueHead import ValueHead
from blocks.policyHead import PolicyHead


class ResNet(nn.Module):
    """
    策略价值网络。
    """

    def __init__(
        self,
        model_name: str,
        n_feature_planes: int,
        nun_block: int,
        in_channels: int,
        out_channels: int,
        is_use_gpu=True,
    ):
        """
        初始化Model
        :param n_feature_planes: board输入特征数
        :param in_channels: 残差输入通道数
        :param out_channels: 残差输出通道数
        :param is_use_gpu: 是否使用GPU
        """
        super().__init__()
        self.model_name = model_name
        self.is_use_gpu = is_use_gpu
        self.n_feature_planes = n_feature_planes
        self.device = torch.device("cuda:0" if is_use_gpu else "cpu")
        self.conv = ConvBlock(
            in_channels=n_feature_planes,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
        )
        # num_block个残差块
        self.residues = nn.Sequential(
            *[ResBlock(in_channels, out_channels) for i in range(nun_block)]
        )
        self.value_head = ValueHead(in_channels=out_channels)
        self.policy_head = PolicyHead(in_channels=in_channels)

    def forward(self, x):
        """
        前馈
        :param x:输入的state-->(N, C, H, W)
        :return: 策略和价值
        """
        x = self.conv(x)
        x = self.residues(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
