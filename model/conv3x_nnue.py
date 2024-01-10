import torch
import torch.nn as nn
from blocks.nnue import NNUE
from blocks.conv2dBlock import ConvBlock
from blocks.res_conv1d import ResConv1Block


class Mapping(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_res_point_wise):
        """
        :param in_channel:
        :param middle_channel:
        :param out_channel:
        """
        super().__init__()
        # num_block个残差块
        self.residues = nn.Sequential(
            *[
                ResConv1Block(middle_channel, middle_channel)
                for _ in range(num_res_point_wise)
            ]
        )
        self.conv_1 = ConvBlock(in_channel, middle_channel, 3, 0)
        self.batch_norm = nn.BatchNorm2d(num_features=middle_channel)
        self.activate_function = nn.Mish(inplace=True)
        self.final_conv = nn.Conv2d(
            middle_channel, out_channel, kernel_size=1, stride=1
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.activate_function(self.batch_norm(self.conv_1(x)))
        x = self.activate_function(self.batch_norm(self.residues(x)))
        x = self.final_conv(x)
        return x


class conv3NNUE(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channel: int,
        middle_channel: int,
        out_channel: int,
        num_point_wise: int,
    ):
        super(conv3NNUE, self).__init__()
        self.model_name = model_name
        self.out_channel = out_channel
        self.mappings = nn.ModuleList(
            [
                Mapping(in_channel, middle_channel,
                        out_channel, num_point_wise)
                for _ in range(25)
            ]
        )
        self.nnue = NNUE(out_channel)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.mappings):
            k = i // 5
            v = i % 5
            features.append(layer(x[:, :, k: k + 3, v: v + 3]))
        feature = torch.stack(features)
        feature = torch.sum(feature, dim=0)
        feature = feature.flatten(2)
        feature = torch.sum(feature, dim=2)
        return self.nnue(feature)
