from torch import nn
import torch
from blocks.ladderBlock import LadderConv
from blocks.res_conv1d import ResConv1Block
from blocks.nnue import NNUE


class Mapping(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_block):
        """
        :param in_channel:
        :param middle_channel:
        :param out_channel:
        """
        super().__init__()
        self.ladder_conv_1 = LadderConv(in_channel, middle_channel)
        self.ladder_conv_2 = LadderConv(middle_channel, middle_channel)
        self.ladder_conv_3 = LadderConv(middle_channel, middle_channel)
        self.batch_norm = nn.BatchNorm2d(num_features=middle_channel)
        self.activate_function = nn.Mish(inplace=True)
        self.res_conv_1 = nn.Sequential(
            *[ResConv1Block(middle_channel, middle_channel) for i in range(num_block)]
        )
        self.res_conv_2 = nn.Sequential(
            *[ResConv1Block(middle_channel, middle_channel) for i in range(num_block)]
        )
        self.res_conv_3 = nn.Sequential(
            *[ResConv1Block(middle_channel, middle_channel) for i in range(num_block)]
        )
        self.conv = nn.Conv2d(
            middle_channel, middle_channel, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(
            middle_channel, out_channel, kernel_size=1, stride=1
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.activate_function(self.batch_norm(self.ladder_conv_1(x)))
        x = self.activate_function(self.res_conv_1(x))
        x = self.activate_function(self.batch_norm(self.ladder_conv_2(x)))
        x = self.activate_function(self.res_conv_2(x))
        x = self.activate_function(self.ladder_conv_3(x))
        x = self.activate_function(self.res_conv_3(x))
        x = self.final_conv(x)
        return x


class LadderConvNNUE(nn.Module):
    def __init__(
        self, model_name: str, in_channel: int, middle_channel: int, out_channel: int, num_blocks: int
    ):
        super(LadderConvNNUE, self).__init__()
        self.model_name = model_name
        self.out_channel = out_channel
        self.mapping = Mapping(in_channel, middle_channel,
                               out_channel, num_blocks)
        self.nnue = NNUE(out_channel)

    def forward(self, x):
        rot_i = [0, 2, 1, -1, 2, 0, -1, 1]
        index = [
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            [0, 4, 2, 6],
            [0, 4, 2, 6],
            [2, 6, 2, 6],
            [2, 6, 2, 6],
            [2, 6, 0, 4],
            [2, 6, 0, 4],
        ]
        feature = []
        for i in range(8):
            split_feature = x[
                :, :, index[i][0]: index[i][1]:, index[i][2]: index[i][3]
            ]
            split_feature = torch.rot90(split_feature, rot_i[i], (2, 3))
            split_feature = torch.clamp(
                self.mapping(split_feature), -1, 127 / 128)
            feature.append(split_feature)
        feature = torch.stack(feature, dim=0)
        feature = torch.sum(feature, dim=0)
        feature = feature.view((x.shape[0], self.out_channel))
        return self.nnue(feature)
