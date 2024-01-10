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
        self.ladder_conv_1 = LadderConv(in_channel//2, middle_channel)
        self.ladder_conv_2 = LadderConv(middle_channel, middle_channel)
        self.ladder_conv_3 = LadderConv(middle_channel, middle_channel)
        self.ladder_conv_4 = LadderConv(middle_channel, middle_channel)
        self.ladder_conv_5 = LadderConv(middle_channel, middle_channel)
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
        self.res_conv_4 = nn.Sequential(
            *[ResConv1Block(middle_channel, middle_channel) for i in range(num_block)]
        )
        self.res_conv_5 = nn.Sequential(
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
        x = self.activate_function(self.batch_norm(self.ladder_conv_3(x)))
        x = self.activate_function(self.res_conv_3(x))
        x = self.activate_function(self.batch_norm(self.ladder_conv_4(x)))
        x = self.activate_function(self.res_conv_4(x))
        x = self.activate_function(self.ladder_conv_5(x))
        x = self.activate_function(self.res_conv_5(x))
        x = self.final_conv(x)
        return x


class LadderConvNNUE(nn.Module):
    def __init__(
        self, model_name: str, in_channel: int, middle_channel: int, out_channel: int, num_block: int
    ):
        super(LadderConvNNUE, self).__init__()
        self.model_name = model_name
        self.out_channel = out_channel
        self.mapping = Mapping(
            in_channel, middle_channel, out_channel, num_block)
        self.nnue = NNUE(out_channel * 2)

    def forward(self, x):
        rot_i = [0, 2, 1, -1, 2, 0, -1, 1]
        index = [
            [0, 6, 0, 6],
            [0, 6, 0, 6],
            [0, 6, 1, 7],
            [0, 6, 1, 7],
            [1, 7, 1, 7],
            [1, 7, 1, 7],
            [1, 7, 0, 6],
            [1, 7, 0, 6],
        ]
        current_x = x[:, 0, :, :].view((x.shape[0], 1, x.shape[2], x.shape[3]))
        oppo_x = x[:, 1, :, :].view((x.shape[0], 1, x.shape[2], x.shape[3]))
        current_feature = []
        oppo_feature = []
        for i in range(8):
            current_rot_feature = torch.rot90(
                current_x[:, :, index[i][0]: index[i][1], index[i][2]: index[i][3]], rot_i[i], (2, 3))
            current_rot_feature = torch.clamp(
                self.mapping(current_rot_feature), -1, 127 / 128
            )
            current_feature.append(current_rot_feature)
            oppo_rot_feature = torch.rot90(
                oppo_x[:, :, index[i][0]: index[i][1], index[i][2]: index[i][3]], rot_i[i], (2, 3))
            oppo_rot_feature = torch.clamp(
                self.mapping(oppo_rot_feature), -1, 127 / 128
            )
            oppo_feature.append(oppo_rot_feature)
        current_feature = torch.stack(current_feature, dim=0)
        oppo_feature = torch.stack(oppo_feature, dim=0)
        current_feature = torch.sum(current_feature, dim=0)
        oppo_feature = torch.sum(oppo_feature, dim=0)
        current_feature = current_feature.view((x.shape[0], self.out_channel))
        oppo_feature = oppo_feature.view((x.shape[0], self.out_channel))
        feature = torch.cat([current_feature, oppo_feature], dim=1)
        return self.nnue(feature)
