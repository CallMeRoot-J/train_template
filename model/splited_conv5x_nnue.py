from torch import nn
import torch
from blocks.circleBlock import CircleConv
from blocks.cornerBlock import CornerConv
from blocks.horseBlock import HorseConv
from blocks.res_conv1d import ResConv1Block
from blocks.nnue import NNUE
from blocks.conv2dBlock import ConvBlock


class Conv3xMapping(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_block):
        """
        :param in_channel:
        :param middle_channel:
        :param out_channel:
        """
        super().__init__()
        self.conv_3x3_in_5x5 = CircleConv(in_channel, middle_channel)
        self.activate_function = nn.Mish(inplace=True)
        self.conv = nn.Conv2d(
            middle_channel, middle_channel, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(
            middle_channel, out_channel, kernel_size=1, stride=1
        )
        self.batch_norm = nn.BatchNorm2d(num_features=middle_channel)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        conv_3x = self.activate_function(
            self.batch_norm(self.conv_3x3_in_5x5(x)))
        x = self.activate_function(self.conv(conv_3x))
        x = self.final_conv(x)
        return x


class ConnerMapping(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_block):
        """
        :param in_channel:
        :param middle_channel:
        :param out_channel:
        """
        super().__init__()
        self.corner_conv = CornerConv(in_channel, middle_channel)
        self.activate_function = nn.Mish(inplace=True)
        self.conv = nn.Conv2d(
            middle_channel, middle_channel, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(
            middle_channel, out_channel, kernel_size=1, stride=1
        )
        self.batch_norm = nn.BatchNorm2d(num_features=middle_channel)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        conv_3x = self.activate_function(self.batch_norm(self.corner_conv(x)))
        x = self.activate_function(self.conv(conv_3x))
        x = self.final_conv(x)
        return x


class HorseMapping(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_block):
        """
        :param in_channel:
        :param middle_channel:
        :param out_channel:
        """
        super().__init__()
        self.horse_conv = HorseConv(in_channel, middle_channel)
        self.activate_function = nn.Mish(inplace=True)
        self.conv = nn.Conv2d(
            middle_channel, middle_channel, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(
            middle_channel, out_channel, kernel_size=1, stride=1
        )
        self.batch_norm = nn.BatchNorm2d(num_features=middle_channel)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        conv_3x = self.activate_function(self.batch_norm(self.horse_conv(x)))
        x = self.activate_function(self.conv(conv_3x))
        x = self.final_conv(x)
        return x


class Mapping(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_block):
        """
        :param in_channel:
        :param middle_channel:
        :param out_channel:
        """
        super().__init__()
        self.out_channel = out_channel
        self.conv_3x3_in_5x5 = nn.ModuleList(
            [
                Conv3xMapping(in_channel, middle_channel,
                              out_channel, num_block)
                for _ in range(9)
            ]
        )
        self.corner_conv = nn.ModuleList(
            [
                ConnerMapping(in_channel, middle_channel,
                              out_channel, num_block)
                for _ in range(9)
            ]
        )
        self.horse_conv = nn.ModuleList(
            [
                HorseMapping(in_channel, middle_channel,
                             out_channel, num_block)
                for _ in range(9)
            ]
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        copy_feature = []
        jump_feature = []
        for i, layer in enumerate(self.conv_3x3_in_5x5):
            k = i // 3
            v = i % 3
            copy_feature.append(layer(x[:, :, k: k + 5, v: v + 5]))
        for i, layer in enumerate(self.corner_conv):
            k = i // 3
            v = i % 3
            jump_feature.append(layer(x[:, :, k: k + 5, v: v + 5]))
        for i, layer in enumerate(self.horse_conv):
            k = i // 3
            v = i % 3
            jump_feature.append(layer(x[:, :, k: k + 5, v: v + 5]))
        copy_features = torch.sum(torch.stack(copy_feature), dim=0)
        jump_features = torch.sum(torch.stack(jump_feature), dim=0)
        features = copy_features * jump_features
        features = features.flatten(2)
        features = torch.sum(features, dim=2)
        return features


class SplitedConv5NNUE(nn.Module):
    def __init__(
        self, model_name: str, in_channel: int, middle_channel: int, out_channel: int, num_blocks: int
    ):
        super(SplitedConv5NNUE, self).__init__()
        self.model_name = model_name
        self.out_channel = out_channel
        self.mapping = Mapping(in_channel, middle_channel,
                               out_channel, num_blocks)
        self.nnue = NNUE(out_channel)

    def forward(self, x):
        features = self.mapping(x)
        features = torch.clamp(features, min=0, max=127 / 128)
        return self.nnue(features)
