import torch
import torch.nn as nn
from blocks.conv2dBlock import ConvBlock
from blocks.nnue import NNUE


class FlatConv3x3NNUE(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_dim=2,
        dim_middle=128,
        dim_feature=32,
    ):
        super().__init__()
        self.model_name = model_name
        self.in_dim = in_dim
        self.dim_middle = dim_middle
        self.dim_feature = dim_feature
        self.conv_mappings = self._make_conv3x3_mappings(
            25, dim_middle, dim_feature)
        # value head
        self.nnue = NNUE(dim_feature)

    def _make_conv3x3_mappings(self, num_conv, dim_middle, dim_mapping):
        conv_list = nn.ModuleList()
        for _ in range(num_conv):
            conv_list.append(
                nn.Sequential(
                    ConvBlock(self.in_dim, self.dim_middle, 2),
                    ConvBlock(self.dim_middle, self.dim_middle, 2),
                    ConvBlock(self.dim_middle, self.dim_middle, 1),
                    ConvBlock(self.dim_middle, self.dim_feature, 1, 0, "none"),
                )
            )
        return conv_list

    def get_feature_sum(self, input_plane):
        _, _, H, W = input_plane.shape
        features = []
        conv_coords = [(y, x) for y in range(0, H - 2)
                       for x in range(0, W - 2)]
        for conv_idx, (y, x) in enumerate(conv_coords):
            chunk = input_plane[:, :, y: y + 3, x: x + 3]
            feat = torch.clamp(
                self.conv_mappings[conv_idx](chunk), min=-1, max=127 / 128
            )
            features.append(feat.squeeze(-1).squeeze(-1))

        feature = torch.sum(torch.stack(features), dim=0)
        return feature

    def forward(self, x):
        input_plane = x  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature = self.get_feature_sum(input_plane)

        # value head
        value = feature  # [B, dim_mapping]
        # for i, layer in enumerate(self.value_linears):
        value = torch.clamp(value, min=0, max=127 / 128)
        value = self.nnue(value)
        return value
