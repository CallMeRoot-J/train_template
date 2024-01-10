import numpy as np
import torch
from torch.utils.data import Dataset


class BoardGameDataSet(Dataset):
    def __init__(self, data, use_policy_target, use_global_feature, use_value_target, use_draw) -> None:
        super().__init__()
        self.use_pt = use_policy_target
        self.use_gf = use_global_feature
        self.use_vt = use_value_target
        self.use_draw = use_draw
        self.bf = torch.FloatTensor(data["bf"])
        if self.use_pt:
            self.pt = torch.FloatTensor(data['pt'])
        if self.use_vt:
            self.vt = torch.FloatTensor(data["vt"])
        if self.use_gf:
            self.gf = torch.FloatTensor(data['gf'])
        else:
            assert "Invalid dataset's feature type."

    def __len__(self):
        return self.bf.shape[0]

    def data_strong_with_pt(self, bf, pt, gf):
        k = np.random.randint(0, 8)
        if k < 4:
            bf = torch.rot90(bf, k, (1, 2))
            pt = torch.rot90(pt, k, (0, 1))
        else:
            k -= 4
            bf = torch.flip(torch.rot90(bf, k, (1, 2)), (1, 2))
            pt = torch.flip(torch.rot90(pt, k, (0, 1)), (0, 1))
        if self.use_gf:
            bf = torch.cat((bf, gf.view((gf.shape[0], 1, 1)) *
                            torch.ones(size=(gf.shape[0], bf.shape[-2], bf.shape[-1]))), dim=0)
        pt = pt.flatten()
        return bf, pt

    def data_strong_without_pt(self, bf, gf):
        k = np.random.randint(0, 8)
        if k < 4:
            bf = torch.rot90(bf, k, (1, 2))
        else:
            k -= 4
            bf = torch.flip(torch.rot90(bf, k, (1, 2)), (1, 2))
        if self.use_gf:
            bf = torch.cat((bf, gf.view((gf.shape[0], 1, 1)) *
                            torch.ones(size=(gf.shape[0], bf.shape[-2], bf.shape[-1]))), dim=0)
        pt = pt.flatten()
        return bf

    def _get_value_no_draw_labels(self, value):
        return [value[0] - value[1]]

    def __getitem__(self, index):
        vt = self.vt[index]
        if self.use_draw:
            vt = self._get_value_no_draw_labels(self.vt[index])
        if self.use_pt:
            bf, pt = self.data_strong_with_pt(
                self.bf[index], self.pt[index], self.gf[index])
            return bf, pt, vt
        else:
            bf = self.data_strong_with_pt(self.bf[index], self.gf[index])
            return bf, vt
