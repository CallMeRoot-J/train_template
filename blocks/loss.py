import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, loss_type: str):
        super().__init__()
        self.loss_type = loss_type

    def _cross_entropy_with_softlabel(
        self, input, target, reduction="mean", adjust=False, weight=None
    ):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input,
            each item must be a valid distribution: target[i, :].sum() == 1.
        :param adjust: subtract soft-label bias from the loss
        :param weight: (batch, *) same shape as input,
            if not none, a weight is specified for each loss item
        """
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        if weight is not None:
            weight = weight.view(weight.shape[0], -1)

        logprobs = F.log_softmax(input, dim=1)
        if weight is not None:
            logprobs = logprobs * weight
        batchloss = -torch.sum(target * logprobs, dim=1)

        if adjust:
            eps = 1e-8
            bias = target * torch.log(target + eps)
            if weight is not None:
                bias = bias * weight
            bias = torch.sum(bias, dim=1)
            batchloss += bias

        if reduction == "none":
            return batchloss
        elif reduction == "mean":
            return torch.mean(batchloss)
        elif reduction == "sum":
            return torch.sum(batchloss)
        else:
            assert 0, f"Unsupported reduction mode {reduction}."

    def forward(self, input, target):
        if self.loss_type == "ce":
            return self._cross_entropy_with_softlabel(input, target, adjust=True)
        elif self.loss_type == "mse":
            return F.mse_loss(input, target)
        else:
            assert f"Unsupported reduction mode {self.loss_type}."
