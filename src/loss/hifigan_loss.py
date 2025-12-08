import torch
from torch import nn


class HiFiGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        loss_d: torch.Tensor,
        loss_g: torch.Tensor,
        loss_mel: torch.Tensor,
        loss_fm: torch.Tensor,
        loss_adv: torch.Tensor,
        **batch,
    ):
        loss = loss_d + loss_g

        return {
            "loss": loss,
            "loss_d": loss_d,
            "loss_g": loss_g,
            "loss_mel": loss_mel,
            "loss_fm": loss_fm,
            "loss_adv": loss_adv,
        }
