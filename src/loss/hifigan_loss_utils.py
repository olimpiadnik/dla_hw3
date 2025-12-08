from typing import List

import torch
import torch.nn.functional as F


def discriminator_loss(
    real_scores: List[torch.Tensor],
    fake_scores: List[torch.Tensor],
) -> torch.Tensor:
    loss = 0.0
    for dr, df in zip(real_scores, fake_scores):
        loss_real = torch.mean((dr - 1.0) ** 2)
        loss_fake = torch.mean((df - 0.0) ** 2)
        loss = loss + loss_real + loss_fake
    return loss


def generator_adv_loss(
    fake_scores: List[torch.Tensor],
) -> torch.Tensor:
    loss = 0.0
    for df in fake_scores:
        loss = loss + torch.mean((df - 1.0) ** 2)
    return loss


def feature_matching_loss(
    fmap_real: List[List[torch.Tensor]],
    fmap_fake: List[List[torch.Tensor]],
) -> torch.Tensor:
    loss = 0.0
    for real_disc_fmaps, fake_disc_fmaps in zip(fmap_real, fmap_fake):
        assert len(real_disc_fmaps) == len(fake_disc_fmaps)
        for fr, ff in zip(real_disc_fmaps, fake_disc_fmaps):
            loss = loss + torch.mean(torch.abs(fr - ff))
    return loss


def mel_l1_loss(
    mel_real: torch.Tensor,
    mel_fake: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(torch.abs(mel_real - mel_fake))
