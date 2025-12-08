
import math
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.loss.hifigan_loss_utils import (
    discriminator_loss,
    generator_adv_loss,
    feature_matching_loss,
    mel_l1_loss,
)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: Tuple[int, int],
        bias: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        d1, d2 = dilation

        padding1 = (kernel_size * d1 - d1) // 2
        padding2 = (kernel_size * d2 - d2) // 2

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=padding1, dilation=d1, bias=bias)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=padding2, dilation=d2, bias=bias)
        )
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.activation(x)
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)

        return out + residual

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)


class MultiReceptiveFieldBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...],
        dilations: Tuple[Tuple[int, int], ...],
        bias: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    kernel_size=k,
                    dilation=d,
                    bias=bias,
                    negative_slope=negative_slope,
                )
                for k, d in zip(kernel_sizes, dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for block in self.blocks:
            out = out + block(x)
        out = out / len(self.blocks)
        return out

    def remove_weight_norm(self):
        for block in self.blocks:
            block.remove_weight_norm()


class HiFiGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, int], ...] = (
            (1, 3),
            (1, 3),
            (1, 3),
        ),
        initial_channels: int = 512,
        bias: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(in_channels, initial_channels, kernel_size=7, padding=3, bias=bias)
        )

        self.upsamples = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        channels = initial_channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        channels // 2,
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                        bias=bias,
                    )
                )
            )
            self.mrfs.append(
                MultiReceptiveFieldBlock(
                    channels=channels // 2,
                    kernel_sizes=resblock_kernel_sizes,
                    dilations=resblock_dilation_sizes,
                    bias=bias,
                    negative_slope=negative_slope,
                )
            )

            channels = channels // 2

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(channels, 1, kernel_size=7, padding=3, bias=bias)
        )

        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)

        for upsample, mrf in zip(self.upsamples, self.mrfs):
            x = self.activation(x)
            x = upsample(x)
            x = mrf(x)

        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)
        for layer in self.upsamples:
            nn.utils.remove_weight_norm(layer)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()

class SubPeriodDiscriminator(nn.Module):
    def __init__(self, period: int, negative_slope: float = 0.1):
        super().__init__()
        self.period = period

        def conv_block(in_c, out_c, k, s, p):
            return nn.utils.weight_norm(
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            )

        self.convs = nn.ModuleList(
            [
                conv_block(1, 32, (5, 1), (3, 1), (2, 0)),
                conv_block(32, 128, (5, 1), (3, 1), (2, 0)),
                conv_block(128, 512, (5, 1), (3, 1), (2, 0)),
                conv_block(512, 1024, (5, 1), (3, 1), (2, 0)),
                conv_block(1024, 1024, (5, 1), (1, 1), (2, 0)),
            ]
        )

        self.conv_post = nn.utils.weight_norm(
            nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            t = t + pad_len

        x = x.view(b, c, t // self.period, self.period)

        feature_maps: List[torch.Tensor] = []
        out = x
        for conv in self.convs:
            out = self.activation(conv(out))
            feature_maps.append(out)

        out = self.conv_post(out)
        feature_maps.append(out)

        out = out.flatten(1, -1)  # [B, *]
        return out, feature_maps

    def remove_weight_norm(self):
        for conv in self.convs:
            nn.utils.remove_weight_norm(conv)
        nn.utils.remove_weight_norm(self.conv_post)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [SubPeriodDiscriminator(p) for p in periods]
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores: List[torch.Tensor] = []
        features: List[List[torch.Tensor]] = []

        for disc in self.discriminators:
            s, f = disc(x)
            scores.append(s)
            features.append(f)

        return scores, features

    def remove_weight_norm(self):
        for d in self.discriminators:
            d.remove_weight_norm()


class SubScaleDiscriminator(nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super().__init__()

        def conv_block(in_c, out_c, k, s, p, groups=1):
            return nn.utils.weight_norm(
                nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p, groups=groups)
            )

        self.convs = nn.ModuleList(
            [
                conv_block(1, 16, 15, 1, 7),
                conv_block(16, 64, 41, 4, 20, groups=4),
                conv_block(64, 256, 41, 4, 20, groups=16),
                conv_block(256, 1024, 41, 4, 20, groups=64),
                conv_block(1024, 1024, 41, 4, 20, groups=256),
                conv_block(1024, 1024, 5, 1, 2),
            ]
        )

        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        )

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feature_maps: List[torch.Tensor] = []
        out = x
        for conv in self.convs:
            out = self.activation(conv(out))
            feature_maps.append(out)

        out = self.conv_post(out)
        feature_maps.append(out)

        out = out.flatten(1, -1)
        return out, feature_maps

    def remove_weight_norm(self):
        for conv in self.convs:
            nn.utils.remove_weight_norm(conv)
        nn.utils.remove_weight_norm(self.conv_post)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [SubScaleDiscriminator() for _ in range(3)]
        )
        self.avgpools = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores: List[torch.Tensor] = []
        features: List[List[torch.Tensor]] = []

        out = x
        for disc, pool in zip(self.discriminators, self.avgpools):
            out = pool(out)
            s, f = disc(out)
            scores.append(s)
            features.append(f)

        return scores, features

    def remove_weight_norm(self):
        for d in self.discriminators:
            d.remove_weight_norm()


class HiFiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> Dict[str, Any]:
        d_real_mpd, f_real_mpd = self.mpd(real)
        d_fake_mpd, f_fake_mpd = self.mpd(fake)

        d_real_msd, f_real_msd = self.msd(real)
        d_fake_msd, f_fake_msd = self.msd(fake)

        return {
            "d_real_mpd": d_real_mpd,
            "d_fake_mpd": d_fake_mpd,
            "f_real_mpd": f_real_mpd,
            "f_fake_mpd": f_fake_mpd,
            "d_real_msd": d_real_msd,
            "d_fake_msd": d_fake_msd,
            "f_real_msd": f_real_msd,
            "f_fake_msd": f_fake_msd,
        }

    def remove_weight_norm(self):
        self.mpd.remove_weight_norm()
        self.msd.remove_weight_norm()



class HiFiGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        lambda_mel: float = 45.0,
        lambda_fm: float = 2.0,
        mel_config: Optional[MelSpectrogramConfig] = None,
    ):
        super().__init__()

        self.generator = HiFiGenerator(in_channels=in_channels)
        self.discriminator = HiFiDiscriminator()

        if mel_config is None:
            mel_config = MelSpectrogramConfig()
        self.mel_extractor = MelSpectrogram(mel_config)

        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm

    def forward(self, audio: torch.Tensor, mel: torch.Tensor, **batch):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1) 
        elif audio.dim() == 3 and audio.size(1) != 1:
            audio = audio.mean(dim=1, keepdim=True)

        y_hat = self.generator(mel) 
        T_real = audio.size(-1)
        T_gen = y_hat.size(-1)
        min_len = min(T_real, T_gen)

        if T_real != T_gen:
            audio = audio[..., :min_len]
            y_hat = y_hat[..., :min_len]

        for p in self.discriminator.parameters():
            p.requires_grad = True

        y_hat_detached = y_hat.detach()

        d_out_d = self.discriminator(audio, y_hat_detached)

        d_loss_mpd = discriminator_loss(
            d_out_d["d_real_mpd"], d_out_d["d_fake_mpd"]
        )
        d_loss_msd = discriminator_loss(
            d_out_d["d_real_msd"], d_out_d["d_fake_msd"]
        )
        loss_d = d_loss_mpd + d_loss_msd

        for p in self.discriminator.parameters():
            p.requires_grad = False

        d_out_g = self.discriminator(audio, y_hat)

        g_adv_mpd = generator_adv_loss(d_out_g["d_fake_mpd"])
        g_adv_msd = generator_adv_loss(d_out_g["d_fake_msd"])
        loss_adv = g_adv_mpd + g_adv_msd

        fm_mpd = feature_matching_loss(
            d_out_g["f_real_mpd"], d_out_g["f_fake_mpd"]
        )
        fm_msd = feature_matching_loss(
            d_out_g["f_real_msd"], d_out_g["f_fake_msd"]
        )
        loss_fm = fm_mpd + fm_msd


        audio_flat = audio.squeeze(1)   
        y_hat_flat = y_hat.squeeze(1)    

        mel_real = self.mel_extractor(audio_flat)  
        mel_fake = self.mel_extractor(y_hat_flat)  
        loss_mel = mel_l1_loss(mel_real, mel_fake)

        loss_g = self.lambda_mel * loss_mel + self.lambda_fm * loss_fm + loss_adv

        for p in self.discriminator.parameters():
            p.requires_grad = True

        return {
            "audio_hat": y_hat,
            "loss_d": loss_d,
            "loss_g": loss_g,
            "loss_mel": loss_mel,
            "loss_fm": loss_fm,
            "loss_adv": loss_adv,
        }
