from typing import Tuple
import torch
from torch import nn, cfloat
from omegaconf import DictConfig
from models.module import TemporalEnc, SpatialEnc, Res2DModule, ComplexLinear

device = torch.device("cuda")


class TS_Encoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.D = cfg.model.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.N = cfg.dataset.n_channels + 1  # 1 for SCT
        self.T = cfg.dataset.n_times

        # Linear Layers
        self.D_to_T = ComplexLinear(self.D, self.T)
        self.T_to_D = ComplexLinear(self.T, self.D)

        # Temporal Transformer Encoder (TemporalEnc)
        self.temporal_blocks = TemporalEnc(cfg)

        # Spatial Transformer Encoder (SpatialEnc)
        self.spatial_blocks = SpatialEnc(cfg)

    def forward(
            self, temp_emb: torch.Tensor, spat_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            temp_emb: Temporal embedding input of shape [B, N+1, F+1, T]
            spat_emb: Spatial embedding input of shape [B, N+1, 1, D]

        Returns:
            temp_emb: Temporal embedding output of shape [B, (N+1), F+1, T]
            spat_emb: Spatial embedding output of shape [B, (N+1), 1, D]
        """
        # Element-wise addition between SE and FCT
        temp_emb = temp_emb + \
                   nn.functional.pad(self.D_to_T(spat_emb), (0, 0, 0, self.F)).to(device)

        # TemporalEnc
        temp_emb = self.temporal_blocks(temp_emb)

        # FCT slicing (first raw) + back to D
        spat_emb = spat_emb + self.T_to_D(temp_emb[:, :, :1, :])

        # SpatialEnc
        spat_emb = self.spatial_blocks(spat_emb)

        return temp_emb, spat_emb


class WaveletTF(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.D = cfg.model.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.N = cfg.dataset.n_channels
        self.T = cfg.dataset.n_times

        # FCT: Frequency Class Token
        self.fct_real = nn.Parameter(torch.zeros((1, self.N, 1, self.T)))
        self.fct_imag = nn.Parameter(torch.zeros((1, self.N, 1, self.T)))

        # Frequency Positional Encoding
        self.fpe_real = nn.Parameter(torch.zeros((1, 1, self.F + 1, self.T)))
        self.fpe_imag = nn.Parameter(torch.zeros((1, 1, self.F + 1, self.T)))

        # SCT: Spatial Class Token
        self.sct_real = nn.Parameter(torch.zeros((1, 1, 1, self.D)))
        self.sct_imag = nn.Parameter(torch.zeros((1, 1, 1, self.D)))

        # Spatial Embedding
        self.se_real = nn.Parameter(torch.rand((1, self.N, 1, self.D)))
        self.se_imag = nn.Parameter(torch.rand((1, self.N, 1, self.D)))

        # WaveletTF blocks
        self.waveletTF_block = nn.ModuleList([
            TS_Encoder(cfg)
            for _ in range(cfg.model.n_blocks)
        ])

    def forward(
            self, x_real: torch.Tensor, x_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_real: Real part of the input tensor of shape [B, N, F, T]
            x_imag: Imaginary part of the input tensor of shape [B, N, F, T]

        Returns:
            temp_emb: Temporal embedding output of shape [B, (N+1), F+1, T]
            spat_emb: Spatial embedding output of shape [B, (N+1), 1, D]
        """
        batch_size = len(x_real)
        x = torch.complex(x_real, x_imag).to(device)

        # Initialize temporal embedding - concat FCT (first raw) + add FPE
        fct = torch.repeat_interleave(
            torch.complex(self.fct_real, self.fct_imag), batch_size, 0
        ).to(device)
        temp_emb = torch.cat([fct, x], dim=2)
        temp_emb = temp_emb + torch.complex(self.fpe_real, self.fpe_imag)
        temp_emb = nn.functional.pad(temp_emb, (0, 0, 0, 0, 1, 0))

        # Initialize spatial embedding
        spat_emb = torch.repeat_interleave(
            torch.complex(self.se_real, self.se_imag), batch_size, 0
        ).to(device) 
        sct = torch.repeat_interleave(
            torch.complex(self.sct_real, self.sct_imag), batch_size, 0
        ).to(device)
        spat_emb = torch.cat([sct, spat_emb], dim=1)

        # WaveletTF blocks inference
        for block in self.waveletTF_block:
            temp_emb, spat_emb = block(temp_emb, spat_emb)

        return temp_emb, spat_emb


class BrainWaveNet(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.D = cfg.model.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.N = cfg.dataset.n_channels
        self.T = cfg.dataset.n_times
        self.n_classes = cfg.dataset.n_classes

        # Residual Conv Blocks
        self.fe_model_real = Res2DModule(cfg)
        self.fe_model_imag = Res2DModule(cfg)

        # Main model
        self.main_model = WaveletTF(cfg)

        # Linear layer
        self.linear_out = nn.Linear(self.D * 2, self.n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input tensor of shape [B, T, F, N, 2] (Real & Imag)

        Returns:
            out: Logits of shape [B, n_classes]
        """
        features_real = features[:, :, :, :, 0].permute(0, 3, 2, 1)
        features_imag = features[:, :, :, :, 1].permute(0, 3, 2, 1)

        # Residual Conv Blocks
        fe_out_real = self.fe_model_real(features_real)
        fe_out_imag = self.fe_model_imag(features_imag)

        # Main model
        _, spat_emb = self.main_model(fe_out_real, fe_out_imag)
        out = torch.stack([spat_emb.real[:, 0, 0, :], spat_emb.imag[:, 0, 0, :]], dim=-1)
        out = self.linear_out(out.reshape(out.size(0), -1))

        return out
