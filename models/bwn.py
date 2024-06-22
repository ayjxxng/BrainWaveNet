import torch
from omegaconf import DictConfig
from torch import nn, cfloat
from models.module import TemporalEnc, SpatialEnc, Res2DModule

device = torch.device("cuda")


class TS_Encoder(nn.Module):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super().__init__()

        self.cfg = cfg.model
        self.D = self.cfg.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.N = cfg.dataset.n_channels + 1  # 1 for SCT
        self.T = cfg.dataset.n_times

        # Linear Layer
        self.D_to_T = nn.Linear(self.D, self.T, dtype=cfloat)
        self.T_to_D = nn.Linear(self.T, self.D, dtype=cfloat)

        # Temporal Transformer Encoder (TemporalEnc)
        self.temporal_blocks = TemporalEnc(cfg)

        # Spatial Transformer Encoder (SpatialEnc)
        self.spatial_blocks = SpatialEnc(cfg)

    def forward(self, temp_emb, spat_emb):
        """
        Inputs:
            temp_emb: temporal embedding input [B, N+1, F+1, T]
            spat_emb: spatial embedding input [B, N+1, 1, D]

        Outputs:
            temp_emb: temporal embedding output [B, (N+1), F+1, T]
            spat_emb: spatial embedding output [B, (N+1), 1, D]
        """

        # Element-wise addition between SE and FCT
        temp_emb = temp_emb + \
                   nn.functional.pad(self.D_to_T(spat_emb), (0, 0, 0, self.F)).to(device)

        temp_emb = self.temporal_blocks(temp_emb)

        # FCT slicing (first raw) + back to D
        spat_emb = spat_emb + self.T_to_D(temp_emb[:, :, :1, :])

        # SpatialEnc
        spat_emb = self.spatial_blocks(spat_emb)

        return temp_emb, spat_emb


class WaveletTF(nn.Module):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super().__init__()
        self.cfg = cfg

        D = self.cfg.model.embed_dim
        F = self.cfg.dataset.n_frequencies
        N = self.cfg.dataset.n_channels
        T = self.cfg.dataset.n_times

        # FCT: Frequency Class Token
        self.fct = nn.Parameter(torch.zeros((1, N, 1, T), dtype=cfloat))

        # Frequency Positional Encoding
        self.fpe = nn.Parameter(torch.zeros((1, 1, F + 1, T), dtype=cfloat))

        # SCT: Spatial Class Token
        self.sct = nn.Parameter(torch.zeros((1, 1, 1, D), dtype=cfloat))

        # Spatial Embedding
        self.se = nn.Parameter(torch.rand((1, N, 1, D), dtype=cfloat))

        # WaveletTF blocks
        self.waveletTF_block = nn.ModuleList([
            TS_Encoder(self.cfg)
            for _ in range(self.cfg.model.n_blocks)
        ])

    def forward(
            self,
            x_real,
            x_imag
    ):
        """
        Input:
            x_real: [B, N, F, T]
            x_imag: [B, N, F, T]

        Output:
            temp_emb: [B, (N+1), 1, D]
            spec_emb: [B, (N+1), 1, D]
        """
        batch_size = len(x_real)
        x = torch.complex(x_real, x_imag).to(device)

        # Initialize temporal embedding - concat FCT (first raw) + add FPE
        fct = torch.repeat_interleave(self.fct, batch_size, 0).to(device)
        temp_emb = torch.cat([fct, x], dim=2)
        temp_emb = temp_emb + self.fpe
        temp_emb = nn.functional.pad(
            temp_emb, (0, 0, 0, 0, 1, 0))

        # Initialize spatial embedding
        spat_emb = torch.repeat_interleave(self.se, batch_size, 0).to(device)
        sct = torch.repeat_interleave(self.sct, batch_size, 0).to(device)
        spat_emb = torch.cat([sct, spat_emb], dim=1)

        # WaveletTF blocks inference
        for block in self.waveletTF_block:
            temp_emb, spat_emb = block(temp_emb, spat_emb)
        return temp_emb, spat_emb


class BrainWaveNet(nn.Module):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super().__init__()
        self.cfg = cfg

        self.D = self.cfg.model.embed_dim
        self.F = self.cfg.dataset.n_frequencies
        self.N = self.cfg.dataset.n_channels
        self.T = self.cfg.dataset.n_times

        # Residual Conv Block
        self.fe_model_real = Res2DModule(self.cfg)
        self.fe_model_imag = Res2DModule(self.cfg)

        # Main model
        self.main_model = WaveletTF(self.cfg)

        # Linear layer
        self.linear_out = nn.Linear(self.D * 2, self.cfg.dataset.n_classes)

    def forward(
            self,
            features
    ):
        """
        Input:
            features: [B, T, F, N, 2] (Real&Imag)

        Output:
            logits: [B, n_classes]
        """
        features_real = features[:, :, :, :, 0]
        features_imag = features[:, :, :, :, 1]
        # Residual Conv Block
        features_real = features_real.permute(0, 3, 2, 1)
        fe_out_real = self.fe_model_real(features_real)
        features_imag = features_imag.permute(0, 3, 2, 1)
        fe_out_imag = self.fe_model_imag(features_imag)

        # Main model
        _, spat_emb = self.main_model(fe_out_real, fe_out_imag)
        out = torch.stack([spat_emb.real[:, 0, 0, :], spat_emb.imag[:, 0, 0, :]], dim=-1)
        out = self.linear_out(out.reshape(out.size(0), -1))
        return out
