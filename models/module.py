import torch
from torch import nn, Tensor, cfloat
import torch.nn.functional as F
from omegaconf import DictConfig

device = torch.device("cuda")


class Res2DModule(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.in_channels = cfg.model.in_channels
        self.out_channels = cfg.model.out_channels
        self.batch_norm = cfg.model.batch_norm

        self.conv_1 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.relu = nn.ReLU()

        if self.batch_norm:
            self.bn_1 = nn.BatchNorm2d(self.out_channels)
            self.bn_2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(x)
        if self.batch_norm:
            out = self.bn_1(out)
        out = self.conv_2(self.relu(out))
        if self.batch_norm:
            out = self.bn_2(out)
        return x + out


class TemporalEnc(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg.model
        self.D = self.cfg.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.N = cfg.dataset.n_channels + 1  # 1 for SCT
        self.T = cfg.dataset.n_times
        self.temporal_dimff = self.cfg.temporal_dmodel * self.cfg.temporal_dim_factor

        # Temporal Transformer Encoder (TemporalEnc)
        self.temporal_linear_in = nn.Linear(self.F + 1, self.cfg.temporal_dmodel, dtype=cfloat)
        self.temporal_encoder_layers_real = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.cfg.temporal_dmodel, nhead=self.cfg.temporal_nheads, dim_feedforward=self.temporal_dimff,
                dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True
            ) for _ in range(self.cfg.n_temporal_blocks)
        ])
        self.temporal_encoder_layers_imag = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.cfg.temporal_dmodel, nhead=self.cfg.temporal_nheads, dim_feedforward=self.temporal_dimff,
                dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True
            ) for _ in range(self.cfg.n_temporal_blocks)
        ])
        self.temporal_linear_out = nn.Linear(self.cfg.temporal_dmodel, self.F + 1, dtype=cfloat)

    def forward(self, temp_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_in: Temporal embedding input of shape [B, N+1, F+1, T]

        Returns:
            temp_out: Temporal embedding output of shape [B, N+1, F+1, T]
        """
        # TemporalEnc in
        temp_in = temp_in.flatten(0, 1).transpose(1, 2)
        temp_emb = self.temporal_linear_in(temp_in)

        # TemporalEnc
        temp_emb_real, temp_emb_imag = temp_emb.real, temp_emb.imag
        for layer_real, layer_imag in zip(self.temporal_encoder_layers_real, self.temporal_encoder_layers_imag):
            temp_emb_real = layer_real(temp_emb_real)
            temp_emb_imag = layer_imag(temp_emb_imag)
        temp_emb = torch.complex(temp_emb_real, temp_emb_imag)

        # TemporalEnc out
        temp_out = self.temporal_linear_out(temp_emb)
        temp_out = temp_out.view(-1, self.N, self.T, self.F + 1).transpose(2, 3)

        return temp_out


class SpatialEnc(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg.model
        self.D = self.cfg.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.N = cfg.dataset.n_channels + 1  # 1 for SCT
        self.T = cfg.dataset.n_times
        self.spatial_dimff = self.cfg.spatial_dmodel * self.cfg.spatial_dim_factor

        # Spatial Transformer Encoder (SpatialEnc)
        self.spatial_linear_in = nn.Linear(self.D, self.cfg.spatial_dmodel, dtype=cfloat)
        self.spatial_encoder_layers_real = nn.ModuleList([
            Cross_TransformerEncoderLayer(
                d_model=self.cfg.spatial_dmodel, nhead=self.cfg.spatial_nheads, dim_feedforward=self.spatial_dimff,
                dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True
            ) for _ in range(self.cfg.n_spatial_blocks)
        ])
        self.spatial_encoder_layers_imag = nn.ModuleList([
            Cross_TransformerEncoderLayer(
                d_model=self.cfg.spatial_dmodel, nhead=self.cfg.spatial_nheads, dim_feedforward=self.spatial_dimff,
                dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True
            ) for _ in range(self.cfg.n_spatial_blocks)
        ])
        self.spatial_linear_out = nn.Linear(self.cfg.spatial_dmodel, self.D, dtype=cfloat, device=device)

    def forward(self, spat_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spat_in: Spatial embedding input of shape [B, N+1, 1, D]

        Returns:
            spat_out: Spatial embedding output of shape [B, N+1, 1, D]
        """
        # SpatialEnc in
        spat_in = spat_in.permute(0, 2, 1, 3).flatten(0, 1)
        spat_emb = self.spatial_linear_in(spat_in)

        # SpatialEnc
        spat_enc_out_real = spat_emb.imag
        spat_enc_out_imag = spat_emb.real
        for layer_real, layer_imag in zip(self.spatial_encoder_layers_real, self.spatial_encoder_layers_imag):
            spat_enc_out_real = layer_real(spat_enc_out_real, spat_emb.real, spat_emb.real)
            spat_enc_out_imag = layer_imag(spat_enc_out_imag, spat_emb.imag, spat_emb.imag)

        # SpatialEnc out
        spat_enc_out = torch.complex(spat_enc_out_real, spat_enc_out_imag)
        spat_out = self.spatial_linear_out(spat_enc_out)
        spat_out = spat_out.unsqueeze(1).permute(0, 2, 1, 3)

        return spat_out


class Cross_TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
            self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
            activation=F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False,
            norm_first: bool = False, bias: bool = True, device=None, dtype=None
    ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               bias=bias, batch_first=batch_first,
                                               **factory_kwargs)

    def forward(self, q, k, v, is_causal: bool = False):
        q, k, v = map(self.norm1, (q, k, v))
        x = v + self._sa_block(q, k, v)
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(self, q, k, v,
                  attn_mask=None, key_padding_mask=None, is_causal: bool = False) -> Tensor:
        x = self.self_attn(q, k, v,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)
