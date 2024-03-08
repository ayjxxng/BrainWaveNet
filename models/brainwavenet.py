import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor, cfloat


class Res2DMaxPoolModule(nn.Module):
    def __init__(self, cfg):
        super(Res2DMaxPoolModule, self).__init__()
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.pooling = cfg.pooling
        self.conv_1 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(self.out_channels)
        self.conv_2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

        self.diff = False
        if self.in_channels != self.out_channels:
            self.conv_3 = nn.Conv2d(
                self.in_channels, self.out_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(self.out_channels)
            self.diff = True

    def forward(self, x): 
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        return out 
    

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        
    def forward(self, q, k, v,
                is_causal: bool = False):

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
    

class TS_Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.D = self.cfg.embed_dim
        self.F = self.cfg.n_frequencies 
        self.K = self.cfg.n_channels 
        self.T = self.cfg.n_times 

        # TCT: Temporal Class Token
        if self.cfg.use_tct:
            self.K += 1

        # Linear Layer
        self.D_to_T = nn.Linear(self.D, self.T, dtype = cfloat)
        self.T_to_D = nn.Linear(self.T, self.D, dtype = cfloat)

        # Temporal Transformer Encoder (TemporalEnc)
        self.temporal_linear_in = nn.Linear(self.F+1, self.cfg.temporal_dmodel, dtype = cfloat)  
        self.temporal_encoder_layer_real = nn.TransformerEncoderLayer(
            d_model=self.cfg.temporal_dmodel, nhead=self.cfg.temporal_nheads, dim_feedforward=self.cfg.temporal_dimff,
            dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal_encoder_layer_imag = nn.TransformerEncoderLayer(
            d_model=self.cfg.temporal_dmodel, nhead=self.cfg.temporal_nheads, dim_feedforward=self.cfg.temporal_dimff,
            dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal_linear_out = nn.Linear(self.cfg.temporal_dmodel, self.F+1, dtype = cfloat)

        # Spatial Transformer Encoder (SpatialEnc)
        self.spatial_linear_in = nn.Linear(self.D, self.cfg.spatial_dmodel, dtype = cfloat)
        self.spatial_encoder_layer_real = TransformerEncoderLayer(
            d_model=self.cfg.spatial_dmodel, nhead=self.cfg.spatial_nheads, dim_feedforward=self.cfg.spatial_dimff,
            dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True)
        self.spatial_encoder_layer_imag = TransformerEncoderLayer(
            d_model=self.cfg.spatial_dmodel, nhead=self.cfg.spatial_nheads, dim_feedforward=self.cfg.spatial_dimff,
            dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True)
        self.spatial_linear_out = nn.Linear(self.cfg.spatial_dmodel, self.D, dtype = cfloat)


    def forward(self, temp_in, spat_in):
        """
        Inputs:
            temp_in: temporal embedding input [B, K+1, F+1, T]
            spat_in: spatial embedding input [B, K+1, 1, D]

        Outputs:
            temp_out: temporal embedding output [B, (K+1), F+1, T]
            spat_out: sptial embedding output [B, (K+1), 1, D]
        """
        # Element-wise addition between SE and FCT
        temp_in = temp_in + \
            nn.functional.pad(self.D_to_T(spat_in), (0, 0, 0, self.F))  

        # TemporalEnc in
        temp_in = temp_in.flatten(0, 1).transpose(1,2)  
        temp_emb = self.temporal_linear_in(temp_emb)
        # TemporalEnc
        temp_enc_out_real = self.temporal_encoder_layer_real(temp_emb.real) 
        temp_enc_out_imag = self.temporal_encoder_layer_imag(temp_emb.imag)  
        # TemporalEnc out
        temp_enc_out = torch.complex(temp_enc_out_real, temp_enc_out_imag) 
        temp_out = self.temporal_linear_out(temp_enc_out) 
        temp_out = temp_out.view(-1, self.K, self.T, self.F+1).transpose(2,3) 

        # FCT slicing (first raw) + back to D
        spat_in = spat_in + self.T_to_D(temp_out[:, :, :1, :])  

        # SpatialEnc in
        spat_in = spat_in.permute(0, 2, 1, 3).flatten(0, 1) 
        spat_emb = self.spatial_linear_in(spat_in)  
        # SpatialEnc
        spat_enc_out_real = self.spatial_encoder_layer_real(spat_emb.imag, spat_emb.real, spat_emb.real)  
        spat_enc_out_imag = self.spatial_encoder_layer_imag(spat_emb.real, spat_emb.imag, spat_emb.imag)  
        # SpatialEnc out
        spat_enc_out = torch.complex(spat_enc_out_real, spat_enc_out_imag) 
        spat_out = self.spatial_linear_out(spat_enc_out) 
        spat_out = spat_out.unsqueeze(1).permute(0, 2, 1, 3)  

        return temp_out, spat_out


class WaveletTF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        D = self.cfg.embed_dim 
        F = self.cfg.n_frequencies 
        K = self.cfg.n_channels 
        T = self.cfg.n_times 

        # FCT: Frequency Class Token
        self.fct = nn.Parameter(torch.zeros((1, K, 1, T), dtype = cfloat)) 

        # Frequency Positional Encoding
        self.fpe = nn.Parameter(torch.zeros((1, 1, F+1, T), dtype = cfloat))

        # TCT: Temporal Class Token
        if self.cfg.use_tct:
            self.tct = nn.Parameter(torch.zeros((1, 1, 1, D), dtype = cfloat)) 
        else:
            self.tct = None

        # Spatial Embedding
        self.se = nn.Parameter(torch.rand((1, K, 1, D), dtype = cfloat)) 

        # WaveletTF blocks
        self.waveletTF_block = nn.ModuleList([
            TS_Encoder(self.cfg)
            for _ in range(self.cfg.n_blocks) 
        ])


    def forward(self, x_real, x_imag):
        """
        Input:
            x_real: [B, K, F, T]
            x_imag: [B, K, F, T]

        Output:
            spec_emb: [B, (K+1), 1, D]
            temp_emb: [B, (K+1), 1, D]
        """
        batch_size = len(x_real)

        x = torch.complex(x_real, x_imag)

        # Initialize temporal embedding - concat FCT (first raw) + add FPE
        fct = torch.repeat_interleave(self.fct, batch_size, 0)  
        temp_emb = torch.cat([fct, x], dim=2)  
        temp_emb = temp_emb + self.fpe  
        temp_emb = nn.functional.pad(
            temp_emb, (0, 0, 0, 0, 1, 0))  

        # Initialize spatial embedding
        spat_emb = torch.repeat_interleave(self.se, batch_size, 0)  
        tct = torch.repeat_interleave(self.tct, batch_size, 0) 
        spat_emb = torch.cat([tct, spat_emb], dim=1)  

        # WaveletTF blocks inference
        for block in self.waveletTF_block: 
            temp_emb, spat_emb = block(temp_emb, spat_emb)
        return temp_emb, spat_emb 


class BrainWaveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.D = self.cfg.embed_dim
        self.F = self.cfg.n_frequencies 
        self.K = self.cfg.n_channels 
        self.T = self.cfg.n_times 

        # TCT: Temporal Class Token
        self.use_tct = self.cfg.use_tct

        # Front-end model
        self.fe_model = Res2DMaxPoolModule(self.cfg)

        # Main model
        self.main_model = WaveletTF(self.cfg)  

        # Linear layer
        self.linear_out = nn.Linear(self.cfg.embed_dim * 2, self.cfg.n_classes) 


    def forward(self, features):
        """
        Input:
            features: [B, T, F, K] 

        Output:
            logits:
                - [B, n_classes] 
        """

        features_real = features[:, :, :, :, 0] 
        features_imag = features[:, :, :, :, 1] 

        if self.cfg.front_end:
            features_real = features_real.permute(0, 3, 2, 1) 
            fe_out_real = self.fe_model(features_real)   
            features_imag = features_imag.permute(0, 3, 2, 1) 
            fe_out_imag = self.fe_model(features_imag)   
        else:
            fe_out_real = features_real.permute(0, 3, 2, 1) 
            fe_out_imag = features_imag.permute(0, 3, 2, 1) 

        # Main model
        _, spat_emb = self.main_model(fe_out_real, fe_out_imag)  
        out = torch.stack([spat_emb.real[:, 0, 0, :], spat_emb.imag[:, 0, 0, :]], dim=-1) 
        out = self.linear_out(out.reshape(out.size(0), -1))
        return out
