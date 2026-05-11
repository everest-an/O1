"""
gwtb.py — Global Workspace Theory Bottleneck

Implements the explicit "compress → process → broadcast" pattern that Baars
(1988) / Dehaene (2011) Global Workspace Theory predicts is necessary for
conscious access. Unlike the GlobalCoherenceLayer's sparse top-k attention
(which is an Orch-OR-flavoured collapse mechanism), GWTB enforces a *capacity
limit* — information must squeeze through a narrow d_gw « d_model bottleneck
before it can influence later computation.

Pipeline (per token, with KV cache for autoregressive decoding):

  1. **Compression (Ignition).**
       z_t = LayerNorm(W_compress · x_t)              shape: (B, T, d_gw)

  2. **Workspace processing.**
       z'_t = SelfAttention(z_<=t)                    causal, multi-head over d_gw
     This is where information from different time steps competes for
     bottleneck capacity. Uses standard SDPA + KV cache.

  3. **Broadcast (Global Ignition).**
       Δh_t = W_broadcast · z'_t                      shape: (B, T, d_model)
       out_t = x_t + γ · Δh_t                          gated residual

The `broadcast_gate` γ is initialised small (default 0.01) so the layer
starts as near-identity and gradually learns to ignite globally.

KV cache: forward(x, past_kv, use_cache) → (out, new_kv) — same contract as
the rest of the model's cached layers.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig


class GWTBLayer(nn.Module):
    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_gw = config.d_model // config.gwtb_compression_ratio
        assert self.d_gw % config.gwtb_n_heads == 0, \
            f"gwtb d_gw={self.d_gw} not divisible by gwtb_n_heads={config.gwtb_n_heads}"

        self.n_heads = config.gwtb_n_heads
        self.d_head = self.d_gw // self.n_heads
        self.scale = math.sqrt(self.d_head)
        self.max_seq_len = config.max_seq_len

        # 1. Compression projection (d_model → d_gw)
        self.compress = nn.Linear(self.d_model, self.d_gw, bias=False)
        self.compress_norm = nn.LayerNorm(self.d_gw)

        # 2. Workspace multi-head self-attention (causal)
        self.q_proj = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.k_proj = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.v_proj = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.attn_out = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.workspace_norm = nn.LayerNorm(self.d_gw)

        # 3. Broadcast projection (d_gw → d_model) + gated residual
        self.broadcast = nn.Linear(self.d_gw, self.d_model, bias=False)
        # Tiny init: layer is near-identity until it learns to ignite.
        self.broadcast_gate = nn.Parameter(torch.tensor(float(config.gwtb_broadcast_init)))

        self.dropout = nn.Dropout(config.dropout)

        # Precomputed causal mask buffer (workspace operates at same T as model)
        causal = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len,
                                       dtype=torch.bool))
        self.register_buffer("_causal", causal, persistent=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                       # (B, T_new, d_model)
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T_new, _ = x.shape
        H, D = self.n_heads, self.d_head

        # ---- 1. Compression / Ignition ----
        z = self.compress_norm(self.compress(x))                  # (B,T_new,d_gw)

        # ---- 2. Workspace SA ----
        Q = self.q_proj(z).view(B, T_new, H, D).transpose(1, 2)    # (B,H,T_new,D)
        K = self.k_proj(z).view(B, T_new, H, D).transpose(1, 2)
        V = self.v_proj(z).view(B, T_new, H, D).transpose(1, 2)

        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)

        T_total = K.shape[2]
        new_kv = (K, V) if use_cache else None

        # Causal slice from precomputed mask: rows = q positions, cols = k positions
        causal_mask = self._causal[
            position_offset: position_offset + T_new, :T_total
        ]                                                          # (T_new, T_total)
        # SDPA expects an additive bias or boolean (True=keep). We use boolean
        # form with attn_mask: True=keep, False=mask.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),       # (1,1,T_new,T_total)
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )                                                           # (B,H,T_new,D)
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_gw)
        z_attn = self.workspace_norm(z + self.attn_out(out))       # residual inside workspace

        # ---- 3. Broadcast / Global Ignition ----
        delta = self.broadcast(z_attn)                              # (B,T_new,d_model)
        gated = self.broadcast_gate * delta
        return x + gated, new_kv
