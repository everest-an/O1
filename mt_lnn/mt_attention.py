"""
mt_attention.py — Microtubule Attention with GQA + KV cache + SDPA backend.

Two microtubule-inspired modifications layered on top of GQA:

  1. Polarity bias  — each Q-head has a learned signed scalar polarity[h] ∈ [-1,1].
                       Bias[h, i_abs, j_abs] = polarity[h] * (j_abs - i_abs) / max_seq_len.
                       Computed in absolute positions so it is cache-consistent.

  2. GTP-cap log-bias — a learned per-head decay γ_h is folded in as an *additive*
                       log-bias to attention logits: -γ_h × (i_abs - j_abs).
                       This is mathematically equivalent to multiplying softmax
                       inputs by exp(-γ_h × distance) (the GTP-cap factor) but
                       composes cleanly with SDPA's attn_mask interface.

Implementation notes:
  - Uses torch.nn.functional.scaled_dot_product_attention (SDPA) so PyTorch can
    pick the Flash-Attention / memory-efficient kernel when available.
  - Causal masking is also expressed as -inf entries in the additive mask, so a
    single tensor carries causal + polarity + GTP + padding constraints.
  - Q has full head count; K/V have n_kv_heads → GQA-shrunk KV cache.

KV-cache contract:
  forward(x, past_kv=None, position_offset=0, use_cache=False)
    → returns (out, new_kv) where new_kv = (K_total, V_total).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig
from .embedding import RotaryEmbedding


def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA helper: repeat each KV head n_rep times along the head dim."""
    if n_rep == 1:
        return kv
    B, H_kv, T, D = kv.shape
    return kv[:, :, None, :, :].expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)


class MicrotubuleAttention(nn.Module):
    def __init__(self, config: MTLNNConfig, rope: RotaryEmbedding):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads    * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        # MT parameters (per Q-head)
        self.polarity_direction = nn.Parameter(torch.zeros(config.n_heads))
        # gtp_gamma stored in raw space; γ = softplus(gtp_gamma) ensures positivity
        # softplus⁻¹(γ_init) = log(exp(γ_init) - 1) ≈ log(γ_init) for small γ_init
        gamma_raw_init = math.log(math.expm1(max(config.gamma_init, 1e-4)))
        self.gtp_gamma = nn.Parameter(torch.full((config.n_heads,), gamma_raw_init))

        self.rope = rope
        self.resid_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Combined additive attention mask: causal + polarity + GTP + (padding)
    # ------------------------------------------------------------------

    def _build_attn_bias(
        self,
        q_pos: torch.Tensor,                  # (T_q,)
        k_pos: torch.Tensor,                  # (T_k,)
        pad_mask: Optional[torch.Tensor],     # (B, T_k) bool, True = valid
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns float bias of shape (B?, n_heads, T_q, T_k) suitable for SDPA.

        bias[h,i,j] = polarity[h] * (j-i)/L  -  γ[h] * (i-j)    when j ≤ i
                    = -inf                                       when j > i (causal)
                    = -inf                                       when key j is padding
        """
        T_q = q_pos.shape[0]
        T_k = k_pos.shape[0]
        H = self.n_heads
        L = float(self.max_seq_len)

        # Absolute distance i - j (positive for past keys, negative for future)
        i_abs = q_pos[:, None].float()                              # (T_q, 1)
        j_abs = k_pos[None, :].float()                              # (1, T_k)
        delta = i_abs - j_abs                                       # (T_q, T_k); ≥0 valid, <0 future

        # Polarity bias: polarity * (j-i)/L = -polarity * delta/L
        pol = self.polarity_direction.clamp(-1.0, 1.0)              # (H,)
        polarity_bias = -pol.view(H, 1, 1) * (delta / L).unsqueeze(0)  # (H, T_q, T_k)

        # GTP log-bias: -γ * (i - j)  on valid (j ≤ i) positions
        gamma = F.softplus(self.gtp_gamma).clamp(min=1e-6)          # (H,)
        gtp_log_bias = -gamma.view(H, 1, 1) * delta.clamp(min=0.0).unsqueeze(0)  # (H, T_q, T_k)

        bias = (polarity_bias + gtp_log_bias).to(dtype=dtype)        # (H, T_q, T_k)

        # Causal mask: j > i (delta < 0) → -inf
        causal_invalid = (delta < 0).unsqueeze(0)                    # (1, T_q, T_k)
        neg_inf = torch.finfo(dtype).min
        bias = bias.masked_fill(causal_invalid.expand(H, -1, -1), neg_inf)

        # Padding mask
        if pad_mask is not None:
            # pad_mask: (B, T_k); broadcast to (B, 1, 1, T_k)
            B = pad_mask.shape[0]
            bias = bias.unsqueeze(0).expand(B, -1, -1, -1).clone()   # (B, H, T_q, T_k)
            invalid = (~pad_mask)[:, None, None, :].expand(B, H, T_q, T_k)
            bias = bias.masked_fill(invalid, neg_inf)
        # else bias stays (H, T_q, T_k); SDPA broadcasts over batch.

        return bias

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                              # (B, T_new, d_model)
        pad_mask: Optional[torch.Tensor] = None,      # (B, T_total) bool
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T_new, _ = x.shape
        H_q, H_kv, D = self.n_heads, self.n_kv_heads, self.d_head

        # Project Q (n_heads), K/V (n_kv_heads) — GQA
        Q = self.q_proj(x).view(B, T_new, H_q,  D).transpose(1, 2)   # (B,H_q,T_new,D)
        K = self.k_proj(x).view(B, T_new, H_kv, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T_new, H_kv, D).transpose(1, 2)

        # RoPE at absolute positions
        Q = self.rope(Q, position_offset=position_offset)
        K = self.rope(K, position_offset=position_offset)

        # Concatenate KV cache
        if past_kv is not None:
            K_total = torch.cat([past_kv[0], K], dim=2)
            V_total = torch.cat([past_kv[1], V], dim=2)
        else:
            K_total, V_total = K, V

        T_total = K_total.shape[2]
        new_kv = (K_total, V_total) if use_cache else None

        # GQA: replicate K/V heads for the matmul
        K_rep = repeat_kv(K_total, self.n_rep)                        # (B,H_q,T_total,D)
        V_rep = repeat_kv(V_total, self.n_rep)

        # Build combined attention bias
        device = x.device
        q_pos = torch.arange(position_offset, position_offset + T_new, device=device)
        k_pos = torch.arange(0, T_total, device=device)
        attn_bias = self._build_attn_bias(q_pos, k_pos, pad_mask, device, Q.dtype)

        # Flash-Attention / memory-efficient SDPA
        out = F.scaled_dot_product_attention(
            Q, K_rep, V_rep,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,                     # causal already encoded in attn_bias
        )                                         # (B,H_q,T_new,D)

        out = out.transpose(1, 2).contiguous().view(B, T_new, H_q * D)
        return self.resid_dropout(self.out_proj(out)), new_kv
