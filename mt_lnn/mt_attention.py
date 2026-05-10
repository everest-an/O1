"""
mt_attention.py — Microtubule Attention with GQA + KV cache.

Two microtubule-inspired modifications layered on top of GQA:

  1. Polarity bias  — each Q-head has a learned signed scalar polarity[h] ∈ [-1,1].
                       Bias[h, i_abs, j_abs] = polarity[h] * (j_abs - i_abs) / max_seq_len.
                       Computed in absolute positions so it is cache-consistent.

  2. GTP-cap gate   — exp(-γ_h × (i_abs - j_abs)) for j_abs ≤ i_abs, else 0.
                       Scores at gated-zero positions are pushed to -inf before softmax.
                       Per-head learned γ_h ∈ ℝ⁺.

KV-cache contract:
  forward(x, past_kv=None, position_offset=0)
    → returns (out, new_kv) where new_kv = (K_total, V_total) along the T axis.
  position_offset = number of tokens already in cache.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig
from .embedding import RotaryEmbedding


def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA helper: repeat each KV head n_rep times along the head dim.
    kv: (B, n_kv_heads, T, d_head) → (B, n_kv_heads * n_rep, T, d_head)
    """
    if n_rep == 1:
        return kv
    B, H_kv, T, D = kv.shape
    return kv[:, :, None, :, :].expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)


class MicrotubuleAttention(nn.Module):
    def __init__(self, config: MTLNNConfig, rope: RotaryEmbedding):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads      # GQA group size
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.scale = math.sqrt(self.d_head)

        # Q has full head count; K/V have n_kv_heads (GQA)
        self.q_proj = nn.Linear(config.d_model, config.n_heads    * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        # MT parameters (per Q-head)
        self.polarity_direction = nn.Parameter(torch.zeros(config.n_heads))
        self.gtp_gamma = nn.Parameter(torch.full((config.n_heads,), config.gamma_init))

        self.rope = rope
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Absolute-position MT masks
    # ------------------------------------------------------------------

    def _polarity_bias(
        self, q_pos: torch.Tensor, k_pos: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        q_pos: (T_q,) absolute positions of queries
        k_pos: (T_k,) absolute positions of keys (cached + new)
        Returns: (n_heads, T_q, T_k) additive bias
        """
        # signed normalised distance: (j_abs - i_abs) / max_seq_len
        dist = (k_pos[None, :] - q_pos[:, None]).float() / float(self.max_seq_len)  # (T_q, T_k)
        # Causal: zero out upper-triangular (j > i) positions in the bias term
        causal = (k_pos[None, :] <= q_pos[:, None]).float()                          # (T_q, T_k)
        dist = dist * causal                                                         # zero above diag
        pol = self.polarity_direction.clamp(-1.0, 1.0)                               # (H,)
        return pol.view(-1, 1, 1) * dist.unsqueeze(0)                                # (H,T_q,T_k)

    def _gtp_gate(
        self, q_pos: torch.Tensor, k_pos: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        GTP hydrolysis gate: exp(-γ_h * (i_abs - j_abs)) for j_abs ≤ i_abs, else 0.
        Returns: (n_heads, T_q, T_k)
        """
        dist = (q_pos[:, None] - k_pos[None, :]).float()                             # (T_q, T_k)
        causal = (dist >= 0).float()                                                 # j ≤ i
        dist = dist.clamp(min=0.0)
        gamma = self.gtp_gamma.clamp(min=1e-4).view(-1, 1, 1)                        # (H,1,1)
        return torch.exp(-gamma * dist.unsqueeze(0)) * causal.unsqueeze(0)           # (H,T_q,T_k)

    # ------------------------------------------------------------------
    # Forward (with optional KV cache)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                              # (B, T_new, d_model)
        pad_mask: Optional[torch.Tensor] = None,      # (B, T_total) or None
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,                    # absolute pos of x[:,0,:]
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
          out: (B, T_new, d_model)
          new_kv: (K_total, V_total) for caching, each (B, n_kv_heads, T_total, d_head)
                  None if use_cache=False.
        """
        B, T_new, _ = x.shape
        H_q, H_kv, D = self.n_heads, self.n_kv_heads, self.d_head

        # Project Q (n_heads), K/V (n_kv_heads) — GQA
        Q = self.q_proj(x).view(B, T_new, H_q,  D).transpose(1, 2)   # (B, H_q,  T_new, D)
        K = self.k_proj(x).view(B, T_new, H_kv, D).transpose(1, 2)   # (B, H_kv, T_new, D)
        V = self.v_proj(x).view(B, T_new, H_kv, D).transpose(1, 2)   # (B, H_kv, T_new, D)

        # RoPE at absolute positions [position_offset, position_offset + T_new)
        Q = self.rope(Q, position_offset=position_offset)
        K = self.rope(K, position_offset=position_offset)

        # Concatenate cached K, V if provided
        if past_kv is not None:
            past_K, past_V = past_kv                                  # (B, H_kv, T_past, D)
            K_total = torch.cat([past_K, K], dim=2)
            V_total = torch.cat([past_V, V], dim=2)
        else:
            K_total, V_total = K, V

        T_total = K_total.shape[2]
        new_kv = (K_total, V_total) if use_cache else None

        # GQA: replicate K/V heads up to Q head count for the matmul
        K_rep = repeat_kv(K_total, self.n_rep)                        # (B, H_q, T_total, D)
        V_rep = repeat_kv(V_total, self.n_rep)

        # Attention scores
        scores = (Q @ K_rep.transpose(-2, -1)) / self.scale           # (B, H_q, T_new, T_total)

        # Absolute-position MT masks
        device = x.device
        q_pos = torch.arange(position_offset, position_offset + T_new, device=device)
        k_pos = torch.arange(0, T_total, device=device)

        polarity_bias = self._polarity_bias(q_pos, k_pos, device)     # (H_q, T_new, T_total)
        scores = scores + polarity_bias.unsqueeze(0)                  # broadcast over batch

        gtp = self._gtp_gate(q_pos, k_pos, device)                    # (H_q, T_new, T_total)
        # Zero-gate positions → -inf so they vanish in softmax
        # gtp = 0 for j_abs > i_abs (already causal) and for very far positions
        scores = scores + (1.0 - gtp.unsqueeze(0)) * (-1e9)

        # Padding mask (over total key length)
        if pad_mask is not None:
            assert pad_mask.shape[-1] == T_total, \
                f"pad_mask length {pad_mask.shape[-1]} must equal T_total={T_total}"
            scores = scores.masked_fill(~pad_mask[:, None, None, :], -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ V_rep                                            # (B, H_q, T_new, D)
        out = out.transpose(1, 2).contiguous().view(B, T_new, H_q * D)
        return self.resid_dropout(self.out_proj(out)), new_kv
