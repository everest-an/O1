"""
global_coherence.py — Global Coherence Layer (Orch-OR collapse).

Sparse top-k causal self-attention with a learned collapse gate.
Supports KV cache for streaming inference (`past_kv`).

Output: x + coherence_scale × gate × sparse_attn_out
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig


class GlobalCoherenceLayer(nn.Module):
    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.n_heads = config.coherence_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.coherence_heads
        self.sparsity = config.coherence_sparsity
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Orch-OR collapse gate parameters
        self.collapse_threshold = nn.Parameter(torch.tensor(0.5))
        self.coherence_scale = nn.Parameter(torch.tensor(0.1))

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _sparse_causal_scores(
        self, scores: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        scores: (B, H, T_q, T_k)
        Apply causal mask (using absolute positions) then keep only top-k entries per row.
        """
        # Causal mask: keep entries where k_pos[j] ≤ q_pos[i]
        causal = (k_pos[None, :] <= q_pos[:, None])                    # (T_q, T_k) bool
        scores = scores.masked_fill(~causal[None, None, :, :], -1e9)

        # Sparse top-k retention
        T_k = scores.shape[-1]
        k = max(1, int(T_k * self.sparsity))
        topk_vals, _ = torch.topk(scores, k=min(k, T_k), dim=-1)
        threshold = topk_vals[..., -1:].detach()
        scores = scores.masked_fill(scores < threshold, -1e9)
        return scores

    def forward(
        self,
        x: torch.Tensor,                                      # (B, T_new, d_model)
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T_new, _ = x.shape
        H, D = self.n_heads, self.d_head
        device = x.device

        Q = self.q_proj(x).view(B, T_new, H, D).transpose(1, 2)        # (B,H,T_new,D)
        K = self.k_proj(x).view(B, T_new, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T_new, H, D).transpose(1, 2)

        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)

        T_total = K.shape[2]
        new_kv = (K, V) if use_cache else None

        scores = (Q @ K.transpose(-2, -1)) / self.scale                # (B,H,T_new,T_total)

        q_pos = torch.arange(position_offset, position_offset + T_new, device=device)
        k_pos = torch.arange(0, T_total, device=device)
        scores = self._sparse_causal_scores(scores, q_pos, k_pos)

        # Collapse gate based on raw (pre-sparse) energy mean
        with torch.no_grad():
            raw = (Q @ K.transpose(-2, -1)) / self.scale
            causal = (k_pos[None, :] <= q_pos[:, None]).float()
            mean_energy = (raw * causal[None, None, :, :]).sum() / (causal.sum() * B * H + 1e-9)
        gate = torch.sigmoid((mean_energy - self.collapse_threshold) * 10.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V                                                  # (B,H,T_new,D)
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        out = self.out_proj(out)

        coherence_out = self.coherence_scale * gate * out
        return self.layer_norm(x + coherence_out), new_kv
