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

        # Optional low-rank bilinear polarity bias.
        # bias = σ((x W_A) (x W_B)^T) — a content-aware, T×T mask that
        # generalises across sequence lengths (no fixed T parameter). Models
        # α/β-tubulin dimer pair interactions.
        self.polarity_mode = config.polarity_mode
        if config.polarity_mode == "low_rank":
            r = config.polarity_rank
            self.pol_W_A = nn.Linear(config.d_model, r, bias=False)
            self.pol_W_B = nn.Linear(config.d_model, r, bias=False)
            # Per-head mixing weight in [0, 1]: learns how much of the bilinear
            # mask to apply, vs. the scalar polarity bias. Init so bilinear is
            # initially disabled (sigmoid(-3) ≈ 0.05) — model can rely on the
            # scalar bias first.
            self.pol_bilinear_gate = nn.Parameter(torch.full((config.n_heads,), -3.0))

        # gtp_gamma stored in raw space; γ = softplus(gtp_gamma) > 0.
        # **ALiBi-style multi-scale init**: heads span a geometric sequence so
        # different heads see different effective receptive fields — some focus
        # locally (large γ), others span far context (small γ).
        # Geometric ratio chosen so head 0 has γ ≈ gamma_init * 8 (strongly local)
        # and head (H-1) has γ ≈ gamma_init / 8 (effectively global).
        gamma_targets = self._build_alibi_gamma(config.n_heads, config.gamma_init)
        raw_init = torch.log(torch.expm1(gamma_targets.clamp(min=1e-4)))
        self.gtp_gamma = nn.Parameter(raw_init)

        self.rope = rope
        self.resid_dropout = nn.Dropout(config.dropout)

        # ------------------------------------------------------------------
        # Precomputed distance matrices (saved as buffers, not parameters).
        # _delta[i, j]   = i - j           (positive for past keys)
        # _causal[i, j]  = True            iff j ≤ i
        # We slice these by [position_offset:position_offset+T_new, :T_total]
        # in the forward path so no fresh arange / tensor allocations occur.
        # ------------------------------------------------------------------
        idx = torch.arange(config.max_seq_len)
        delta = idx[:, None] - idx[None, :]                  # (L, L) signed int
        self.register_buffer("_delta", delta.float(), persistent=False)
        self.register_buffer("_causal", (delta >= 0), persistent=False)

    @staticmethod
    def _build_alibi_gamma(n_heads: int, base_gamma: float) -> torch.Tensor:
        """
        Geometric γ schedule: γ_h = base_gamma * 2^(slope_h), where slope_h
        ranges from +3 (head 0 → very local) down to -3 (head H-1 → very global).
        For n_heads = 16 this gives γ ∈ [base/8, base*8] — a 64× spread.
        """
        if n_heads == 1:
            return torch.tensor([base_gamma])
        slopes = torch.linspace(3.0, -3.0, n_heads)
        return base_gamma * (2.0 ** slopes)

    # ------------------------------------------------------------------
    # Combined additive attention mask: causal + polarity + GTP + (padding)
    # ------------------------------------------------------------------

    def _build_attn_bias(
        self,
        x_q: torch.Tensor,                    # (B, T_q, d_model) — new tokens only
        x_kv: Optional[torch.Tensor],         # (B, T_k, d_model) or None (use x_q)
        q_start: int,
        k_len: int,
        pad_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Combined SDPA-compatible attention bias:

          bias[h,i,j] = polarity[h] * (j-i)/L                           (scalar polarity)
                     + α_h * σ((x_q W_A)(x_kv W_B)^T)                   (low-rank bilinear, if enabled)
                     - γ[h] * (i-j)                                     (GTP log-decay)
                     +∞-mask-out for j > i (causal) or pad.

        Note: low-rank mode is *content-aware* — depends on x. With KV caching
        we'd need the full (cached + new) x sequence to compute it. For now
        the bilinear bias is computed only over the *new* tokens × new tokens
        block; old positions get scalar polarity only. This is exact when
        prefilling and a mild approximation during incremental decode.
        """
        B, T_q, _ = x_q.shape
        H = self.n_heads
        L = float(self.max_seq_len)

        # Precomputed distance / causal slices
        delta  = self._delta[q_start: q_start + T_q, :k_len]          # (T_q, T_k)
        causal = self._causal[q_start: q_start + T_q, :k_len]         # (T_q, T_k) bool

        # Scalar polarity bias
        pol = self.polarity_direction.clamp(-1.0, 1.0)                # (H,)
        polarity_bias = -pol.view(H, 1, 1) * (delta / L).unsqueeze(0) # (H,T_q,T_k)

        # GTP log-bias
        gamma = F.softplus(self.gtp_gamma).clamp(min=1e-6)            # (H,)
        gtp_log_bias = -gamma.view(H, 1, 1) * delta.clamp(min=0.0).unsqueeze(0)

        bias = (polarity_bias + gtp_log_bias).to(dtype=dtype)         # (H,T_q,T_k)
        bias = bias.unsqueeze(0).expand(B, -1, -1, -1).clone()        # (B,H,T_q,T_k)

        # Low-rank content-aware polarity bias (opt-in)
        if self.polarity_mode == "low_rank":
            xq_A = self.pol_W_A(x_q)                                  # (B,T_q,r)
            xq_B = self.pol_W_B(x_q)                                  # (B,T_q,r)
            # Bilinear: M_qq[b,i,j] = σ(xq_A[b,i] · xq_B[b,j])
            M_qq = torch.sigmoid(xq_A @ xq_B.transpose(-2, -1))       # (B,T_q,T_q)

            # Place M_qq into the rightmost T_q×T_q sub-block of the (T_q,T_k)
            # bias matrix — corresponds to attention among the new tokens only.
            full = torch.zeros(B, T_q, k_len, device=x_q.device, dtype=dtype)
            full[:, :, k_len - T_q:] = M_qq.to(dtype)
            gate = torch.sigmoid(self.pol_bilinear_gate).view(1, H, 1, 1)  # (1,H,1,1)
            bias = bias + gate * full.unsqueeze(1)                    # broadcast over H

        # Causal: invalid positions get -inf
        neg_inf = torch.finfo(dtype).min
        invalid = (~causal).unsqueeze(0).unsqueeze(0)                 # (1,1,T_q,T_k)
        bias = bias.masked_fill(invalid, neg_inf)

        if pad_mask is not None:
            invalid_pad = (~pad_mask)[:, None, None, :]
            bias = bias.masked_fill(invalid_pad, neg_inf)

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

        # Build combined attention bias (zero new arange allocations)
        attn_bias = self._build_attn_bias(
            x_q=x, x_kv=None,
            q_start=position_offset, k_len=T_total,
            pad_mask=pad_mask, dtype=Q.dtype,
        )

        # Flash-Attention / memory-efficient SDPA
        out = F.scaled_dot_product_attention(
            Q, K_rep, V_rep,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,                     # causal already encoded in attn_bias
        )                                         # (B,H_q,T_new,D)

        out = out.transpose(1, 2).contiguous().view(B, T_new, H_q * D)
        return self.resid_dropout(self.out_proj(out)), new_kv
