"""
benchmarks/baselines.py — Reference architectures for head-to-head comparison.

Two baselines, parameter-matched to a target budget (~200K) so the comparison
against MT-LNN on Selective Copy is genuinely apples-to-apples:

  - SimpleCausalTransformer: vanilla pre-norm Transformer
        Embedding + learned absolute PE → N × (LN + MHA + LN + FFN) → LN → lm_head
    No tricks. The standard "mainstream small LM" baseline.

  - SimpleCausalLNN:        same backbone but the FFN is replaced by a closed-form
                              LTC layer (no 13-protofilament split, no GWTB, no
                              microtubule attention) — isolates the *liquid*
                              contribution from the *microtubule* contribution.

Both expose the same `model(input_ids, labels=...)` interface as MTLNNModel so
they can drop into the existing Selective-Copy training loop unchanged.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    vocab_size: int = 16
    max_seq_len: int = 37
    d_model: int = 104
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256        # tuned so total params ≈ MT-LNN's ~200K
    dropout: float = 0.0
    tie_embeddings: bool = True


# ---------------------------------------------------------------------------
# Vanilla causal Transformer
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False,
                                 is_causal=True)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


def _gpt_init(module: nn.Module):
    """GPT-2-style init: matters for cold-start LM loss to be near log(vocab)."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SimpleCausalTransformer(nn.Module):
    """Pre-norm causal Transformer with learned absolute position embeddings."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(_TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        # GPT-2 init so initial cross-entropy ≈ log(vocab_size) at step 0
        self.apply(_gpt_init)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False, **_) -> dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1,
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if use_cache:
            # No cache for the baseline; return a dummy structure so the
            # greedy decoder in evaluate_selective_copy can pass cache around
            # without crashing.
            out["cache"] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()
                    if p is not self.lm_head.weight or not self.cfg.tie_embeddings)


# ---------------------------------------------------------------------------
# Vanilla LNN (closed-form LTC FFN, no microtubule structure)
# ---------------------------------------------------------------------------

class _LTCFFN(nn.Module):
    """Single-channel closed-form LTC — the LNN paper's basic unit, no MT.

    Parallel-mode (h_prev=0 each step) for clean comparison.
    """

    def __init__(self, d_model: int, tau_init: float = 1.0,
                 tau_min: float = 0.01, tau_max: float = 10.0, dt: float = 1.0):
        super().__init__()
        import math
        self.tau_min, self.tau_max, self.dt = tau_min, tau_max, dt
        self.W_in = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        # log_tau init same as MT-LNN's per-protofilament LTC
        log_tau_init = math.log(math.expm1(max(tau_init - tau_min, 1e-6)))
        self.log_tau = nn.Parameter(torch.full((), log_tau_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + self.tau_min
        tau = tau.clamp(self.tau_min, self.tau_max)
        decay = torch.exp(-self.dt / tau)
        A = torch.sigmoid(self.W_in(x))
        h = A * (1.0 - decay)   # parallel mode: h_prev = 0
        return self.W_out(h)


class _LNNBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ltc = _LTCFFN(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False,
                                 is_causal=True)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ltc(self.norm2(x)))
        return x


class SimpleCausalLNN(nn.Module):
    """Same as the Transformer baseline but with the FFN replaced by closed-form LTC.
    Isolates the *liquid* contribution from the *microtubule* contribution.
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(_LNNBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(_gpt_init)

    def forward(self, input_ids, labels=None, use_cache=False, **_):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1,
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if use_cache:
            out["cache"] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()
                    if p is not self.lm_head.weight or not self.cfg.tie_embeddings)
