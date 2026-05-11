"""
benchmarks/selective_copy.py — Selective Copy task for MT-LNN.

Task definition (matches Mamba paper §3.2 "Selective Copying")
--------------------------------------------------------------
Each example is a sequence:

    [n n n m_1 n n m_2 n n n m_3 n n m_4 n n n ... SEP m_1 m_2 m_3 m_4]
                                                ^^^ targets

where:
  - n  ∈ {NOISE_LO, ..., NOISE_HI - 1}    random noise tokens
  - m  ∈ {0, ..., K_MEM - 1}              "memorable" tokens to be retrieved
  - SEP                                   special separator token

Memorable tokens are scattered uniformly at random in the noise prefix (in
sorted order). After SEP, the model must autoregressively produce m_1 m_2
... m_K in order. Random noise positions force the model to *select* which
inputs to remember — a vanilla feedforward layer cannot do this; only a
recurrent or attention-based architecture can.

This is exactly the task Mamba uses to show selective SSMs beat
input-invariant LTI models. It's a natural fit for MT-LNN's selectivity
claims (GWTB compression, MAP-gate stabilisation, RMC content-aware
lateral coupling all contribute).

Metrics
-------
  - cross-entropy loss on the K_MEM target positions only (label = -100
    elsewhere, ignored by the loss)
  - token-level greedy accuracy on the K_MEM positions
  - sequence-level exact match (all K_MEM tokens correct)
"""

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SelectiveCopyConfig:
    K_mem: int = 4              # number of memorable tokens to retrieve
    T_noise: int = 32           # noise sequence length
    vocab_size: int = 16        # total vocab (0..K_mem-1 memorable, K_mem..vocab-2 noise, vocab-1 SEP)
    batch: int = 16
    steps: int = 1000
    lr: float = 3e-3
    eval_batches: int = 8
    log_every: int = 100

    @property
    def T_total(self) -> int:
        # noise window + SEP + K_mem target positions
        return self.T_noise + 1 + self.K_mem

    @property
    def SEP(self) -> int:
        return self.vocab_size - 1


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def make_selective_copy_batch(cfg: SelectiveCopyConfig, B: int,
                                device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (input_ids, labels):
      input_ids: (B, T_total) — the full sequence including the target tokens
                 appended after SEP (so the model has access to them via
                 standard causal LM teacher forcing during training).
      labels:    (B, T_total) — -100 everywhere except the K_mem positions
                 that follow SEP (where the loss is applied).
    """
    K = cfg.K_mem
    T_n = cfg.T_noise
    T = cfg.T_total
    SEP = cfg.SEP
    NOISE_LO, NOISE_HI = K, cfg.vocab_size - 1   # noise tokens

    input_ids = torch.empty(B, T, dtype=torch.long, device=device)
    labels = torch.full((B, T), -100, dtype=torch.long, device=device)

    # Noise prefill — vectorised over the whole batch
    input_ids[:, :T_n] = torch.randint(NOISE_LO, NOISE_HI, (B, T_n), device=device)

    # For each row: pick K random positions and overwrite with memorable tokens
    # (sorted so retrieval order is well-defined).
    for b in range(B):
        pos = torch.randperm(T_n, device=device)[:K].sort().values   # (K,)
        mem = torch.randint(0, K, (K,), device=device)               # (K,)
        input_ids[b, pos] = mem
        input_ids[b, T_n] = SEP
        # Target tokens appear at positions T_n + 1 .. T_n + K (post-SEP)
        input_ids[b, T_n + 1: T_n + 1 + K] = mem
        # Label at position p is what we expect the model to predict next.
        # With model's standard shift: labels[p] is the target predicted from
        # logits at position p-1. So labels[T_n+1..T_n+K] = mem[0..K-1].
        labels[b, T_n + 1: T_n + 1 + K] = mem

    return input_ids, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_selective_copy(model, cfg: SelectiveCopyConfig, device: str = "cpu",
                          verbose: bool = True) -> list:
    """
    Train `model` on Selective Copy for cfg.steps optimisation steps.
    Returns a list of (step, loss, target_token_accuracy) tuples.

    The model must have config.vocab_size >= cfg.vocab_size.
    """
    from mt_lnn.utils import make_param_groups
    model.train()
    opt = torch.optim.AdamW(make_param_groups(model, cfg.lr), betas=(0.9, 0.95))

    history = []
    for step in range(cfg.steps):
        ids, labels = make_selective_copy_batch(cfg, cfg.batch, device=device)
        opt.zero_grad()
        out = model(ids, labels=labels)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            # Quick token-level accuracy estimate on this batch
            with torch.no_grad():
                shift_logits = out["logits"][:, :-1, :]
                shift_labels = labels[:, 1:]
                preds = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                acc = ((preds == shift_labels) & mask).float().sum() / mask.float().sum()
            history.append((step + 1, out["loss"].item(), acc.item()))
            if verbose:
                print(f"  step {step+1:5d}  loss {out['loss'].item():.4f}  "
                      f"tok_acc {acc.item():.3f}")

    return history


# ---------------------------------------------------------------------------
# Evaluation (greedy decoding)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_selective_copy(model, cfg: SelectiveCopyConfig, device: str = "cpu",
                              n_batches: int = None) -> dict:
    """
    Held-out greedy-decoding evaluation:
      - Run the model on the noise prefix + SEP (i.e. positions 0..T_noise).
      - Autoregressively decode K_mem tokens.
      - Compare against the true memorable tokens.

    Returns dict with token_accuracy and sequence_exact_match.
    """
    model.eval()
    n_batches = n_batches or cfg.eval_batches
    K = cfg.K_mem
    T_n = cfg.T_noise

    tot_correct_tok = 0
    tot_tok = 0
    tot_correct_seq = 0
    tot_seq = 0

    for _ in range(n_batches):
        ids, labels = make_selective_copy_batch(cfg, cfg.batch, device=device)

        # The "answer" tokens are at positions T_n+1..T_n+K in `ids` and `labels`.
        # We greedy-decode K tokens starting from the prefix [..SEP].
        prefix = ids[:, : T_n + 1]                              # (B, T_n+1) ending in SEP
        true_tokens = ids[:, T_n + 1: T_n + 1 + K]              # (B, K)

        # Use dual-cache incremental decode for speed.
        out = model(prefix, use_cache=True)
        cache = out["cache"]
        logits = out["logits"][:, -1, :]                        # (B, V)

        preds = []
        for _ in range(K):
            tok = logits.argmax(dim=-1, keepdim=True)           # (B, 1)
            preds.append(tok)
            out = model(tok, cache=cache, use_cache=True)
            cache = out["cache"]
            logits = out["logits"][:, -1, :]
        preds = torch.cat(preds, dim=1)                          # (B, K)

        tot_correct_tok += (preds == true_tokens).sum().item()
        tot_tok += preds.numel()
        tot_correct_seq += (preds == true_tokens).all(dim=-1).sum().item()
        tot_seq += preds.shape[0]

    return {
        "token_accuracy":   tot_correct_tok / max(tot_tok, 1),
        "sequence_exact":   tot_correct_seq / max(tot_seq, 1),
        "n_sequences":      tot_seq,
        "K_mem":            K,
        "T_noise":          T_n,
    }
