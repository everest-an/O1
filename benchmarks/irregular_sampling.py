"""
benchmarks/irregular_sampling.py — continuous-time / irregular-sampling probe.

This is the task the variable-Δt upgrade was built for. A continuous signal
x(t) = sin(2π f t + φ) is sampled at NON-uniform time points; the model sees the
observed values and the inter-sample gaps Δt, and must predict the next observed
value x(t_L) given the query gap Δt_L. Because the samples are irregular, the
phase advance between two observations depends on the real elapsed time — a model
that ignores Δt cannot know how far the signal moved.

The decisive comparison is an ABLATION on MT-LNN itself:
  * MT-LNN (Δt as input feature only, dt=None)  — the old behaviour: Δt is fed
    as a channel but the CfLTC decay still treats every step as unit-spaced.
  * MT-LNN + Δt-in-decay (dt threaded into exp(-Δt/τ)) — the new continuous-time
    path unlocked by the architecture change.
If the second beats the first, the continuous-time mechanism does real work.
Transformer / LNN (Δt as feature) are reported as references.

Usage:
    python benchmarks/irregular_sampling.py --seeds 3
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)


def make_batch(B, L, device, gen):
    """Irregular-sampling batch.

    Returns:
      feats : (B, L, 2)  channel 0 = observed value x_i, channel 1 = gap Δt_i
              (Δt_0 = 0). The gap at position i is t_i - t_{i-1}.
      dt    : (B, L)      the same gaps, to thread into the CfLTC decay.
      target: (B, 1)      x at t_{L-1}+Δt_query; the query gap is placed as the
              LAST entry of dt / feats channel-1, with value channel 0 (unknown
              future value) zeroed.
    """
    f = torch.rand(B, 1, generator=gen, device=device) * 1.5 + 0.5      # freq [0.5,2]
    phi = torch.rand(B, 1, generator=gen, device=device) * 6.2832       # phase
    gaps = torch.rand(B, L, generator=gen, device=device) * 0.45 + 0.05  # Δt [0.05,0.5]
    gaps[:, 0] = 0.0
    t = torch.cumsum(gaps, dim=1)                                        # (B,L) times
    x = torch.sin(2 * np.pi * f * t + phi)                              # (B,L) values
    # Query: predict value at the LAST time point given its gap; hide its value.
    target = x[:, -1:].clone()                                           # (B,1)
    x_in = x.clone()
    x_in[:, -1] = 0.0                                                    # unknown future value
    feats = torch.stack([x_in, gaps], dim=-1)                          # (B,L,2)
    return feats, gaps, target


class Forecaster(nn.Module):
    def __init__(self, backbone, kind, d_model, n_feat, L, use_dt_decay):
        super().__init__()
        self.backbone, self.kind = backbone, kind
        self.use_dt_decay = use_dt_decay
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        self.head = nn.Linear(d_model, 1)

    def encode(self, x, dt):
        B, L, _ = x.shape
        if self.kind == "mt":
            bb = self.backbone
            dt_arg = dt if self.use_dt_decay else None
            for block in bb.blocks:
                x, _ = block(x, layer_cache=None, pad_mask=None, position_offset=0,
                             use_cache=False, use_lnn_recurrence=True, dt=dt_arg)
            if getattr(bb, "gwtb", None) is not None:
                x, _ = bb.gwtb(x, past_kv=None, position_offset=0, use_cache=False)
            x, _ = bb.coherence(x, past_kv=None, position_offset=0, use_cache=False)
            x = bb.final_norm(x)
        else:
            bb = self.backbone
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device),
                              diagonal=1)
            for block in bb.blocks:
                x = block(x, mask)
            x = bb.final_norm(x)
        return x

    def forward(self, feats, dt):
        B, L, _ = feats.shape
        pos = torch.arange(L, device=feats.device).unsqueeze(0).expand(B, L)
        x = self.input_proj(feats) + self.pos_emb(pos)
        h = self.encode(x, dt)[:, -1, :]
        return self.head(h)                                             # (B,1)


def build(name, d_model, n_layers, n_heads, L, device):
    if name.startswith("Transformer"):
        cfg = BaselineConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model)
        return SimpleCausalTransformer(cfg).to(device), "baseline"
    if name.startswith("LNN"):
        cfg = BaselineConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model)
        return SimpleCausalLNN(cfg).to(device), "baseline"
    cfg = MTLNNConfig(vocab_size=8, max_seq_len=L, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, n_kv_heads=max(1, n_heads // 2),
                      d_head=d_model // n_heads, dropout=0.0, attention_dropout=0.0,
                      gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2)
    return MTLNNModel(cfg).to(device), "mt"


VARIANTS = [
    ("Transformer",            False),
    ("LNN",                    False),
    ("MT-LNN (Δt feature only)", False),   # dt NOT threaded into decay
    ("MT-LNN + Δt-in-decay",     True),    # dt threaded into CfLTC decay
]


def train_eval(name, use_dt_decay, L, d_model, n_layers, n_heads,
               steps, lr, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    backbone, kind = build(name, d_model, n_layers, n_heads, L, device)
    model = Forecaster(backbone, kind, d_model, 2, L, use_dt_decay).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    model.train()
    t0 = time.time()
    for step in range(steps):
        feats, dt, tgt = make_batch(128, L, device, gen)
        opt.zero_grad()
        loss = lossf(model(feats, dt), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        eval_gen = torch.Generator(device=device).manual_seed(10_000 + seed)
        se = n = 0.0
        for _ in range(20):
            feats, dt, tgt = make_batch(128, L, device, eval_gen)
            pred = model(feats, dt)
            se += ((pred - tgt) ** 2).sum().item()
            n += tgt.numel()
    return {"test_mse": se / n, "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=48)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out_json", default="benchmarks/irregular_sampling_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | irregular sampling | L={args.L} | {args.seeds} seeds")

    results = {}
    for name, use_dt in VARIANTS:
        runs = []
        for seed in range(args.seeds):
            r = train_eval(name, use_dt, args.L, args.d_model, args.n_layers,
                           args.n_heads, args.steps, args.lr, seed, device)
            runs.append(r)
            print(f"  [{name}] seed {seed}: test MSE {r['test_mse']:.4f}", flush=True)
        results[name] = runs

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    def ms(v):
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    print(f"\n### Irregular sampling (L={args.L}), {args.seeds} seeds\n")
    print("| Variant | test MSE |")
    print("|---|---|")
    for name, _ in VARIANTS:
        m, s = ms([r["test_mse"] for r in results[name]])
        print(f"| {name} | {m:.4f} ± {s:.4f} |")


if __name__ == "__main__":
    main()
