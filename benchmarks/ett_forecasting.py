"""
benchmarks/ett_forecasting.py — ETT long-sequence time-series forecasting.

The Electricity Transformer dataset (Zhou et al., AAAI 2021) is a standard
long-sequence forecasting benchmark. This is a natural home for a liquid /
continuous-time architecture: the task is to predict future values of a real-
valued signal from its history, exactly the regime LTC/CfC networks were
designed for.

Setup (univariate, target = oil temperature "OT"):
  * ETTh1 hourly. Standard 12/4/4-month train/val/test split (Informer proto).
  * Input length L=96, predict horizon H in {96, 192}.
  * Standardise with TRAIN statistics only. Metric: MSE / MAE on standardised OT.

All three architectures share a unified forecasting head:
  input_proj (C->d_model) + learned positional embedding -> backbone encoder
  -> last-step hidden -> linear head (d_model -> H*C).
So the ONLY thing that differs across the three rows is the sequence-mixing
backbone (Transformer attention vs CfLTC vs full MT-LNN).

Usage:
    python benchmarks/ett_forecasting.py --pred_len 96 --seeds 3
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ETTDataset(Dataset):
    """Sliding-window (input L -> predict H) over one contiguous split.

    Standardisation uses statistics passed in (computed from TRAIN only).
    """

    def __init__(self, series, L, H, mean, std):
        self.x = (series - mean) / std           # (N, C) standardised
        self.L, self.H = L, H

    def __len__(self):
        return len(self.x) - self.L - self.H + 1

    def __getitem__(self, i):
        inp = self.x[i: i + self.L]                      # (L, C)
        tgt = self.x[i + self.L: i + self.L + self.H]    # (H, C)
        return (torch.tensor(inp, dtype=torch.float32),
                torch.tensor(tgt, dtype=torch.float32))


def load_ett(csv_path, cols):
    import csv
    rows = []
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        idx = [header.index(c) for c in cols]
        for line in r:
            rows.append([float(line[j]) for j in idx])
    return np.array(rows, dtype=np.float64)          # (N, C)


# ---------------------------------------------------------------------------
# Unified forecasting wrapper (backbone-agnostic)
# ---------------------------------------------------------------------------

class Forecaster(nn.Module):
    def __init__(self, backbone, kind, d_model, n_feat, L, H):
        super().__init__()
        self.backbone = backbone
        self.kind = kind                              # "mt" | "baseline"
        self.n_feat, self.H = n_feat, H
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        self.head = nn.Linear(d_model, H * n_feat)

    def encode(self, x):                              # x: (B, L, d_model)
        B, L, _ = x.shape
        if self.kind == "mt":
            bb = self.backbone
            for block in bb.blocks:
                x, _ = block(x, layer_cache=None, pad_mask=None,
                             position_offset=0, use_cache=False,
                             use_lnn_recurrence=True)
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

    def forward(self, inp):                           # inp: (B, L, C)
        B, L, _ = inp.shape
        pos = torch.arange(L, device=inp.device).unsqueeze(0).expand(B, L)
        x = self.input_proj(inp) + self.pos_emb(pos)
        h = self.encode(x)[:, -1, :]                  # last-step summary (B, d_model)
        out = self.head(h).view(B, self.H, self.n_feat)
        return out


def build_backbone(name, d_model, n_layers, n_heads, L, device):
    """Backbone only — embeddings/lm_head are unused by the Forecaster."""
    if name == "Transformer":
        cfg = BaselineConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model)
        return SimpleCausalTransformer(cfg).to(device), "baseline"
    if name == "LNN":
        cfg = BaselineConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model)
        return SimpleCausalLNN(cfg).to(device), "baseline"
    cfg = MTLNNConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                      n_layers=n_layers, n_heads=n_heads,
                      n_kv_heads=max(1, n_heads // 2), d_head=d_model // n_heads,
                      dropout=0.0, attention_dropout=0.0,
                      gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2)
    return MTLNNModel(cfg).to(device), "mt"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se = ae = n = 0.0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        pred = model(inp)
        se += ((pred - tgt) ** 2).sum().item()
        ae += (pred - tgt).abs().sum().item()
        n += tgt.numel()
    model.train()
    return se / n, ae / n                              # MSE, MAE


def train_eval(name, series, splits, L, H, d_model, n_layers, n_heads,
               steps, lr, seed, device):
    n_feat = series.shape[1]
    tr_s, tr_e, va_e, te_e = splits
    mean = series[tr_s:tr_e].mean(0, keepdims=True)
    std = series[tr_s:tr_e].std(0, keepdims=True) + 1e-6
    train_ds = ETTDataset(series[tr_s:tr_e], L, H, mean, std)
    val_ds = ETTDataset(series[tr_e:va_e], L, H, mean, std)
    test_ds = ETTDataset(series[va_e:te_e], L, H, mean, std)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    torch.manual_seed(seed)
    backbone, kind = build_backbone(name, d_model, n_layers, n_heads, L, device)
    model = Forecaster(backbone, kind, d_model, n_feat, L, H).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    model.train()
    it = iter(train_loader)
    best_val = float("inf"); best_test = (float("inf"), float("inf"))
    t0 = time.time()
    for step in range(steps):
        try:
            inp, tgt = next(it)
        except StopIteration:
            it = iter(train_loader); inp, tgt = next(it)
        inp, tgt = inp.to(device), tgt.to(device)
        opt.zero_grad()
        loss = lossf(model(inp), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0:
            vmse, vmae = evaluate(model, val_loader, device)
            if vmse < best_val:
                best_val = vmse
                best_test = evaluate(model, test_loader, device)
    return {"test_mse": best_test[0], "test_mae": best_test[1],
            "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/ett/ETTh1.csv")
    p.add_argument("--cols", nargs="+", default=["OT"], help="target columns")
    p.add_argument("--L", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out_json", default="benchmarks/ett_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    series = load_ett(args.csv, args.cols)
    # Informer ETTh1 split: 12/4/4 months of hourly data.
    m = 30 * 24
    splits = (0, 12 * m, 12 * m + 4 * m, 12 * m + 8 * m)
    print(f"device {device} | ETT {args.csv} cols={args.cols} | "
          f"L={args.L} H={args.pred_len} | N={len(series)} | {args.seeds} seeds")

    results = {}
    for name in ["Transformer", "LNN", "MT-LNN"]:
        runs = []
        for seed in range(args.seeds):
            r = train_eval(name, series, splits, args.L, args.pred_len,
                           args.d_model, args.n_layers, args.n_heads,
                           args.steps, args.lr, seed, device)
            runs.append(r)
            print(f"  [{name}] seed {seed}: test MSE {r['test_mse']:.4f}  "
                  f"MAE {r['test_mae']:.4f}", flush=True)
        results[name] = runs

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    def ms(v):
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    print(f"\n### ETTh1 univariate (L={args.L}, H={args.pred_len}), {args.seeds} seeds\n")
    print("| Model | test MSE | test MAE |")
    print("|---|---|---|")
    for name in ["Transformer", "LNN", "MT-LNN"]:
        mmse, smse = ms([r["test_mse"] for r in results[name]])
        mmae, smae = ms([r["test_mae"] for r in results[name]])
        print(f"| {name} | {mmse:.4f} ± {smse:.4f} | {mmae:.4f} ± {smae:.4f} |")


if __name__ == "__main__":
    main()
