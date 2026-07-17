"""
benchmarks/physionet_mortality.py — PhysioNet-2012 in-hospital mortality.

The canonical irregularly-sampled clinical benchmark for continuous-time models
(LTC/CfC evaluate on exactly this). Each ICU patient is an event stream of
(Δt, variable, value) measurements over 48h; the task is to predict in-hospital
mortality (binary, ~14% positive). Because the sampling is genuinely irregular,
a model that consumes the real inter-event gaps Δt should have an edge — which
is precisely the capability the variable-Δt architecture change unlocked.

We run the decisive ablation ON MT-LNN (Δt as feature only vs Δt threaded into
the CfLTC decay) plus Transformer / LNN references, and report AUROC + AUPRC
(the standard metrics for this imbalanced task), 3 seeds.

Prereq: python benchmarks/preprocess_physionet.py   (builds data/physionet/cache.npz)
Usage:  python benchmarks/physionet_mortality.py --seeds 3
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "physionet")


# ---------------------------------------------------------------------------
# Metrics (self-contained, no sklearn dependency)
# ---------------------------------------------------------------------------

def auroc(y, s):
    """Rank-based AUROC with tie-averaged ranks."""
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0            # 1-based average rank for ties
        ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = (y == 1).sum(); n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(y, s):
    """Average precision (area under precision-recall via step interpolation)."""
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    # AP = sum over thresholds of (recall[i]-recall[i-1]) * precision[i]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


# ---------------------------------------------------------------------------
# Model wrapper: event embedding -> backbone -> masked mean -> classifier
# ---------------------------------------------------------------------------

class MortalityNet(nn.Module):
    def __init__(self, backbone, kind, d_model, V, L, use_dt_decay):
        super().__init__()
        self.backbone, self.kind = backbone, kind
        self.use_dt_decay = use_dt_decay
        self.var_emb = nn.Embedding(V + 1, d_model, padding_idx=V)
        self.val_proj = nn.Linear(1, d_model)
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

    def forward(self, var_ids, values, dt, length_mask):
        B, L = var_ids.shape
        pos = torch.arange(L, device=var_ids.device).unsqueeze(0).expand(B, L)
        x = self.var_emb(var_ids) + self.val_proj(values.unsqueeze(-1)) + self.pos_emb(pos)
        h = self.encode(x, dt)                                   # (B,L,d_model)
        m = length_mask.unsqueeze(-1).float()                   # (B,L,1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)         # masked mean
        return self.head(pooled).squeeze(-1)                    # (B,)


def build_backbone(name, d_model, n_layers, n_heads, L, device):
    if name.startswith("Transformer"):
        cfg = BaselineConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=2 * d_model)
        return SimpleCausalTransformer(cfg).to(device), "baseline"
    if name.startswith("LNN"):
        cfg = BaselineConfig(vocab_size=8, max_seq_len=L, d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=2 * d_model)
        return SimpleCausalLNN(cfg).to(device), "baseline"
    cfg = MTLNNConfig(vocab_size=8, max_seq_len=L, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, n_kv_heads=max(1, n_heads // 2),
                      d_head=d_model // n_heads, dropout=0.1, attention_dropout=0.1,
                      gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2)
    return MTLNNModel(cfg).to(device), "mt"


VARIANTS = [
    ("Transformer",             False),
    ("LNN",                     False),
    ("MT-LNN (Δt feature only)", False),
    ("MT-LNN + Δt-in-decay",     True),
]


def load_splits(seed):
    z = np.load(os.path.join(DATA, "cache.npz"))
    N = len(z["labels"])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_tr, n_va = int(0.7 * N), int(0.15 * N)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    # Per-variable standardisation from TRAIN events only.
    vals, vids = z["values"], z["var_ids"]
    V = int(json.load(open(os.path.join(DATA, "vars.json")))["V"])
    means = np.zeros(V + 1, np.float32); stds = np.ones(V + 1, np.float32)
    tr_v, tr_id = vals[tr], vids[tr]
    for v in range(V):
        m = tr_id == v
        if m.sum() > 1:
            means[v] = tr_v[m].mean(); stds[v] = tr_v[m].std() + 1e-6
    vals_std = (vals - means[vids]) / stds[vids]
    vals_std = vals_std * (vids != V)          # zero out pad positions
    return z, vals_std.astype(np.float32), tr, va, te, V


def make_loader(z, vals_std, idx, L, batch, shuffle, device):
    var_ids = torch.tensor(z["var_ids"][idx], dtype=torch.long)
    values = torch.tensor(vals_std[idx], dtype=torch.float32)
    dt = torch.tensor(z["dt"][idx], dtype=torch.float32)
    lengths = torch.tensor(z["lengths"][idx], dtype=torch.long)
    labels = torch.tensor(z["labels"][idx], dtype=torch.float32)
    mask = (torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1))
    ds = TensorDataset(var_ids, values, dt, mask, labels)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ss = [], []
    for var_ids, values, dt, mask, y in loader:
        s = model(var_ids.to(device), values.to(device), dt.to(device), mask.to(device))
        ys.append(y.numpy()); ss.append(torch.sigmoid(s).cpu().numpy())
    model.train()
    y = np.concatenate(ys); s = np.concatenate(ss)
    return auroc(y, s), auprc(y, s)


def train_eval(name, use_dt, d_model, n_layers, n_heads, L, epochs, lr, seed, device):
    z, vals_std, tr, va, te, V = load_splits(seed)
    train_loader = make_loader(z, vals_std, tr, L, 64, True, device)
    val_loader = make_loader(z, vals_std, va, L, 128, False, device)
    test_loader = make_loader(z, vals_std, te, L, 128, False, device)

    torch.manual_seed(seed)
    backbone, kind = build_backbone(name, d_model, n_layers, n_heads, L, device)
    model = MortalityNet(backbone, kind, d_model, V, L, use_dt).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor((z["labels"][tr] == 0).sum() / max((z["labels"][tr] == 1).sum(), 1),
                         dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_val = -1; best_test = (float("nan"), float("nan"))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for var_ids, values, dt, mask, y in train_loader:
            opt.zero_grad()
            s = model(var_ids.to(device), values.to(device), dt.to(device), mask.to(device))
            loss = lossf(s, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v_auroc, _ = evaluate(model, val_loader, device)
        if v_auroc > best_val:
            best_val = v_auroc
            best_test = evaluate(model, test_loader, device)
    return {"test_auroc": best_test[0], "test_auprc": best_test[1],
            "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out_json", default="benchmarks/physionet_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = int(json.load(open(os.path.join(DATA, "vars.json")))["max_len"])
    print(f"device {device} | PhysioNet-2012 mortality | L={L} | {args.seeds} seeds")

    results = {}
    for name, use_dt in VARIANTS:
        runs = []
        for seed in range(args.seeds):
            r = train_eval(name, use_dt, args.d_model, args.n_layers, args.n_heads,
                           L, args.epochs, args.lr, seed, device)
            runs.append(r)
            print(f"  [{name}] seed {seed}: AUROC {r['test_auroc']:.4f}  "
                  f"AUPRC {r['test_auprc']:.4f}", flush=True)
        results[name] = runs

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    def ms(v):
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    print(f"\n### PhysioNet-2012 mortality, {args.seeds} seeds\n")
    print("| Variant | AUROC ↑ | AUPRC ↑ |")
    print("|---|---|---|")
    for name, _ in VARIANTS:
        rm, rs = ms([r["test_auroc"] for r in results[name]])
        pm, ps = ms([r["test_auprc"] for r in results[name]])
        print(f"| {name} | {rm:.4f} ± {rs:.4f} | {pm:.4f} ± {ps:.4f} |")


if __name__ == "__main__":
    main()
