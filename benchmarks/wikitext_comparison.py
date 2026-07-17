"""
benchmarks/wikitext_comparison.py — Fair WikiText-103 LM comparison.

Trains Transformer / LNN / MT-LNN at matched scale (~80M params, d_model=832,
12 layers) on WikiText-103 with an IDENTICAL token budget, and reports
validation perplexity + throughput. This replaces the paper's previously
un-backed 125M table with a real, reproducible (if compute-limited) run.

Design notes for a HONEST comparison:
  * All three share d_model / n_layers / n_heads / seq_len and the SAME
    optimizer-step budget and global batch, so each sees the same #tokens.
  * Label convention is unified: we pass labels = input_ids and every model's
    forward shifts internally (logits[:, :-1] vs labels[:, 1:]) — so the loss
    target is the true next token for all three.
  * Runs on an 8 GB laptop GPU with batch=4 + grad-accum (bf16). This is NOT a
    full-convergence run (the paper's 100K-step A100 recipe is out of reach
    locally); it is a fair, same-budget snapshot. The step budget is reported.

Usage:
    python benchmarks/wikitext_comparison.py --steps 3000 --batch 4 --grad_accum 8
    python benchmarks/wikitext_comparison.py --models MT-LNN   # single model
"""

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups, WarmupCosineScheduler
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)
from train import BinDataset


def build_model(name, vocab_size, seq_len, device):
    """Three architectures at ~80M scale (d_model=832, 12 layers, 13 heads)."""
    if name == "Transformer":
        cfg = BaselineConfig(vocab_size=vocab_size, max_seq_len=seq_len,
                             d_model=832, n_layers=12, n_heads=13,
                             d_ff=450, dropout=0.1, tie_embeddings=True)
        return SimpleCausalTransformer(cfg).to(device), cfg
    if name == "LNN":
        cfg = BaselineConfig(vocab_size=vocab_size, max_seq_len=seq_len,
                             d_model=832, n_layers=12, n_heads=13,
                             d_ff=450, dropout=0.1, tie_embeddings=True)
        return SimpleCausalLNN(cfg).to(device), cfg
    if name == "MT-LNN":
        cfg = MTLNNConfig(vocab_size=vocab_size, max_seq_len=seq_len,
                          d_model=832, n_layers=12, n_heads=13, n_kv_heads=1,
                          d_head=64, dropout=0.1)
        return MTLNNModel(cfg).to(device), cfg
    raise ValueError(name)


@torch.no_grad()
def eval_ppl(model, loader, device, max_batches, amp_dtype):
    model.eval()
    tot_loss, tot = 0.0, 0
    for i, (x, _y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(x, labels=x)          # unified label convention
        tot_loss += out["loss"].item()
        tot += 1
    model.train()
    return math.exp(min(tot_loss / max(tot, 1), 20.0))


def train_one(name, args, train_loader, val_loader, vocab_size, device, amp_dtype):
    model, cfg = build_model(name, vocab_size, args.seq_len, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: {n_params/1e6:.1f}M params ===", flush=True)

    if name == "MT-LNN":
        opt = torch.optim.AdamW(make_param_groups(model, args.lr), betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    sched = WarmupCosineScheduler(opt, args.warmup, args.steps, min_lr=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda")

    hist = []
    step = 0
    t0 = time.time()
    tokens_seen = 0
    model.train()
    while step < args.steps:
        for x, _y in train_loader:
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(x, labels=x)
                loss = out["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            tokens_seen += x.numel()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            step += 1
            if step % args.log_every == 0:
                tps = tokens_seen / max(time.time() - t0, 1e-3)
                print(f"  [{name}] step {step:5d}/{args.steps}  "
                      f"loss {out['loss'].item():.3f}  "
                      f"ppl {math.exp(min(out['loss'].item(),20)):.1f}  "
                      f"{tps:.0f} tok/s", flush=True)
            if step % args.eval_every == 0:
                vp = eval_ppl(model, val_loader, device, args.eval_batches, amp_dtype)
                print(f"    [{name}] val PPL @ step {step}: {vp:.2f}", flush=True)
                hist.append({"step": step, "val_ppl": vp})
    final_ppl = eval_ppl(model, val_loader, device, args.eval_batches * 2, amp_dtype)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return {"name": name, "params_M": n_params / 1e6, "final_val_ppl": final_ppl,
            "history": hist, "peak_gb": peak_gb,
            "tok_per_sec": tokens_seen / max(time.time() - t0, 1e-3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--models", nargs="+", default=["Transformer", "LNN", "MT-LNN"])
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--out_json", default="benchmarks/wikitext_comparison_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab_size = meta["vocab_size"]
    print(f"device {device} | vocab {vocab_size} | budget {args.steps} steps "
          f"× global batch {args.batch*args.grad_accum} | seq {args.seq_len}")

    train_ds = BinDataset(os.path.join(args.data_dir, "train.bin"), args.seq_len)
    val_path = os.path.join(args.data_dir, "validation.bin")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.data_dir, "test.bin")
    val_ds = BinDataset(val_path, args.seq_len)
    print(f"train tokens {len(train_ds.data):,} | val tokens {len(val_ds.data):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)

    results = {}
    for name in args.models:
        torch.manual_seed(0)
        results[name] = train_one(name, args, train_loader, val_loader,
                                  vocab_size, device, amp_dtype)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        torch.cuda.empty_cache()

    print("\n" + "=" * 56)
    print(f"{'Model':<14s} {'#Params':>10s} {'val PPL':>10s} {'tok/s':>8s} {'peak GB':>8s}")
    for name in args.models:
        r = results[name]
        print(f"{name:<14s} {r['params_M']:>9.1f}M {r['final_val_ppl']:>10.2f} "
              f"{r['tok_per_sec']:>8.0f} {r['peak_gb']:>7.1f}")
    print(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
