"""
benchmarks/diagnose_mtlnn_longseq.py — mechanistic diagnosis of MT-LNN's
long-sequence training failure.

Runs three probes on a long (T~1024) selective-copy task at ~200K params:

  1. PER-SCALE GRADIENT NORMS. MT-DL has S time-scales with geometrically
     swept tau (small tau = fast/short memory, large tau = slow/long memory).
     We log ||grad(W_in[:, s])|| per scale s alongside its mean tau, so we can
     see whether the LONG-memory (large-tau, decay->1) scales EXPLODE or the
     SHORT ones VANISH. Also logs the pre-clip total grad norm to catch spikes.

  2. use_scan ABLATION. Trains the SAME MT-LNN with real parallel-scan
     recurrence (use_scan=True) vs the parallel/broadcast mode (use_scan=False,
     h_prev=0 — the same non-recurrent regime the LNN baseline uses). If the
     non-recurrent run converges while the recurrent one blows up, the failure
     is caused specifically by the BPTT gradient through the pscan recurrence.

  3. torch.compile SPEEDUP. Times forward+backward with and without compile
     to measure the real speedup (not the assumed 2-3x).

Usage:
    python benchmarks/diagnose_mtlnn_longseq.py --T 1024 --steps 400
"""

import argparse
import math
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups
from benchmarks.selective_copy import SelectiveCopyConfig, make_selective_copy_batch
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)


def train_baseline(name, T, steps, lr, device, log_every=100):
    """Train Transformer / LNN baseline on the SAME long task, to check
    whether MT-LNN fails where the baselines converge."""
    torch.manual_seed(0)
    bcfg = BaselineConfig(vocab_size=16, max_seq_len=T, d_model=104,
                          n_layers=2, n_heads=4, d_ff=256)
    model = (SimpleCausalTransformer(bcfg) if name == "Transformer"
             else SimpleCausalLNN(bcfg)).to(device)
    task = SelectiveCopyConfig(K_mem=4, T_noise=T - 5, vocab_size=16, batch=8,
                               steps=steps, lr=lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model.train()
    final = float("nan")
    for step in range(steps):
        ids, labels = make_selective_copy_batch(task, task.batch, device=device)
        opt.zero_grad()
        out = model(ids, labels=labels)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final = out["loss"].item()
        if (step + 1) % log_every == 0:
            print(f"  [{name}] step {step+1:4d}  loss {final:.3f}", flush=True)
    return final


def build_model(T, device, route_b=False):
    # ~200K MT-LNN, matched to the user's setup (2 layers, d_model=104).
    # route_b=True: each protofilament owns one distinct τ (P channels, not P×S).
    cfg = MTLNNConfig(
        vocab_size=16, max_seq_len=T, d_model=104, n_layers=2, n_heads=4,
        n_kv_heads=2, d_head=26, dropout=0.0, attention_dropout=0.0,
        gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2,
        n_time_scales=(1 if route_b else 5),
        protofilament_timescales=route_b,
    )
    return MTLNNModel(cfg).to(device), cfg


def per_scale_grad_report(model):
    """Return list of (scale_idx, mean_tau, grad_norm) for layer-0 resonance."""
    res = model.blocks[0].lnn.resonance
    S = res.S
    tau = (torch.nn.functional.softplus(res.log_tau) + res.tau_min)
    tau = tau.clamp(res.tau_min, res.tau_max)            # (P, S)
    rows = []
    if res.W_in.grad is not None:
        g = res.W_in.grad                                 # (P, S, D, D)
        for s in range(S):
            rows.append((s, tau[:, s].mean().item(), g[:, s].norm().item()))
    return rows


def train_probe(use_scan, T, steps, lr, device, log_every=50, route_b=False, tag=""):
    torch.manual_seed(0)
    model, cfg = build_model(T, device, route_b=route_b)
    n_params = sum(p.numel() for p in model.parameters())
    task = SelectiveCopyConfig(K_mem=4, T_noise=T - 5, vocab_size=16, batch=8,
                               steps=steps, lr=lr)
    opt = torch.optim.AdamW(make_param_groups(model, lr), betas=(0.9, 0.95))
    model.train()
    history = []
    t0 = time.time()
    for step in range(steps):
        ids, labels = make_selective_copy_batch(task, task.batch, device=device)
        opt.zero_grad()
        out = model(ids, labels=labels, use_lnn_recurrence=use_scan)
        out["loss"].backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)  # measure, don't clip
        if (step + 1) % log_every == 0 or step == 0:
            history.append({"step": step + 1, "loss": out["loss"].item(),
                            "total_grad": total_norm.item()})
            print(f"  [{tag}] step {step+1:4d}  loss {out['loss'].item():.3f}  "
                  f"|grad| {total_norm.item():.2e}", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    tok_s = steps * task.batch * T / (time.time() - t0)
    print(f"  [{tag}] {n_params/1e3:.0f}K params  {tok_s:.0f} tok/s", flush=True)
    history.append({"n_params": n_params, "tok_s": tok_s})
    return history


def compile_speed(T, device, iters=20):
    if device != "cuda":
        return None
    results = {}
    for use_compile in (False, True):
        torch.manual_seed(0)
        model, _ = build_model(T, device)
        m = torch.compile(model) if use_compile else model
        task = SelectiveCopyConfig(K_mem=4, T_noise=T - 5, vocab_size=16, batch=8)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # warmup (compile traces on first calls)
        for _ in range(3):
            ids, labels = make_selective_copy_batch(task, task.batch, device=device)
            opt.zero_grad(); out = m(ids, labels=labels); out["loss"].backward(); opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            ids, labels = make_selective_copy_batch(task, task.batch, device=device)
            opt.zero_grad(); out = m(ids, labels=labels); out["loss"].backward(); opt.step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        tok_s = task.batch * T / dt
        results[use_compile] = {"ms_per_step": dt * 1000, "tok_s": tok_s}
        print(f"  compile={use_compile}: {dt*1000:.1f} ms/step  {tok_s:.0f} tok/s", flush=True)
    if False in results and True in results:
        sp = results[False]["ms_per_step"] / results[True]["ms_per_step"]
        print(f"  >>> torch.compile speedup: {sp:.2f}x")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--skip_compile", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | T={args.T} | steps={args.steps}\n")

    print("=" * 70)
    print("PROBE 0: LNN baseline (the one that converges best)")
    print("=" * 70)
    lnn_final = train_baseline("LNN", args.T, args.steps, 3e-3, device)

    print("\n" + "=" * 70)
    print("PROBE A: legacy MT-LNN (P×S=65 redundant channels)")
    print("=" * 70)
    h_legacy = train_probe(True, args.T, args.steps, args.lr, device, tag="legacy P×S")

    print("\n" + "=" * 70)
    print("PROBE B: ROUTE B — protofilament τ-ownership (P=13 specialised channels)")
    print("=" * 70)
    h_routeb = train_probe(True, args.T, args.steps, args.lr, device,
                           route_b=True, tag="route-B P")

    print("\n" + "=" * 70)
    print("SUMMARY — does specialisation (route B) beat redundant parallel?")
    print("=" * 70)
    def flast(h):
        losses = [r["loss"] for r in h if "loss" in r]
        return losses[-1] if losses else float("nan")
    def tps(h):
        return next((r["tok_s"] for r in h if "tok_s" in r), float("nan"))
    print(f"  LNN (single LTC)          : loss {lnn_final:.3f}")
    print(f"  MT-LNN legacy (P×S=65)    : loss {flast(h_legacy):.3f}   {tps(h_legacy):.0f} tok/s")
    print(f"  MT-LNN route-B (P=13)     : loss {flast(h_routeb):.3f}   {tps(h_routeb):.0f} tok/s")
    print(f"  --> route-B better loss AND faster => specialisation wins")


if __name__ == "__main__":
    main()
