"""
benchmarks/multi_seed_sweep.py — Multi-seed fair comparison for the paper.

Runs the head-to-head Selective Copy comparison (Transformer / LNN / MT-LNN)
across multiple random seeds and both training budgets:

  - headline:      T_noise=32,  1500 steps, batch 16
  - long-context:  T_noise in {32, 96, 224}, 600/600/500 steps

and reports mean ± std for held-out token accuracy and sequence exact match,
plus MT-LNN's AVP Phi_hat response. Uses the corrected greedy-decode
evaluation (models without a cache fall back to full-sequence recompute).

Usage:
    python benchmarks/multi_seed_sweep.py            # 5 seeds, full sweep
    python benchmarks/multi_seed_sweep.py --seeds 3  # quicker

Output: benchmarks/multi_seed_results.json + a Markdown summary on stdout.
"""

import argparse
import json
import os
import statistics
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNModel, phi_hat_anesthesia_sweep
from benchmarks.selective_copy import (
    SelectiveCopyConfig, make_selective_copy_batch, evaluate_selective_copy,
)
from benchmarks.long_context import build_models, train_model


CONFIGS = [
    # (label, T_noise, steps, batch)
    ("headline_T37",     32, 1500, 16),
    ("longctx_T37",      32,  600, 16),
    ("longctx_T101",     96,  600, 16),
    ("longctx_T229",    224,  500,  8),
]


def run_one(seed: int, T_noise: int, steps: int, batch: int, device: str) -> dict:
    task = SelectiveCopyConfig(
        K_mem=4, T_noise=T_noise, vocab_size=16, batch=batch,
        steps=steps, lr=3e-3, eval_batches=8, log_every=10 ** 9,
    )
    torch.manual_seed(seed)
    models = build_models(task, device)
    out = {}
    for name, model in models.items():
        torch.manual_seed(seed)
        train_info = train_model(model, task, name, device=device)
        torch.manual_seed(1000 + seed)      # eval stream independent of train
        eval_info = evaluate_selective_copy(model, task, device=device, n_batches=16)
        entry = {
            "tok_acc":    eval_info["token_accuracy"],
            "seq_exact":  eval_info["sequence_exact"],
            "train_time": train_info["train_time"],
            "final_loss": train_info["final_loss"],
        }
        if isinstance(model, MTLNNModel):
            torch.manual_seed(1000 + seed)
            ids_eval, _ = make_selective_copy_batch(task, B=4, device=device)
            sweep = phi_hat_anesthesia_sweep(model, ids_eval, kappas=[1.0, 10.0])
            entry["phi_clean"] = sweep[1.0]
            entry["phi_full"] = sweep[10.0]
            entry["phi_delta"] = sweep[10.0] - sweep[1.0]
        out[name] = entry
    return out


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    p.add_argument("--out_json", default="benchmarks/multi_seed_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__} | seeds: {args.seeds}")

    results = {label: {} for label, *_ in CONFIGS}
    for label, T_noise, steps, batch in CONFIGS:
        for seed in range(args.seeds):
            print(f"[{label}] seed {seed} ...", flush=True)
            r = run_one(seed, T_noise, steps, batch, device)
            for name, entry in r.items():
                results[label].setdefault(name, []).append(entry)
            for name, entry in r.items():
                print(f"    {name:<12s} tok {entry['tok_acc']:.3f}  "
                      f"seq {entry['seq_exact']:.3f}  "
                      f"loss {entry['final_loss']:.4f}", flush=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nraw results -> {args.out_json}\n")

    # ------------------------------------------------------------------
    # Markdown summary: mean ± std over seeds
    # ------------------------------------------------------------------
    for label, *_ in CONFIGS:
        print(f"### {label}\n")
        print("| Model | tok acc (mean±std) | seq exact (mean±std) | train s |")
        print("|---|---|---|---|")
        for name in ["Transformer", "LNN", "MT-LNN"]:
            runs = results[label][name]
            tok_m, tok_s = mean_std([r["tok_acc"] for r in runs])
            seq_m, seq_s = mean_std([r["seq_exact"] for r in runs])
            t_m, _ = mean_std([r["train_time"] for r in runs])
            print(f"| {name} | {tok_m:.3f} ± {tok_s:.3f} "
                  f"| {seq_m:.3f} ± {seq_s:.3f} | {t_m:.0f} |")
        mt_runs = results[label]["MT-LNN"]
        if "phi_delta" in mt_runs[0]:
            d_m, d_s = mean_std([r["phi_delta"] for r in mt_runs])
            print(f"\nMT-LNN AVP Phi_hat delta (k=10 vs k=1): {d_m:+.3f} ± {d_s:.3f}")
        print()


if __name__ == "__main__":
    main()
