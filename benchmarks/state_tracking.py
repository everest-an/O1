"""
benchmarks/state_tracking.py — Parity & modular-arithmetic state tracking.

These are Chomsky-hierarchy tasks where soft attention is provably limited
(parity is not solvable by fixed-depth transformers without CoT; see Hahn 2020,
Merrill & Sabharwal 2023), while a recurrent state can track the running
accumulator exactly. This is the regime where MT-LNN's recurrent protofilament
state SHOULD have a genuine, defensible advantage over a Transformer — unlike
Selective Copy, which attention already solves.

Tasks:
  parity   : input is a random bit string; the target at each position is the
             running XOR (parity) of all bits seen so far. Requires 1 bit of
             persistent state that flips on every 1.
  mod_k    : input is a stream of digits 0..k-1; target is the running sum mod k.
             Requires log2(k) bits of persistent state.

We train Transformer / LNN / MT-LNN at matched ~200K params and report
next-token accuracy on the running-state prediction, across multiple seeds and
sequence lengths (to test length generalisation).

Usage:
    python benchmarks/state_tracking.py --task parity --seeds 3
    python benchmarks/state_tracking.py --task mod --k 5 --seeds 3
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)


def make_batch(task, k, T, B, device):
    """Return (input_ids, labels) for the running-state task.

    input_ids[:, t] = digit at t   (0..k-1)
    labels[:, t]    = running (sum mod k) up to and including t
                      (parity is the k=2 case)
    Both are length T. The model is trained as causal LM: predict labels[t]
    from inputs[0..t]. We pass labels aligned so each forward's internal
    shift (logits[:, :-1] vs labels[:, 1:]) targets labels[t] from position
    t-1's hidden state — i.e. we prepend a BOS so position t sees inputs[0..t].
    """
    digits = torch.randint(0, k, (B, T), device=device)
    running = torch.cumsum(digits, dim=1) % k                 # (B, T)
    # Build a length-(T+1) sequence: [BOS, d_0, d_1, ..., d_{T-1}]
    # target at output position t (0-indexed over T+1) should be running[t-1]
    # for t>=1. With internal shift, logits[:, :-1] predicts labels[:, 1:].
    bos = torch.full((B, 1), k, device=device, dtype=torch.long)   # BOS = id k
    inp = torch.cat([bos, digits], dim=1)                     # (B, T+1)
    lab = torch.cat([bos, running], dim=1)                    # (B, T+1)
    return inp, lab


def evaluate(model, task, k, T, device, n_batches=16, B=32):
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inp, lab = make_batch(task, k, T, B, device)
            out = model(inp)
            logits = out["logits"][:, :-1, :]                 # predict lab[:,1:]
            preds = logits.argmax(dim=-1)
            target = lab[:, 1:]
            correct += (preds == target).sum().item()
            tot += target.numel()
    model.train()
    return correct / tot


def build(name, vocab, T, device):
    # vocab = k + 1 (digits 0..k-1 plus BOS). seq len = T+1.
    if name == "Transformer":
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=T + 1,
                             d_model=104, n_layers=2, n_heads=4, d_ff=256)
        return SimpleCausalTransformer(cfg).to(device)
    if name == "LNN":
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=T + 1,
                             d_model=104, n_layers=2, n_heads=4, d_ff=256)
        return SimpleCausalLNN(cfg).to(device)
    cfg = MTLNNConfig(vocab_size=vocab, max_seq_len=T + 1, d_model=104,
                      n_layers=2, n_heads=4, n_kv_heads=2, d_head=26,
                      dropout=0.0, attention_dropout=0.0,
                      gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2)
    return MTLNNModel(cfg).to(device)


def train_eval(name, task, k, T_train, T_test, steps, lr, seed, device):
    vocab = k + 1
    torch.manual_seed(seed)
    model = build(name, vocab, max(T_train, T_test), device)
    if name == "MT-LNN":
        opt = torch.optim.AdamW(make_param_groups(model, lr), betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    for step in range(steps):
        inp, lab = make_batch(task, k, T_train, 32, device)
        opt.zero_grad()
        out = model(inp, labels=lab)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    acc_in = evaluate(model, task, k, T_train, device)
    acc_out = evaluate(model, task, k, T_test, device)   # length generalisation
    return {"acc_in_dist": acc_in, "acc_len_gen": acc_out,
            "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["parity", "mod"], default="parity")
    p.add_argument("--k", type=int, default=2, help="modulus (parity => k=2)")
    p.add_argument("--T_train", type=int, default=64)
    p.add_argument("--T_test", type=int, default=256, help="length generalisation")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out_json", default="benchmarks/state_tracking_results.json")
    args = p.parse_args()
    if args.task == "parity":
        args.k = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | task {args.task} | k={args.k} | "
          f"T_train {args.T_train} -> T_test {args.T_test} | {args.seeds} seeds")

    results = {}
    for name in ["Transformer", "LNN", "MT-LNN"]:
        runs = []
        for seed in range(args.seeds):
            r = train_eval(name, args.task, args.k, args.T_train, args.T_test,
                           args.steps, args.lr, seed, device)
            runs.append(r)
            print(f"  [{name}] seed {seed}: in-dist {r['acc_in_dist']:.3f}  "
                  f"len-gen {r['acc_len_gen']:.3f}", flush=True)
        results[name] = runs

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    def ms(vals):
        return (statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n### {args.task} (k={args.k}), T_train={args.T_train}, "
          f"T_test={args.T_test}\n")
    print("| Model | in-dist acc | length-gen acc |")
    print("|---|---|---|")
    for name in ["Transformer", "LNN", "MT-LNN"]:
        im, isd = ms([r["acc_in_dist"] for r in results[name]])
        gm, gsd = ms([r["acc_len_gen"] for r in results[name]])
        print(f"| {name} | {im:.3f} ± {isd:.3f} | {gm:.3f} ± {gsd:.3f} |")


if __name__ == "__main__":
    main()
