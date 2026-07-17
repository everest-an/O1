"""
benchmarks/mqar.py — Multi-Query Associative Recall (Zoology-style).

MQAR is the standard synthetic probe for recall in recurrent/SSM models
(Arora et al. 2023). The input is a sequence of key-value pairs followed by
queries; for each query the model must emit the value bound to that key
earlier in the sequence. Softmax attention solves MQAR near-perfectly; the
interesting question is how much a recurrent model degrades as the number of
distinct key-value pairs (memory pressure) grows.

Sequence layout (single "kv then query" block):
    k1 v1 k2 v2 ... kN vN  q_{i1} a_{i1} q_{i2} a_{i2} ...
where keys/values are drawn from disjoint vocab ranges and the answer token
a_{ij} = the value bound to the queried key. Loss/accuracy is measured only on
the answer positions.

Usage:
    python benchmarks/mqar.py --n_kv 8 --seeds 3
    python benchmarks/mqar.py --n_kv 16 --seq_len 128 --seeds 3
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)


def make_mqar_batch(n_kv, n_query, vocab_kv, B, device):
    """Build an MQAR batch.

    Vocab layout: keys in [0, vocab_kv), values in [vocab_kv, 2*vocab_kv).
    Sequence: [k1 v1 ... kN vN  q1 a1 ... qM aM]. Labels are -100 except at the
    answer positions (a_j), where the target is the value bound to q_j.
    """
    T = 2 * n_kv + 2 * n_query
    inp = torch.zeros(B, T, dtype=torch.long, device=device)
    lab = torch.full((B, T), -100, dtype=torch.long, device=device)
    for b in range(B):
        keys = torch.randperm(vocab_kv, device=device)[:n_kv]
        vals = torch.randint(0, vocab_kv, (n_kv,), device=device) + vocab_kv
        # kv section
        inp[b, 0:2 * n_kv:2] = keys
        inp[b, 1:2 * n_kv:2] = vals
        # query section: sample queries from the stored keys
        qidx = torch.randint(0, n_kv, (n_query,), device=device)
        base = 2 * n_kv
        inp[b, base + 0::2] = keys[qidx]
        inp[b, base + 1::2] = vals[qidx]          # teacher forcing of answers
        lab[b, base + 1::2] = vals[qidx]          # loss only on answers
    return inp, lab


@torch.no_grad()
def evaluate(model, n_kv, n_query, vocab_kv, device, n_batches=16, B=32):
    model.eval()
    correct = tot = 0
    for _ in range(n_batches):
        inp, lab = make_mqar_batch(n_kv, n_query, vocab_kv, B, device)
        out = model(inp)
        logits = out["logits"][:, :-1, :]
        preds = logits.argmax(dim=-1)
        target = lab[:, 1:]
        mask = target != -100
        correct += ((preds == target) & mask).sum().item()
        tot += mask.sum().item()
    model.train()
    return correct / max(tot, 1)


def build(name, vocab, T, device):
    if name == "Transformer":
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=T,
                             d_model=104, n_layers=2, n_heads=4, d_ff=256)
        return SimpleCausalTransformer(cfg).to(device)
    if name == "LNN":
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=T,
                             d_model=104, n_layers=2, n_heads=4, d_ff=256)
        return SimpleCausalLNN(cfg).to(device)
    cfg = MTLNNConfig(vocab_size=vocab, max_seq_len=T, d_model=104,
                      n_layers=2, n_heads=4, n_kv_heads=2, d_head=26,
                      dropout=0.0, attention_dropout=0.0,
                      gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2)
    return MTLNNModel(cfg).to(device)


def train_eval(name, n_kv, n_query, vocab_kv, steps, lr, seed, device):
    vocab = 2 * vocab_kv
    T = 2 * n_kv + 2 * n_query
    torch.manual_seed(seed)
    model = build(name, vocab, T, device)
    if name == "MT-LNN":
        opt = torch.optim.AdamW(make_param_groups(model, lr), betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    for step in range(steps):
        inp, lab = make_mqar_batch(n_kv, n_query, vocab_kv, 32, device)
        opt.zero_grad()
        out = model(inp, labels=lab)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    acc = evaluate(model, n_kv, n_query, vocab_kv, device)
    return {"acc": acc, "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_kv", type=int, default=8, help="# distinct key-value pairs")
    p.add_argument("--n_query", type=int, default=4)
    p.add_argument("--vocab_kv", type=int, default=32, help="key/value vocab range")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out_json", default="benchmarks/mqar_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | MQAR n_kv={args.n_kv} n_query={args.n_query} "
          f"vocab_kv={args.vocab_kv} | {args.seeds} seeds")

    results = {}
    for name in ["Transformer", "LNN", "MT-LNN"]:
        runs = []
        for seed in range(args.seeds):
            r = train_eval(name, args.n_kv, args.n_query, args.vocab_kv,
                           args.steps, args.lr, seed, device)
            runs.append(r)
            print(f"  [{name}] seed {seed}: acc {r['acc']:.3f}", flush=True)
        results[name] = runs

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    def ms(v):
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    print(f"\n### MQAR (n_kv={args.n_kv}, n_query={args.n_query})\n")
    print("| Model | answer-token accuracy |")
    print("|---|---|")
    for name in ["Transformer", "LNN", "MT-LNN"]:
        m, s = ms([r["acc"] for r in results[name]])
        print(f"| {name} | {m:.3f} ± {s:.3f} |")


if __name__ == "__main__":
    main()
