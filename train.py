"""
train.py — Training script for MT-LNN on WikiText-103.

Usage:
    python train.py                         # default 125M config
    python train.py --d_model 512 --n_layers 6   # small debug run
    python train.py --steps 1000 --batch 4       # quick smoke test
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups, WarmupCosineScheduler, save_checkpoint


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """Flat token array sliced into overlapping fixed-length windows."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start: start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]   # input, labels


def load_wikitext(seq_len: int, tokenizer_name: str = "gpt2"):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading WikiText-103 …")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        text = " ".join(batch["text"])
        return {"ids": tok.encode(text)}

    train_ids = torch.tensor(tokenize(ds["train"])["ids"], dtype=torch.long)
    val_ids   = torch.tensor(tokenize(ds["validation"])["ids"], dtype=torch.long)

    train_ds = TokenDataset(train_ids, seq_len)
    val_ds   = TokenDataset(val_ids, seq_len)
    return train_ds, val_ds, tok.vocab_size


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: MTLNNModel, val_loader: DataLoader, device: str, max_batches: int = 50) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (inp, lbl) in enumerate(val_loader):
        if i >= max_batches:
            break
        inp, lbl = inp.to(device), lbl.to(device)
        out = model(inp, labels=lbl)
        total_loss += out["loss"].item()
        n += 1
    model.train()
    return math.exp(total_loss / max(n, 1))   # perplexity


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config
    config = MTLNNConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )

    # Data
    if args.dummy:
        # Fast dummy dataset for smoke-testing
        vocab = config.vocab_size
        n_tok = args.seq_len * 200
        train_ds = TokenDataset(torch.randint(0, vocab, (n_tok,)), args.seq_len)
        val_ds   = TokenDataset(torch.randint(0, vocab, (n_tok // 10,)), args.seq_len)
    else:
        train_ds, val_ds, _ = load_wikitext(args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = MTLNNModel(config).to(device)
    n_params = model.get_num_params()
    print(f"Parameters: {n_params/1e6:.1f}M")

    # Optimiser with separate LR groups
    param_groups = make_param_groups(model, base_lr=args.lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, args.steps, min_lr=args.lr * 0.1)

    # Mixed precision
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Training
    os.makedirs(args.ckpt_dir, exist_ok=True)
    step = 0
    accum_loss = 0.0
    t0 = time.time()

    model.train()
    while step < args.steps:
        for inp, lbl in train_loader:
            if step >= args.steps:
                break

            inp, lbl = inp.to(device), lbl.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(inp, labels=lbl)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / args.log_every
                ppl = math.exp(min(avg_loss, 20))
                print(f"step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                      f"lr {scheduler.current_lr:.2e} | {elapsed:.0f}s")
                accum_loss = 0.0
                t0 = time.time()

            if step % args.eval_every == 0:
                val_ppl = evaluate(model, val_loader, device)
                print(f"  → val PPL: {val_ppl:.2f}")
                diag = model.get_mt_diagnostics()
                print(f"  → τ mean/std: {diag.get('tau_mean', 0):.3f}/{diag.get('tau_std', 0):.3f} "
                      f"| polarity spread: {diag.get('polarity_std', 0):.3f} "
                      f"| coherence_scale: {diag.get('coherence_scale', 0):.3f}")

            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:06d}.pt")
                save_checkpoint(model, optimizer, step, avg_loss if step > 0 else 0, ckpt_path, config)
                print(f"  → saved {ckpt_path}")

    print("Training complete.")
    # Final save
    save_checkpoint(model, optimizer, step, 0.0, os.path.join(args.ckpt_dir, "final.pt"), config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model",       type=int,   default=1024)
    p.add_argument("--n_layers",      type=int,   default=12)
    p.add_argument("--n_heads",       type=int,   default=16)
    p.add_argument("--seq_len",       type=int,   default=1024)
    p.add_argument("--batch",         type=int,   default=8)
    p.add_argument("--grad_accum",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_steps",  type=int,   default=2000)
    p.add_argument("--steps",         type=int,   default=50000)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--log_every",     type=int,   default=100)
    p.add_argument("--eval_every",    type=int,   default=500)
    p.add_argument("--save_every",    type=int,   default=2000)
    p.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    p.add_argument("--dummy",         action="store_true", help="Use random data (no dataset download)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
