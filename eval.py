"""
eval.py — Evaluation utilities for MT-LNN.

Usage:
    python eval.py --ckpt checkpoints/final.pt
    python eval.py --ckpt checkpoints/final.pt --diagnostics
"""

import argparse
import math
import dataclasses

import torch
from torch.utils.data import DataLoader

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import load_checkpoint


@torch.no_grad()
def evaluate_perplexity(model: MTLNNModel, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total_nll, n_tokens = 0.0, 0
    for inp, lbl in dataloader:
        inp, lbl = inp.to(device), lbl.to(device)
        out = model(inp, labels=lbl)
        # loss is mean NLL; multiply back by number of tokens (T-1)
        total_nll += out["loss"].item() * lbl.numel()
        n_tokens += lbl.numel()
    return math.exp(total_nll / max(n_tokens, 1))


@torch.no_grad()
def evaluate_mt_diagnostics(model: MTLNNModel) -> dict:
    """Extended MT-specific health metrics."""
    return model.get_mt_diagnostics()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("config", {})

    # Rebuild config; fall back to defaults for missing keys
    config = MTLNNConfig(**{k: v for k, v in cfg_dict.items()
                            if k in {f.name for f in dataclasses.fields(MTLNNConfig)}
                            and k not in {"d_proto", "d_proto_total"}})
    model = MTLNNModel(config).to(device)
    load_checkpoint(args.ckpt, model)
    print(f"Loaded checkpoint: {args.ckpt}  ({model.get_num_params()/1e6:.1f}M params)")

    if args.diagnostics:
        diag = evaluate_mt_diagnostics(model)
        print("\n=== MT Diagnostics ===")
        for k, v in diag.items():
            print(f"  {k}: {v:.4f}")

    if args.eval_data:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from train import TokenDataset

        tok = AutoTokenizer.from_pretrained("gpt2")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        text = " ".join(ds["test"]["text"])
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        test_ds = TokenDataset(ids, config.max_seq_len)
        loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)
        ppl = evaluate_perplexity(model, loader, device)
        print(f"\nTest PPL: {ppl:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True, help="Path to checkpoint .pt file")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--eval_data",   action="store_true", help="Run perplexity on WikiText-103 test set")
    p.add_argument("--batch",       type=int, default=8)
    main(p.parse_args())
