"""
prepare_data.py — pre-tokenise a HuggingFace text dataset to a flat .bin file
on disk, so training can use numpy.memmap and avoid loading everything to RAM.

Output:
    data/{split}.bin   uint16 token stream (each token < 65535)
    data/meta.json     {vocab_size, n_train_tokens, n_val_tokens, tokenizer}

Usage:
    python prepare_data.py                            # WikiText-103 default
    python prepare_data.py --dataset wikitext --config wikitext-2-raw-v1
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def main(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading {args.dataset}/{args.config} …")
    ds = load_dataset(args.dataset, args.config)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size < 65535, "Use uint32 if vocab_size > 65535"

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size}

    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        out_path = os.path.join(args.out_dir, f"{split}.bin")
        n_tokens = 0
        with open(out_path, "wb") as f:
            for item in tqdm(ds[split], desc=f"tokenising {split}"):
                text = item["text"]
                if not text:
                    continue
                ids = tok.encode(text)
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                n_tokens += len(arr)
        print(f"  → {out_path}: {n_tokens:,} tokens")
        meta[f"n_{split}_tokens"] = n_tokens

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Meta: {os.path.join(args.out_dir, 'meta.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="wikitext")
    p.add_argument("--config",   default="wikitext-103-raw-v1")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out_dir",  default="data")
    main(p.parse_args())
