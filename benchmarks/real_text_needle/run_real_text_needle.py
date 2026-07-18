"""Real-text Needle-in-a-Haystack retrieval supplement.

The background tokens come from real text when HuggingFace datasets are
available. A controlled key-value needle is inserted into the text, and the
model must retrieve the value at the final query. This bridges synthetic
Selective Copy and template-only natural-language recall without claiming
open-domain QA.

Provenance: this is the original experiment script (contributed 2026-07) that
produced the "Real-text needle retrieval" figure where Transformer/LNN/Mamba
converge to ~1.0 but MT-LNN collapses at L>=1024. It is the CLEAN testbed for
the MT-LNN long-sequence-convergence investigation (unlike synthetic Selective
Copy, which is too hard — even Transformer stalls there). Path handling below
is made robust so it runs from anywhere inside the O1 repo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR


def _find_repo_root(start: Path) -> Path:
    """Walk up until we find the dir that contains both mt_lnn/ and benchmarks/."""
    for p in [start, *start.parents]:
        if (p / "mt_lnn").is_dir() and (p / "benchmarks").is_dir():
            return p
    return start


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines import BaselineConfig, SimpleCausalLNN, SimpleCausalTransformer
from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups


RAW_FIELDS = [
    "experiment", "model", "context_length", "needle_depth", "seed", "steps",
    "batch_size", "eval_batches", "dataset_name", "dataset_config", "dataset_split",
    "dataset_revision", "manifest_sha256", "vocab_size", "num_params",
    "final_train_loss", "recall_accuracy", "train_time_s", "device", "status", "notes",
]

BASE_TOKENS = [
    "<pad>", "<unk>", "record", "key", "has", "value", "question", "what",
    "for", "answer", ".", "?", "the", "a", "of", "and", "to", "in",
]
KEY_TOKENS = [f"k{i:02d}" for i in range(64)]
VALUE_TOKENS = [f"v{i:02d}" for i in range(64)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text.lower())


def load_real_texts(args) -> Tuple[List[str], Dict[str, object]]:
    notes = ""
    texts: List[str] = []
    try:
        from datasets import load_dataset
        dataset_candidates = [args.dataset_name]
        if args.dataset_name == "wikitext":
            dataset_candidates.insert(0, "Salesforce/wikitext")
        last_exc = None
        ds = None
        for dataset_name in dataset_candidates:
            try:
                ds = load_dataset(
                    dataset_name,
                    args.dataset_config or None,
                    split=args.dataset_split,
                    revision=args.dataset_revision or None,
                )
                break
            except Exception as exc:
                last_exc = exc
        if ds is None:
            raise last_exc if last_exc is not None else RuntimeError("dataset load failed")
        field = args.text_field
        for item in ds.select(range(min(args.max_docs, len(ds)))):
            txt = str(item.get(field, "")).strip()
            if len(txt.split()) > 20:
                texts.append(txt)
    except Exception as exc:
        notes = f"dataset fallback used: {exc!r}"
        texts = [
            "The history of mathematics contains long chains of argument, definitions, examples, and digressions that often separate a statement from the later question that uses it.",
            "In a long nineteenth century novel, a small fact about a letter, a room, or a family relation can be introduced early and become important many pages later.",
            "Scientific articles frequently introduce abbreviations, datasets, and experimental conditions before returning to them in the results and discussion sections.",
            "A travel diary may mention the name of an inn, the color of a bridge, and the time of a train long before the narrator asks the reader to connect those details.",
        ]
    joined = "\n".join(texts)
    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_revision": args.dataset_revision,
        "text_field": args.text_field,
        "max_docs": args.max_docs,
        "num_docs": len(texts),
        "sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        "notes": notes,
    }
    return texts, manifest


def build_vocab(texts: Sequence[str], max_text_vocab: int) -> Tuple[List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    for text in texts:
        for tok in simple_tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
    text_vocab = [tok for tok, _count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_text_vocab]]
    vocab = []
    for tok in BASE_TOKENS + KEY_TOKENS + VALUE_TOKENS + text_vocab:
        if tok not in vocab:
            vocab.append(tok)
    return vocab, {tok: i for i, tok in enumerate(vocab)}


class MambaCausalLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 104, n_layers: int = 2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError("mamba-ssm is unavailable; Mamba baseline skipped.") from exc
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None, **_):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        logits = self.lm_head(self.norm(x))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return out


def build_model(model_name: str, context_length: int, vocab_size: int, device: str,
                route_b: bool = False) -> nn.Module:
    key = model_name.lower()
    if key == "transformer":
        cfg = BaselineConfig(vocab_size=vocab_size, max_seq_len=context_length, d_model=104, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
        return SimpleCausalTransformer(cfg).to(device)
    if key == "lnn":
        cfg = BaselineConfig(vocab_size=vocab_size, max_seq_len=context_length, d_model=104, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
        return SimpleCausalLNN(cfg).to(device)
    if key in ("mt-lnn", "mt-lnn-routeb"):
        rb = route_b or key == "mt-lnn-routeb"
        cfg = MTLNNConfig(vocab_size=vocab_size, max_seq_len=context_length, d_model=104, n_layers=2, n_heads=4, n_kv_heads=2, d_head=26, n_protofilaments=13, dropout=0.0, attention_dropout=0.0, gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2,
                          n_time_scales=(1 if rb else 5), protofilament_timescales=rb)
        return MTLNNModel(cfg).to(device)
    if key == "mamba":
        return MambaCausalLM(vocab_size, context_length).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def count_params(model: nn.Module) -> int:
    if hasattr(model, "get_num_params"):
        return int(model.get_num_params())
    return int(sum(p.numel() for p in model.parameters()))


def optimizer_for(model: nn.Module, lr: float):
    if isinstance(model, MTLNNModel):
        return torch.optim.AdamW(make_param_groups(model, lr), betas=(0.9, 0.95))
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))


def make_one(text_pool: List[int], tok: Dict[str, int], context_length: int, depth: float, device: str):
    key = tok[random.choice(KEY_TOKENS)]
    value_tok = random.choice(VALUE_TOKENS)
    value = tok[value_tok]
    record = [tok["record"], key, tok["has"], tok["value"], value, tok["."]]
    prompt = [tok["question"], tok["what"], tok["value"], tok["for"], key, tok["?"], tok["answer"]]
    needed = context_length - len(prompt) - 1
    start = random.randrange(max(1, len(text_pool) - needed - 1))
    ids = list(text_pool[start:start + needed])
    if len(ids) < needed:
        ids = (ids * ((needed // max(1, len(ids))) + 1))[:needed]
    insert_max = max(1, len(ids) - len(record))
    pos = min(insert_max, max(0, int(depth * insert_max)))
    ids[pos:pos + len(record)] = record
    ids = ids[:needed] + prompt + [value]
    labels = [-100] * len(ids)
    labels[-1] = value
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(labels, dtype=torch.long, device=device), value


def make_batch(text_pool, tok, context_length, batch, depths, device):
    xs, ys = [], []
    for _ in range(batch):
        x, y, _ = make_one(text_pool, tok, context_length, random.choice(depths), device)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def train_model(model, text_pool, tok, context_length, batch, steps, lr, depths, device):
    model.train()
    opt = optimizer_for(model, lr)
    final_loss = float("nan")
    t0 = time.time()
    for _ in range(steps):
        x, y = make_batch(text_pool, tok, context_length, batch, depths, device)
        opt.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(out["loss"].detach().cpu().item())
    return {"final_train_loss": final_loss, "train_time_s": time.time() - t0}


@torch.no_grad()
def eval_recall(model, text_pool, tok, context_length, depth, batch, eval_batches, value_ids, device):
    model.eval()
    correct = total = 0
    for _ in range(eval_batches):
        prefixes, targets = [], []
        for _b in range(batch):
            x, _y, value = make_one(text_pool, tok, context_length, depth, device)
            prefixes.append(x[:-1])
            targets.append(value)
        prefix = torch.stack(prefixes)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        out = model(prefix)
        pred = out["logits"][:, -1, value_ids].argmax(dim=-1)
        pred_values = torch.tensor(value_ids, device=device)[pred]
        correct += int((pred_values == target).sum().item())
        total += int(target.numel())
    return correct / max(total, 1)


def write_rows(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--models", nargs="+", default=["Transformer", "LNN", "MT-LNN", "Mamba"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--max_docs", type=int, default=200)
    parser.add_argument("--max_text_vocab", type=int, default=384)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--suite_name", choices=["smoke", "core_full", "extended_full", "full"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out_root", default=None, help="override results root (default: this script's dir/results)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.context_lengths = [256]
        args.depths = [0.5]
        args.steps = min(args.steps, 2)
        args.seeds = [0]
        args.batch = 1
        args.eval_batches = 1
        args.max_docs = 20
    if args.suite_name is None:
        args.suite_name = "smoke" if args.smoke else "full"
    device = device_from_arg(args.device)
    texts, manifest = load_real_texts(args)
    vocab, tok = build_vocab(texts, args.max_text_vocab)
    text_pool = [tok.get(t, tok["<unk>"]) for text in texts for t in simple_tokens(text)]
    if len(text_pool) < max(args.context_lengths):
        text_pool = (text_pool * ((max(args.context_lengths) // max(1, len(text_pool))) + 2))
    value_ids = [tok[t] for t in VALUE_TOKENS]
    root = Path(args.out_root) if args.out_root else (EXP_DIR / "results")
    out_dir = root / args.suite_name
    raw = out_dir / "raw" / "real_text_needle_raw.csv"
    config = out_dir / "configs" / "real_text_needle_config.json"
    manifest_path = out_dir / "manifests" / "real_text_needle_manifest.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(vars(args) | {"vocab_size": len(vocab), "device_resolved": device}, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    for length in args.context_lengths:
        for seed in args.seeds:
            for model_name in args.models:
                base = {
                    "experiment": "real_text_needle_retrieval",
                    "model": model_name,
                    "context_length": length,
                    "seed": seed,
                    "steps": args.steps,
                    "batch_size": args.batch,
                    "eval_batches": args.eval_batches,
                    "dataset_name": manifest["dataset_name"],
                    "dataset_config": manifest["dataset_config"],
                    "dataset_split": manifest["dataset_split"],
                    "dataset_revision": manifest["dataset_revision"],
                    "manifest_sha256": manifest["sha256"],
                    "vocab_size": len(vocab),
                    "device": device,
                    "status": "ok",
                    "notes": manifest["notes"],
                }
                try:
                    set_seed(seed)
                    print(f"[real-text-needle] L={length} seed={seed} model={model_name}", flush=True)
                    model = build_model(model_name, length, len(vocab), device)
                    params = count_params(model)
                    train_info = train_model(model, text_pool, tok, length, args.batch, args.steps, args.lr, args.depths, device)
                    for depth in args.depths:
                        rows.append(dict(base, needle_depth=depth, num_params=params, **train_info, recall_accuracy=eval_recall(model, text_pool, tok, length, depth, args.batch, args.eval_batches, value_ids, device)))
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except ImportError as exc:
                    for depth in args.depths:
                        rows.append(dict(base, needle_depth=depth, num_params="", final_train_loss="", recall_accuracy="", train_time_s="", status="skipped", notes=str(exc)))
                except Exception as exc:
                    for depth in args.depths:
                        rows.append(dict(base, needle_depth=depth, num_params="", final_train_loss="", recall_accuracy="", train_time_s="", status="failed", notes=repr(exc)))
                # Incremental checkpoint: rewrite CSV after every model so a
                # session teardown mid-run does not lose completed trainings.
                write_rows(raw, rows, RAW_FIELDS)
    write_rows(raw, rows, RAW_FIELDS)
    print(f"wrote {raw}")
    print(f"wrote {config}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
