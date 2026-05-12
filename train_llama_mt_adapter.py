"""
Train MT-LNN residual adapters on top of a frozen HuggingFace causal LM.

Example:
    python train_llama_mt_adapter.py \
        --model meta-llama/Llama-3.2-1B \
        --dataset wikitext --dataset_config wikitext-2-raw-v1 \
        --steps 200 --batch 1 --seq_len 512
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from mt_lnn.llama_adapter import (
    attach_mt_adapters,
    count_trainable_parameters,
)


def maybe_apply_lora(model, args):
    if not args.lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requested but `peft` is not installed. Install with "
            "`pip install peft` or run without --lora."
        ) from exc

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_targets.split(","),
    )
    return get_peft_model(model, config)


def build_dataloader(tokenizer, args):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    def tokenize(batch):
        text = [t for t in batch[args.text_column] if t]
        if not text:
            return {"input_ids": []}
        return tokenizer(text, add_special_tokens=False)

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenizing",
    )

    def group_texts(examples):
        ids = []
        for row in examples["input_ids"]:
            ids.extend(row + [tokenizer.eos_token_id])
        total = (len(ids) // args.seq_len) * args.seq_len
        ids = ids[:total]
        chunks = [ids[i : i + args.seq_len] for i in range(0, total, args.seq_len)]
        return {"input_ids": chunks, "labels": [c.copy() for c in chunks]}

    lm_ds = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="chunking",
    )
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm_ds, batch_size=args.batch, shuffle=True, drop_last=True)


def save_adapter_checkpoint(model, args, step):
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "step": step,
        "model": args.model,
        "state_dict": {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if "mt_adapter" in k or "lora_" in k
        },
        "args": vars(args),
    }
    path = os.path.join(args.out_dir, f"llama_mt_adapter_{step:06d}.pt")
    torch.save(payload, path)
    print(f"saved {path}")


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    wrapped = attach_mt_adapters(
        model,
        every=args.mt_every,
        n_protofilaments=args.mt_proto,
        n_time_scales=args.mt_scales,
        map_hidden_dim=args.mt_map_hidden,
        dropout=args.mt_dropout,
        init_scale=args.mt_init_scale,
        use_scan=not args.mt_no_scan,
    )
    model = maybe_apply_lora(model, args)
    model.to(device)
    model.train()

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"Wrapped decoder layers: {wrapped}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    loader = build_dataloader(tokenizer, args)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0:
                elapsed = max(time.time() - t0, 1e-3)
                toks = args.log_every * args.batch * args.seq_len
                print(
                    f"step {step:6d} | loss {loss.item() * args.grad_accum:.4f} | "
                    f"{toks / elapsed:.0f} tok/s"
                )
                t0 = time.time()
            if step % args.save_every == 0:
                save_adapter_checkpoint(model, args, step)

    save_adapter_checkpoint(model, args, step)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", default="text")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--out_dir", default="checkpoints/llama_mt_adapter")

    p.add_argument("--mt_every", type=int, default=4)
    p.add_argument("--mt_proto", type=int, default=13)
    p.add_argument("--mt_scales", type=int, default=5)
    p.add_argument("--mt_map_hidden", type=int, default=64)
    p.add_argument("--mt_dropout", type=float, default=0.0)
    p.add_argument("--mt_init_scale", type=float, default=1e-3)
    p.add_argument("--mt_no_scan", action="store_true")

    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
