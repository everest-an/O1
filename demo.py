"""
demo.py — Interactive text generation with MT-LNN.

Features
--------
- Dual-cache decoding (KV cache for attention + recurrent h_prev for the LNN).
- Streaming output: each token is printed the moment it's produced.
- Sampling: temperature + (top-k OR top-p / nucleus) — pick one. Top-p tends
  to behave better with MT-LNN because the GlobalCoherenceLayer's collapse
  gate dynamically narrows the distribution; top-p adapts to that.
- Sanity checks: tokenizer vocab matches model vocab, RoPE table is long
  enough for prompt + max_new_tokens.

Usage
-----
    python demo.py --ckpt checkpoints/final.pt --prompt "The human brain"
    python demo.py --ckpt checkpoints/final.pt --prompt "..." --top_p 0.9 --top_k 0
    python demo.py --ckpt checkpoints/final.pt --interactive
"""

import argparse
import dataclasses
import sys
from typing import Optional

import torch
import torch.nn.functional as F

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import load_checkpoint


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < v[:, [-1]], float("-inf"))


def _filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability ≥ p.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    # Keep tokens *up to* the first index where cumulative ≥ p (inclusive)
    keep = cum <= p
    keep[..., 0] = True                                   # always keep top token
    # Map back to original indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep)
    return logits.masked_fill(~mask, float("-inf"))


def sample_next(
    logits: torch.Tensor,              # (1, V)
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:                      # (1, 1)
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        logits = _filter_top_k(logits, top_k)
    if 0.0 < top_p < 1.0:
        logits = _filter_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_streaming(
    model: MTLNNModel,
    input_ids: torch.Tensor,           # (1, T_prompt)
    tokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
    print_prompt: bool = True,
) -> torch.Tensor:
    """
    Streams generated tokens to stdout the moment each is produced.
    Uses incremental decoding with the dual cache, so cost per new token
    is O(1) plus the cache-grow cost — not O(T²).
    """
    if print_prompt:
        prompt_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        print(prompt_text, end="", flush=True)

    # 1. Prefill
    out = model(input_ids, use_cache=True)
    cache = out["cache"]
    logits = out["logits"][:, -1, :]
    generated = input_ids.clone()

    # 2. Stream
    for _ in range(max_new_tokens):
        next_tok = sample_next(logits, temperature, top_k, top_p)   # (1, 1)
        token_text = tokenizer.decode(next_tok[0].tolist(), skip_special_tokens=True)
        print(token_text, end="", flush=True)
        generated = torch.cat([generated, next_tok], dim=1)

        if eos_token_id is not None and next_tok.item() == eos_token_id:
            break

        out = model(next_tok, cache=cache, use_cache=True)
        cache = out["cache"]
        logits = out["logits"][:, -1, :]

    print()      # final newline
    return generated


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_compat(model: MTLNNModel, config: MTLNNConfig, tokenizer,
                  prompt_len: int, max_new_tokens: int):
    # 1. Tokenizer vocab matches model
    if tokenizer.vocab_size != config.vocab_size:
        print(f"\nERROR: tokenizer vocab_size ({tokenizer.vocab_size}) "
              f"!= model vocab_size ({config.vocab_size}).")
        print("Generation would index out of bounds or produce gibberish.")
        sys.exit(1)

    # 2. RoPE table long enough
    total_len = prompt_len + max_new_tokens
    rope_max = model.embedding.rope.cos_table.shape[0]
    if total_len > rope_max:
        print(f"\nERROR: prompt+max_new_tokens={total_len} exceeds RoPE "
              f"table length ({rope_max}).")
        print(f"Rebuild the model with MTLNNConfig(max_seq_len >= {total_len}) "
              "or shorten the run.")
        sys.exit(1)

    # 3. Attention distance buffers
    attn_max = model.blocks[0].attn._delta.shape[0]
    if total_len > attn_max:
        print(f"\nERROR: prompt+max_new_tokens={total_len} exceeds "
              f"MicrotubuleAttention distance buffer ({attn_max}).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    valid = {f.name for f in dataclasses.fields(MTLNNConfig)} - {"d_proto", "d_proto_total"}
    config = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = MTLNNModel(config).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()
    print(f"Loaded {model.get_num_params()/1e6:.1f}M param MT-LNN "
          f"(vocab={config.vocab_size}, max_seq_len={config.max_seq_len})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id

    def run(prompt: str):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        check_compat(model, config, tok, ids.shape[1], args.max_tokens)
        print()         # blank line before generation
        generate_streaming(
            model, ids, tok,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id,
        )

    if args.interactive:
        print("Interactive mode — Ctrl-C to exit.")
        while True:
            try:
                prompt = input("\n>>> ")
                if prompt.strip():
                    run(prompt)
            except KeyboardInterrupt:
                print("\nBye.")
                break
    else:
        run(args.prompt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--prompt",      default="The human brain")
    p.add_argument("--max_tokens",  type=int,   default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=0,
                                       help="0 = disabled (use top_p only)")
    p.add_argument("--top_p",       type=float, default=0.9,
                                       help="Nucleus sampling. 1.0 = disabled")
    p.add_argument("--tokenizer",   default="gpt2")
    p.add_argument("--interactive", action="store_true")
    main(p.parse_args())
