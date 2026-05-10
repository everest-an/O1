"""
demo.py — Interactive text generation with MT-LNN, dual-cache inference.

Inference state per layer is a tuple (kv_cache, h_prev):
  - kv_cache: (K, V) attention cache, grows by 1 token per step
  - h_prev:   recurrent microtubule hidden state (B, n_proto, d_proto)

The first forward pass encodes the full prompt and populates both caches.
Each subsequent step processes only the most recent token (T=1) and reuses
the cache, giving O(T) total cost instead of O(T²).

Usage:
    python demo.py --ckpt checkpoints/final.pt --prompt "The human brain"
    python demo.py --ckpt checkpoints/final.pt --interactive
"""

import argparse
import dataclasses

import torch
import torch.nn.functional as F

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import load_checkpoint


@torch.no_grad()
def generate(
    model: MTLNNModel,
    input_ids: torch.Tensor,        # (1, T_prompt)
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_token_id: int = None,
) -> torch.Tensor:
    """
    Autoregressive generation with full dual-cache.
    """
    # 1. Prefill: encode the entire prompt, build cache
    out = model(input_ids, use_cache=True)
    cache = out["cache"]
    logits = out["logits"][:, -1, :]

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Sample next token from logits of last position
        scaled = logits / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)        # (1,1)
        generated = torch.cat([generated, next_tok], dim=1)

        if eos_token_id is not None and next_tok.item() == eos_token_id:
            break

        # 2. Incremental decode: feed only the new token, advance cache
        out = model(next_tok, cache=cache, use_cache=True)
        cache = out["cache"]
        logits = out["logits"][:, -1, :]

    return generated


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    valid_keys = {f.name for f in dataclasses.fields(MTLNNConfig)} - {"d_proto", "d_proto_total"}
    config = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    model = MTLNNModel(config).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()
    print(f"Loaded {model.get_num_params()/1e6:.1f}M param MT-LNN.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tok.eos_token_id

    def run(prompt: str):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out_ids = generate(
            model, ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_id,
        )
        print("\n" + tok.decode(out_ids[0].tolist()))

    if args.interactive:
        print("Enter prompts (Ctrl-C to exit):")
        while True:
            try:
                prompt = input("\n>>> ")
                run(prompt)
            except KeyboardInterrupt:
                break
    else:
        run(args.prompt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--prompt",      default="The human brain")
    p.add_argument("--max_tokens",  type=int,   default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--interactive", action="store_true")
    main(p.parse_args())
