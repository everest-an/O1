# MT-LNN

**Microtubule-Enhanced Liquid Neural Network** — an open-source small language model that combines:

- **Liquid Neural Networks** (closed-form LTC) replacing the Transformer FFN
- **Microtubule architecture** — 13 protofilaments, dynamic-instability ODE, lateral coupling, GTP hydrolysis gating, MAP-protein gating, multi-scale resonance
- **Microtubule attention** — polarity-biased causal attention with GTP-cap distance gating
- **Global coherence layer** — sparse top-k attention with an Orch-OR-inspired collapse gate
- **GQA + KV cache** — efficient streaming inference with dual-state management (attention KV + LNN recurrent state)

The goal is a biologically-inspired architecture for long-text and dynamic tasks, drawing on Penrose-Hameroff Orch-OR and Liquid AI's LFM line.

## Status

Research-grade code. The full test suite passes:

```
[ok] test_shapes_and_loss
[ok] test_gradient_flow                  407 params, all finite
[ok] test_kv_cache_parity                cached vs full diff = 3.4e-7
[ok] test_lnn_recurrence_active          h_prev verifiably flows
[ok] test_prefill_then_decode            mixed-mode diff = 3.9e-7
[ok] test_gqa_kv_cache_size              50% KV cache memory saved
[ok] test_mt_diagnostics                 τ, γ, polarity, lateral all healthy
[ok] test_overfit_single_batch           loss 5.29 → 0.00 in 200 steps
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

### Run the test suite

```bash
python tests/test_model.py
```

### Smoke train on dummy data (no dataset download)

```bash
python train.py --d_model 128 --n_layers 2 --n_heads 4 --batch 2 \
                --seq_len 32 --steps 200 --dummy
```

### Train ~125M model on WikiText-103

```bash
python train.py        # defaults: d_model=1024, n_layers=12, n_heads=16
```

### Generate with KV cache

```bash
python demo.py --ckpt checkpoints/final.pt --prompt "The human brain"
```

## Architecture

```
input_ids
   ↓
Token Embedding + RoPE
   ↓
─── × n_layers ──────────────────────────────────────
MTLNNBlock  (pre-norm)
  • MicrotubuleAttention          [GQA, KV cache]
      polarity bias  +  GTP-cap gate (absolute positions)
  + residual
  • MTLNNLayer                    [recurrent h_prev cache]
      d_model → 13 protofilaments
      13× MultiScaleResonance     [τ_fast / τ_mid / τ_slow]
      LateralCoupling (13×13) + GTP hydrolysis temporal gate
      MAPGate (per protofilament)
      → d_model
  + residual
─────────────────────────────────────────────────────
   ↓
GlobalCoherenceLayer              [KV cache, sparse top-k, collapse gate]
   ↓
LayerNorm → lm_head (weight-tied)
   ↓
logits
```

### Two inference modes

| Mode | LNN behavior | Use case |
|---|---|---|
| `use_lnn_recurrence=False` | h_prev = 0 each step (parallel) | Bit-exact match with full forward; matches training-time semantics |
| `use_lnn_recurrence=True` *(default at inference)* | h_prev threaded across steps | True RNN-style microtubule state accumulation |

The dual-cache `ModelCacheStruct` carries:
- per layer: `(KV cache, h_prev)`
- coherence layer: `KV cache`

## File map

```
mt_lnn/
  config.py            MTLNNConfig (single source of truth)
  embedding.py         TokenEmbedding + RoPE (with position offset)
  mt_attention.py      MicrotubuleAttention — GQA, KV cache, polarity, GTP
  mt_lnn_layer.py      ProtofilamentLTC, MultiScaleResonance,
                       LateralCoupling, MAPGate, MTLNNLayer
  global_coherence.py  GlobalCoherenceLayer (sparse top-k + collapse gate)
  model.py             MTLNNBlock, MTLNNModel, ModelCacheStruct
  utils.py             init_weights, count_parameters, scheduler,
                       checkpointing, separate-LR param groups
train.py               WikiText-103 + dummy data training
eval.py                Perplexity + MT diagnostics
demo.py                KV-cached autoregressive generation
tests/test_model.py    Full test suite
```

## Design references

- **Closed-form LTC** — Hasani et al., *Closed-form continuous-time neural networks*, Nature MI 2022
- **Liquid Foundation Models** — Liquid AI LFM2 / LFM2.5 (2025–2026)
- **Orch-OR** — Penrose & Hameroff; recent experimental support: Wiest, *Neuroscience of Consciousness*, Oxford Academic, 2025
- **GQA** — Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, EMNLP 2023
- **RoPE** — Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*

## License

MIT — see [LICENSE](LICENSE).
