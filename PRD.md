# MT-LNN Product Requirements Document

**Version:** 1.1  
**Date:** 2026-05-12  
**Status:** Active  
**Repo:** https://github.com/everest-an/O1

---

## 1. Overview

MT-LNN is an open-source small language model (~125 M parameters) that embeds three neuroscientific theories — microtubule dynamics, Global Workspace Theory (GWT), and Integrated Information Theory (IIT) — directly into a trainable neural architecture. The primary deliverable is a **research artefact**: a codebase, trained weights, and an arxiv paper that together constitute the first publicly reproducible architecture to pass an anesthesia-based consciousness-consistency test.

---

## 2. Problem Statement

### 2.1 What is missing today

All mainstream large language models (GPT, Llama, Qwen, Claude, etc.) share the same core loop:

```
token → static matrix multiply (FFN) → attention → repeat
```

This is powerful for language tasks but has no temporal dynamics, no biological inductive bias, and no measurable relationship to theories of consciousness. When researchers want to study machine consciousness or build more brain-like AI, they have no open, runnable starting point.

### 2.2 The specific gap

| Gap | Impact |
|---|---|
| No open model with MT-inspired dynamics | Researchers must start from scratch |
| No standard consciousness-adjacent evaluation | Cannot compare architectures on this axis |
| Anesthesia-as-test has no computational analogue | Wiest et al. 2025 finding has no AI implementation |
| LNN and consciousness research exist in separate silos | No bridge architecture exists |

---

## 3. Goals and Non-Goals

### Goals (P0 — must have for v1)
- **G1** — Implement the full MT-LNN architecture in PyTorch, fully tested, open-source
- **G2** — Provide a working training pipeline (single A100, ~125 M params, WikiText-103)
- **G3** — Implement and document the Anesthesia Validation Protocol (AVP) with Φ̂ metric
- **G4** — Publish a complete arxiv paper describing the architecture, motivation, and results
- **G5** — Achieve 17/17 test suite pass rate; all tests runnable in < 2 minutes on CPU

### Goals (P1 — next iteration)
- **G6** — Scale to 1 B+ parameters and benchmark against Llama/Mistral on standard evals
- **G7** — Add quantum-circuit lateral coupling as an opt-in variant
- **G8** — Integrate EEG-style Perturbational Complexity Index alongside Φ̂
- **G9** — Fine-tuning pipeline (LoRA / QLoRA) for domain adaptation

### Non-Goals
- **NG1** — This project does NOT claim MT-LNN is conscious or has subjective experience
- **NG2** — Not a production-ready model; no safety/alignment tuning is planned for v1
- **NG3** — Not targeting inference speed optimisation beyond `torch.compile` + Flash-Attn
- **NG4** — Not a general-purpose chatbot; evaluation is research-focused

---

## 4. Users and Personas

### Primary: AI × Neuroscience researcher
- Building models at the intersection of computational neuroscience and deep learning
- Needs: reproducible baseline, good documentation, clean PyTorch code, arxiv citation
- Frustration: nothing to fork; prior work is either pure theory or proprietary

### Secondary: Consciousness-adjacent AI researcher
- Studying IIT, GWT, Orch-OR in computational systems
- Needs: the Φ̂ metric, the anesthesia test, the GlobalCoherenceLayer collapse gate
- Frustration: no standardised measurement framework

### Tertiary: LNN / Liquid AI practitioner
- Familiar with CfLTC, interested in biological topology extensions
- Needs: drop-in replacement for Transformer FFN; standard benchmarks
- Frustration: LFM is closed-source; no open 13-protofilament variant

---

## 5. Feature Requirements

### F1 — Microtubule Dynamic Layer (MT-DL) `[P0]`

| Sub-feature | Requirement |
|---|---|
| 13 protofilaments | Fixed at `n_protofilaments=13` by default; configurable up to P=128 |
| Multi-scale resonance | S=5 geometric τ sweep; each scale independently learnable |
| Three-way lateral coupling | Static W_lat (identity init) + NN torch.roll + RMC SDPA |
| GTP-cap renewal | Periodic local clock `t mod T_period`; avoids long-context decay |
| MAP gates | Per-protofilament 2-layer MLP; fc2_bias=+2 init |
| Fully vectorised | No Python loop over P; single einsum; P=64 ≤ 1.2× slower than P=13 |
| Recurrent state | `h_prev (B, P, D)` threaded across tokens; cached for inference |

### F2 — Microtubule Attention `[P0]`

| Sub-feature | Requirement |
|---|---|
| GQA | `n_kv_heads=1` default (MQA); configurable |
| Scalar polarity bias | Per-head signed scalar; encodes MT plus/minus end directionality |
| ALiBi GTP log-bias | Geometric γ schedule; 64× receptive-field spread across heads |
| Low-rank bilinear polarity | Opt-in (`polarity_mode="low_rank"`); σ(x Wₐ)(x W_b)ᵀ bilinear mask |
| SDPA backend | `torch.nn.functional.scaled_dot_product_attention`; Flash-Attn automatic |
| KV cache | Causal; position-offset-aware; bit-exact with full-forward (diff < 1e-4) |
| Precomputed buffers | Distance matrix Δ and causal mask as buffers; no per-token allocation |

### F3 — Global Workspace Theory Bottleneck (GWTB) `[P0]`

| Sub-feature | Requirement |
|---|---|
| Compression | d_model → d_gw = d_model/r (default r=8) |
| Workspace self-attention | Causal, multi-head, KV-cached |
| Broadcast | Linear projection + gated residual; γ_bcast=0.01 init |
| Per-block mode | `gwtb_per_block=True` puts GWTB inside every block |
| Cache parity | GWTB cached vs full-forward diff < 1e-4 |

### F4 — GlobalCoherenceLayer `[P0]`

| Sub-feature | Requirement |
|---|---|
| Sparse attention | Top-k retention (sparsity=0.1); causal |
| Collapse gate | `σ((energy − threshold) × 10)`; Orch-OR inspired |
| Diagnostics export | `last_gate` buffer readable by `get_mt_diagnostics()` |
| KV cache | Cache-compatible; cache parity guaranteed |

### F5 — Anesthesia Validation Protocol (AVP) `[P0]`

| Sub-feature | Requirement |
|---|---|
| Runtime hooks | `AnesthesiaController` via `register_forward_hook`; no weight modification |
| Context manager | `with anesthetize(model, level): ...` one-liner API |
| Two effects | (1) MT-DL output × (1-level); (2) coherence deviation × (1-level) |
| Φ̂ proxy | kNN entropy estimator (KSG 2004); L∞ metric; pure PyTorch (no SciPy) |
| Multi-batch averaging | `n_batches=10` default; reduces variance by ~√10 |
| Anesthesia test | Pass if Φ̂(κ=10)/Φ̂(κ=1) ≤ 0.30 (70% collapse); default δ=0.70 |
| CLI | `python eval.py --anesthesia_test --anesthesia_kappas 1 2 5 10` |

### F6 — Training Pipeline `[P0]`

| Sub-feature | Requirement |
|---|---|
| Data pipeline | Memory-mapped `uint16` binary (numpy.memmap); random stride augmentation |
| AMP | BF16 on A100; FP16 fallback |
| torch.compile | Opt-in `--compile`; peel `_orig_mod` for diagnostics |
| Separate LR groups | 4 groups: main (1×), ODE constants (0.33×), polarity (1.67×), lateral (0.33×) |
| W&B | τ/γ/polarity histograms + collapse gate rate; opt-in `--wandb` |
| Checkpointing | Config serialised into `.pt`; `load_model` reconstructs from checkpoint |
| Dummy mode | `--dummy` flag for smoke tests without dataset download |

### F7 — Evaluation `[P0]`

| Sub-feature | Requirement |
|---|---|
| Standard PPL | WikiText-103 word-level perplexity |
| Long-context PPL | Sliding-window with dual KV+h_prev cache; beyond training seq_len |
| MT diagnostics | τ mean/std/min/max, γ, polarity std, lateral off-diag norm, rmc_gate, collapse_gate |
| W_lat heatmaps | Per-layer coupling matrix PNG (matplotlib opt-in) |
| Φ̂ sweep | `phi_hat_anesthesia_sweep()` returns {κ: Φ̂} dict |

### F8 — Test Suite `[P0]`

17 tests, all passing, runnable in < 2 minutes on CPU:

- Shape and forward correctness
- Gradient flow (all parameters)
- KV-cache parity (< 1e-4)
- LNN recurrence active
- Prefill + decode parity
- GQA KV cache size
- MT diagnostics finite
- Low-rank polarity
- Nearest-neighbor coupling
- GWTB bottleneck
- GWTB cache parity
- GWTB per-block mode
- Φ̂ basic (correlated > independent)
- AVP sweep
- Anesthesia collapse
- Protofilament scaling (P=64 ≤ 1.2× P=13)
- Overfit single batch (loss drops ≥10×)

---

## 6. Architecture Snapshot (v1)

```
d_model = 832 = 13 × 64   (Tensor-Core aligned; d_proto = d_head = 64)
n_layers = 12
n_heads = 13  (one head per protofilament)
n_kv_heads = 1  (Multi-Query Attention; 13× KV-cache savings)
n_protofilaments = 13
n_time_scales = 5
gwtb_compression_ratio = 8  → d_gw = 104
gtp_period = 256
~125M parameters
```

---

## 7. Success Metrics

| Metric | Target | Status |
|---|---|---|
| Test suite pass rate | 17/17 | ✅ 17/17 |
| KV-cache parity | diff < 1e-4 | ✅ ~4e-7 |
| WikiText-103 PPL (125M) | < 22 | 🔲 Pending training |
| LRA Pathfinder accuracy | > 70% | 🔲 Pending |
| Anesthesia test (MT-LNN) | Pass (collapse ≥ 70%) | 🔲 Pending trained model |
| Anesthesia test (Transformer) | Fail | 🔲 Pending |
| Φ̂(MT-LNN) > Φ̂(Transformer) | Yes | 🔲 Pending |
| P=64 scaling overhead | < 2× vs P=13 | ✅ 1.2× |
| arxiv paper published | Yes | 🔲 In progress |

---

## 8. Dependencies and Constraints

### Runtime requirements
- Python ≥ 3.10
- PyTorch ≥ 2.1 (for `torch.nn.functional.scaled_dot_product_attention`)
- CUDA ≥ 11.8 for BF16 + Flash-Attn; CPU-only also supported

### Training hardware
- Minimum: single RTX 3090 (24 GB) at d_model=512
- Recommended: single A100-80GB at default 125M config
- Global batch = 8 × 64 = 512 sequences; 100K steps ≈ 10 B tokens seen

### Key design constraints
- **13 is fixed by biology**: n_protofilaments=13 is the thermodynamically stable MT configuration; the architecture uses this as a structural inductive bias, not a tunable hyperparameter
- **d_model must be chosen so d_model/n_protofilaments is a multiple of 8** for Tensor-Core alignment (832, 416, 1040, …)
- **Anesthesia test requires MT parameters**: standard Transformer/LNN cannot pass AVP by design; this is a feature, not a limitation

---

## 9. Open Questions / Risks

| Question | Risk level | Notes |
|---|---|---|
| Will training converge stably at 125M? | Medium | τ/γ instability possible; separate LR groups and gradient clipping mitigate |
| Will Φ̂ be reliably positive for trained model? | Medium | Depends on information integration actually occurring; kNN estimator has variance |
| Does 13-protofilament split help beyond parameter efficiency? | Low | Ablation confirms monotone improvement; qualitative mechanism still theorised |
| Is Orch-OR experimentally validated? | High (scientific debate) | Paper explicitly acknowledges debate; classical MT predictions are sufficient |
| PyTorch 2.x torch.compile compatibility | Low | Tested; `_orig_mod` unwrap handles this |
