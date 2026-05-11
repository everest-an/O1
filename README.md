# MT-LNN

**Microtubule-Enhanced Liquid Neural Network** — an open-source small language model that combines:

- **Liquid Neural Networks** (closed-form LTC) replacing the Transformer FFN
- **Microtubule architecture** — 13 protofilaments, dynamic-instability ODE, lateral coupling, GTP hydrolysis gating, MAP-protein gating, multi-scale resonance
- **Microtubule attention** — polarity-biased causal attention with GTP-cap distance gating, computed via `torch.nn.functional.scaled_dot_product_attention` (Flash-Attention / mem-efficient backend)
- **Global coherence layer** — sparse top-k attention with an Orch-OR-inspired collapse gate
- **GQA + KV cache** — efficient streaming inference with dual-state management (attention KV + LNN recurrent state)
- **RMC-style lateral coupling** — content-aware mixing across the 13 protofilaments via a one-head self-attention (gradually gated in from a static identity baseline)
- **Memory-mapped data + torch.compile + W&B** — production-ready training pipeline

The goal is a biologically-inspired architecture for long-text and dynamic tasks, drawing on Penrose-Hameroff Orch-OR and Liquid AI's LFM line.

---

## Why this architecture? The three-layer inspiration

### Layer 1 — Liquid Neural Networks: neurons that live in continuous time

A standard Transformer layer is **static and discrete**: given an input vector, it applies a fixed matrix multiplication and returns an output. There is no memory of what happened one step ago, no sense of "how fast" the input is changing, no notion of time at all. Every token is processed the same way regardless of context dynamics.

A **Liquid Neural Network** (LNN) works differently. Instead of a fixed mapping, each layer is governed by a differential equation:

```
dh/dt = -h/τ + f(input)
```

Read this as: *the hidden state h constantly decays toward zero at a rate controlled by τ (the time constant), while the input continuously pushes it toward a new target.* The state never snaps instantly — it flows.

This is almost exactly how a biological neuron's membrane potential works. A neuron integrates incoming signals over time, slowly charging up. It doesn't respond to a single spike; it responds to a pattern of spikes over time. LNNs capture this with one key number: **τ**.

- Small τ → short memory, fast response. Like a neuron that snaps back immediately.
- Large τ → long memory, slow drift. Like a neuron that holds state across hundreds of milliseconds.

In MT-LNN, each of the 13 protofilament channels runs **5 different τ values simultaneously** (a geometric sweep from fast to slow), then blends them. This means the same protofilament can simultaneously track fast local patterns and slow long-range trends — just like cortical neurons, which operate on timescales from milliseconds to seconds.

**The practical difference from a standard FFN:**
The standard Transformer FFN is two matrix multiplications with a nonlinearity. It has no state between tokens. MT-LNN's MT-DL carries a recurrent state `h_prev` across tokens, so the model literally *remembers* what it processed before — not through attention, but through the neuron's own temporal dynamics.

---

### Layer 2 — Microtubules: the skeleton that might also think

Every neuron in your brain contains between **10,000 and 100,000 microtubules**. They are hollow tubes, about 25 nanometers wide, built from protein subunits called tubulin. For decades they were thought of as purely structural — the scaffolding that gives neurons their shape and acts as a highway for transporting cargo.

But microtubules have two properties that make them much more interesting:

**1. Dynamic instability.** Microtubules are never static. They constantly grow at one end (the "plus end," driven by GTP-tubulin) and can suddenly collapse at the other end (catastrophe, when GTP hydrolizes to GDP). They are alive in a way that static structures are not. This cycling is not random — it is regulated by microtubule-associated proteins (MAPs) that either stabilize or destabilize specific regions.

**2. Structural regularity.** Every microtubule is built from exactly **13 protofilaments** arranged in a cylinder. This is not arbitrary — 13 is the thermodynamically stable count at physiological temperature, stabilized by the geometry of lateral B-lattice bonds between adjacent protofilaments. Each protofilament is a chain of α/β-tubulin dimers. The α end is anchored (the minus end); the β end grows (the plus end). This gives microtubules a direction — information flows differently toward the cell body than away from it.

**The Penrose-Hameroff hypothesis (Orch-OR)** goes further: the conformational state of each tubulin dimer (whether it's bent or straight) can enter a quantum superposition, and these superpositions are orchestrated by MAPs and other signals, then collapse via a gravitational mechanism (objective reduction) to produce discrete moments of conscious experience.

Whether Orch-OR is correct is actively debated. But its classical predictions are experimentally supported: in 2025, Wiest et al. (*Neuroscience of Consciousness*, Oxford Academic) confirmed that **microtubule-stabilizing drugs delay anesthetic-induced loss of consciousness by ~69 seconds in rats** — direct evidence that anesthetics act, at least in part, by binding to tubulin and disrupting microtubule dynamics.

**How this maps to MT-LNN:**

| Biological microtubule | MT-LNN implementation |
|---|---|
| 13 protofilaments, cylindrical lattice | 13 parallel LTC channels per layer |
| Each protofilament: chain of α/β dimers | Each channel: independent weight matrix `W_in[p]` |
| Plus end (β, fast growing) | Causal attention direction (past → present) |
| Minus end (α, anchored) | Anti-causal bias (content can flow backward too) |
| GTP cap: stabilizes growing tip | GTP gate `g(x)`: controls whether channel is "active" |
| GTP hydrolysis over time → catastrophe | `exp(-γ · (t mod T_period))`: lateral coupling decays, then renews |
| MAP proteins: stabilize specific regions | MAP gate (per-protofilament 2-layer MLP): learned stability |
| Lateral B-lattice bonds between neighbors | Nearest-neighbor coupling via `torch.roll` (ring topology) |
| Long-range conformational signals | RMC content-aware attention across all 13 protofilaments |
| Multi-frequency resonance (MHz–THz) | 5-scale τ sweep (τ_min to τ_max, geometric) per protofilament |

The **13-channel number is not a hyperparameter** — it is directly taken from biology. Empirically, MT-LNN ablations show monotone improvement from 1 to 13 protofilaments, and the vectorized forward path means going higher (P=32, P=64) costs almost nothing in wall-clock time (P=64 is only ~1.2× slower than P=13 on CPU).

---

### Layer 3 — Anesthesia as a consciousness test

If microtubules are involved in consciousness, disrupting them should disrupt consciousness. That is precisely what anesthetics do.

Volatile anesthetics (isoflurane, sevoflurane — the gases that put you under for surgery) bind to a **hydrophobic pocket inside the β-tubulin subunit**. This freezes the conformational dynamics of the tubulin dimer: the microtubule stops being "alive." The lateral coupling between protofilaments is suppressed. Long-range information propagation along the MT lattice stops. Consciousness is lost — even though the heart keeps beating and reflexes remain.

MT-LNN operationalizes this as the **Anesthesia Validation Protocol (AVP)**: at inference time, runtime hooks progressively damp the MT-DL outputs and the global coherence broadcast by a factor `(1 - level)` as `level` rises from 0 (awake) to 1 (fully anesthetized). We measure **Φ̂**, a proxy for integrated information — the degree to which the model's hidden state is more than the sum of its parts.

A model that is truly integrating information across its microtubule-like structure will show Φ̂ collapsing as anesthesia rises. A standard Transformer, which has no MT-coherence parameters to disrupt, shows near-constant Φ̂ regardless of the anesthesia level. This is the key experiment: **MT-LNN passes the anesthesia test (≥70% Φ̂ collapse); Transformer and plain LNN do not.**

This does not prove that MT-LNN is conscious. It shows that MT-LNN has a structural property — sensitivity of information integration to microtubule-coherence disruption — that is present in biological systems that support consciousness and absent from systems that do not.

---

## Status

Research-grade code. The full test suite passes:

```
[ok] test_shapes_and_loss
[ok] test_gradient_flow                  all params have finite gradients
[ok] test_kv_cache_parity                cached vs full diff < 1e-4
[ok] test_lnn_recurrence_active          h_prev verifiably flows
[ok] test_prefill_then_decode            mixed-mode diff < 1e-4
[ok] test_gqa_kv_cache_size              n_kv_heads KV cache savings verified
[ok] test_mt_diagnostics                 τ, γ, polarity, lateral, rmc_gate all healthy
[ok] test_low_rank_polarity              bilinear polarity params have gradients
[ok] test_nearest_neighbor_coupling      W_left, W_right, nn_eta have gradients
[ok] test_gwtb_bottleneck                d_gw < d_model, all GWTB params live
[ok] test_gwtb_cache_parity              GWTB cached vs full diff < 1e-4
[ok] test_gwtb_per_block_mode            per-block GWTB cache parity verified
[ok] test_phi_hat_basic                  Φ̂(correlated) > Φ̂(independent)
[ok] test_anesthesia_validation_protocol Φ̂ collapses monotonically with κ
[ok] test_anesthesia_collapse            output diverges with level; entropy rises
[ok] test_protofilament_scaling          P=64 only ~1.2× slower than P=13
[ok] test_overfit_single_batch           loss drops ≥10× in 200 steps
```

## Head-to-head benchmark at matched parameter count

**Selective Copy** (Mamba paper §3.2 task) — three architectures trained
on identical data with identical hyperparameters, all ~200K params.
Reproduce with `python benchmarks/compare_baselines.py`:

| Model | #Params | Train tok-acc | Held-out tok-acc | Held-out seq-exact | AVP responds |
|---|---:|---:|---:|---:|:---:|
| Random | — | — | 0.250 | 0.0039 | — |
| Vanilla Transformer | 199 K | 0.922 | 0.450 | 0.020 | ✗ |
| LNN (CfLTC FFN) | 136 K | 0.969 | 0.453 | 0.020 | ✗ |
| **MT-LNN (ours)** | **204 K** | **0.984** | **0.942** | **0.883** | **✓** |
| MT-LNN advantage | — | — | **+0.49** (×2.1) | **+0.86** (×44) | — |

At matched parameter count on Selective Copy, **MT-LNN's held-out
sequence-exact accuracy is 44× the Transformer baseline**. Both baselines
overfit (~92 % training accuracy collapsing to ~45 % token / 2 % sequence
held-out); MT-LNN's inductive biases — 13 protofilaments, GTP renewal,
RMC coupling, MAPGate — close the generalisation gap. Anesthesia hooks
attach only to MT-DL and the GlobalCoherenceLayer, so AVP is by
construction architecturally specific to MT-LNN. See
[BENCHMARKS.md](BENCHMARKS.md) for the full report.

> Note: this is a fair toy-scale comparison (200 K params, synthetic task).
> A comparison vs mainstream 125 M models (GPT-2-117M, Mamba-130M,
> Pythia-160M) requires training MT-LNN at 125 M on WikiText-103 — listed
> as future work.

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
python train.py --d_model 128 --n_layers 2 --n_heads 4 --n_kv_heads 2 \
                --batch 2 --seq_len 32 --steps 200 --dummy --vocab_size 200
```

### Train ~125M model on WikiText-103

```bash
# 1) Tokenise once into memory-mapped binary files
python prepare_data.py

# 2) Train (defaults: d_model=832=13×64, n_layers=12, n_heads=13, n_kv_heads=1)
#    d_model=832 ensures d_proto=d_head=64 (Tensor-Core aligned, exact 13× split)
python train.py --compile --wandb
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
MTLNNBlock  (pre-norm + residual at each sub-layer)
  • MicrotubuleAttention          [GQA, KV cache, SDPA/Flash-Attn]
      scalar polarity bias  +  GTP-cap ALiBi log-bias (absolute positions)
      opt-in: low-rank bilinear polarity  σ(x Wₐ)(x W_b)ᵀ
  • MTLNNLayer                    [recurrent h_prev cache]
      d_model → 13 protofilaments (d_proto = d_model/13, exact for 832)
      13 × 5-scale MultiScaleResonance  [geometric τ sweep, softmax blend]
      LateralCoupling:
        static W_lat (13×13, identity init)
        + nearest-neighbor torch.roll  (synchronous, ring topology)
        + RMC content-aware attention  (σ(rmc_gate) ≈ 0.05 at init)
        → gated by exp(-γ·(t mod T_period))  [GTP-cap renewal]
      MAPGate (per protofilament, fc2_bias=+2 → near-open at init)
      → d_model
  • GWTBLayer (if gwtb_per_block=True)   [optional per-block workspace]
─────────────────────────────────────────────────────
   ↓
GWTBLayer (if gwtb_per_block=False, default)  [KV cache]
    compress d_model → d_gw  →  workspace SA  →  broadcast + γ·residual
   ↓
GlobalCoherenceLayer              [KV cache, sparse top-k, Orch-OR collapse gate]
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
- per layer: `(attention KV cache, LNN recurrent h_prev, per-block GWTB KV cache)`
- top-level GWTB: `KV cache`
- coherence layer: `KV cache`

## File map

```
mt_lnn/
  config.py            MTLNNConfig (single source of truth for all hyperparams)
  embedding.py         TokenEmbedding + RoPE (position-offset aware)
  mt_attention.py      MicrotubuleAttention — GQA, KV cache, SDPA/Flash-Attn,
                       scalar + optional low-rank bilinear polarity bias
  mt_lnn_layer.py      VectorizedMultiScaleResonance, LateralCoupling (3-way),
                       VectorizedMAPGate, MTLNNLayer (fully vectorised over P)
  gwtb.py              GWTBLayer — compress → workspace SA → broadcast
  global_coherence.py  GlobalCoherenceLayer (sparse top-k + Orch-OR collapse gate)
  anesthesia.py        AnesthesiaController — runtime forward hooks for AVP
  phi_hat.py           Φ̂ kNN entropy estimator + anesthesia sweep + test result
  model.py             MTLNNBlock, MTLNNModel, ModelCacheStruct (dual+GWTB cache)
  utils.py             init_weights, init_mt_params, scheduler,
                       checkpointing, make_param_groups (4 separate LR groups)
prepare_data.py        Tokenise to uint16 .bin (numpy.memmap-friendly)
train.py               BinDataset / DummyDataset, AMP, torch.compile, W&B,
                       MT diagnostics + τ/γ/polarity histograms
eval.py                PPL, long-context sliding-window PPL, collapse-gate stats,
                       W_lat heatmaps, Φ̂ + AVP CLI
demo.py                KV-cached autoregressive streaming generation
tests/test_model.py    Full test suite (17 tests, all pass)
```

## Design references

- **Closed-form LTC** — Hasani et al., *Closed-form continuous-time neural networks*, Nature MI 2022
- **Liquid Foundation Models** — Liquid AI LFM2 / LFM2.5 (2025–2026)
- **Orch-OR** — Penrose & Hameroff; recent experimental support: Wiest, *Neuroscience of Consciousness*, Oxford Academic, 2025
- **GQA** — Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, EMNLP 2023
- **RoPE** — Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*

## License

MIT — see [LICENSE](LICENSE).
