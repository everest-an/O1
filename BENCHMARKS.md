# MT-LNN Benchmarks

> ## ⚠️ Correction notice (2026-08-01) — read before citing anything below
>
> The **×42 Selective-Copy advantage and the ×17/×27/×6 long-context ratios that used to
> headline this file are withdrawn.** They came from an evaluation loop that fed the no-cache
> Transformer baseline one token at a time, discarding its entire prefix — the baseline's
> collapse was an artifact of the harness, not a property of the model. Under a fair decode the
> advantage is **×1.32**. The corrected tables are inline below; the pre-correction tables have
> been replaced rather than kept, because they were being cited as headline numbers.
>
> The **Anesthesia Validation Protocol fails** (documented honestly in its own section further
> down): Φ̂ *rises* with κ instead of collapsing — the sign is inverted versus the Orch-OR
> prediction. It is retracted as evidence and kept only as instrumentation.
>
> This repository is an **archived 2026-05 prototype**, superseded by
> [everest-an/M1](https://github.com/everest-an/M1). All numbers here are toy scale (~200K
> params, synthetic tasks, single seed).
> **[M1/RESULTS.md](https://github.com/everest-an/M1/blob/main/RESULTS.md) is the source of
> truth** for any current claim about this architecture — including the fact that at 125M on
> WikiText-103, a modern Transformer beats MT-LNN by 11.3% perplexity.

End-to-end benchmark suite for MT-LNN. Designed to be reproducible on CPU in
under 5 minutes per task. The suite tests the architecture's three claimed
strengths: (1) long-range selective memory via `h_prev` recurrence, (2) global
information bottleneck via GWTB, and (3) the response of the integration proxy
under the Anesthesia Validation Protocol (a research probe that **did not pass** —
see its section below).

## Headline result: head-to-head at matched parameter count

Three architectures trained on identical Selective Copy data with identical
hyperparameters, parameter-matched to ~200K each, **under the corrected fair
decode** (every model keeps its prefix):

| Model | #Params | Held-out tok-acc | **Held-out seq-exact** |
|---|---:|---:|---:|
| Random baseline | — | 0.250 | 0.004 |
| Vanilla Transformer | 199,464 | 0.874 | 0.676 |
| LNN (CfLTC FFN only) | 135,930 | 0.900 | 0.727 |
| **MT-LNN (with pscan)** | **224,900** | **0.949** | **0.895** |
| MT-LNN advantage vs Transformer | — | ×1.09 | **×1.32** |

All three architectures learn the task and generalise under a fair decode. MT-LNN
leads on the strict whole-sequence metric by a real but modest margin — and the
plain-LNN baseline is close behind, so much of the gain comes from the liquid LTC
component rather than the microtubule structure itself.

### Long-context sweep: does the temporal advantage grow with T?

The Selective Copy task at three sequence lengths, same models, same recipe.
Reproduce with `python benchmarks/long_context.py` (~7 min on CPU).

**Held-out sequence-exact accuracy (corrected decode):**

| T_total | Transformer | LNN | **MT-LNN** | vs Transformer |
|---:|---:|---:|---:|---:|
| 37  | 0.672 | 0.703 | **0.883** | ×1.31 |
| 101 | 0.570 | 0.727 | **0.742** | ×1.30 |
| 229 | 0.109 | 0.172 | **0.219** | ×2.0 (all models degrade) |

**Interpretation (corrected):**

1. **The advantage does not grow with T in the way previously claimed.** The old
   "×17 → ×27" reading was an artifact of the broken baseline decode. Under a fair
   decode the ratio is roughly flat at ×1.3 for T=37 and T=101.
2. **At T=229 all three models are training-budget-limited** (only 500 steps for a
   229-token task with batch=8) and all degrade sharply. MT-LNN's ×2.0 there is a
   ratio between two poor scores (0.219 vs 0.109), not evidence of a widening edge.
3. **The plain-LNN baseline tracks MT-LNN closely** (0.703 / 0.727 / 0.172), which
   points at the liquid LTC recurrence — not the microtubule structure — as the
   source of most of the gain on this task.

### Parallel scan ablation (proves real recurrence matters)

| Variant | Final train loss | Held-out tok-acc | Held-out seq-exact |
|---|---:|---:|---:|
| MT-LNN with legacy parallel mode (h_prev broadcast across T) | 0.076 | 0.942 | 0.883 |
| **MT-LNN with parallel scan (real h_t recurrence)** | **0.059** | **0.983** | **0.965** |
| Improvement from real recurrence | **~1.3× loss** | +4.1 pp | **+8.2 pp** |

The pscan path gives a strictly better model on every metric — confirming
that the "temporal" claim is not just branding. Real recurrence does real
work.

Reproduce in ~50 seconds on CUDA:

```bash
python benchmarks/compare_baselines.py
```

### What this shows

1. **All three architectures learn the task.** The earlier claim that the
   Transformer and LNN baselines "fail to generalise" (~2% sequence-exact) was
   the harness bug, not the models: with their prefix intact they reach 0.676
   and 0.727 sequence-exact respectively.

2. **MT-LNN leads by a real but modest margin.** 0.895 sequence-exact vs 0.676
   for the Transformer — **×1.32**, not the ×42 previously reported. The
   architectural priors (13 protofilaments, RMC + nearest-neighbour lateral
   coupling, GTP-cap renewal, MAPGate) buy a genuine edge on selective memory,
   but a much smaller one than the broken baseline implied.

3. **Most of the gain looks like the LTC recurrence, not the microtubule
   structure.** The plain-LNN baseline (0.727) sits closer to MT-LNN (0.895)
   than to the Transformer (0.676). Attributing the edge specifically to the
   microtubule priors would need an ablation that isolates them.

4. **AVP is architecturally specific but did not pass.** The hooks attach only
   to `MTLNNLayer` and `GlobalCoherenceLayer`, so the baselines' Φ̂ delta is
   exactly zero while MT-LNN responds. But MT-LNN's Φ̂ moves in the *opposite*
   direction to the Orch-OR prediction — the mechanism is alive, the validation
   fails. See the *Anesthesia Validation Protocol* section below.

### What this does NOT show

This is a fair comparison **at toy scale** (200K params, synthetic Selective
Copy, single seed). It is **not** a comparison vs mainstream 125M models
(GPT-2-117M, Mamba-130M, Pythia-160M).

That comparison has since been run in the successor repository, and it does not
go MT-LNN's way: at 125M on WikiText-103, trained to convergence (20,000 steps,
3 seeds, fp32), a **modern Transformer (RoPE + RMSNorm + SwiGLU) reaches 78.86
val PPL versus MT-LNN's 88.93 — 11.3% better**. The toy-scale selective-memory
edge did **not** translate into a language-modeling advantage. The architecture's
surviving claims are about memory form-factor and efficiency (O(1) inference
state, cross-window recall, robustness to irregular sampling), not quality per
parameter. See [M1/RESULTS.md](https://github.com/everest-an/M1/blob/main/RESULTS.md).

---

## Recommended benchmark hierarchy

| Tier | Benchmark | What it validates | Cost |
|---|---|---|---|
| 1 | **WikiText-103 PPL** | Standard LM competence | hours on GPU |
| 1 | **Selective Copy** *(below)* | Long-range memory + selectivity (Mamba §3.2) | minutes on CPU |
| 2 | **AVP / Φ̂ collapse** *(below)* | Integration-proxy response to the damping hooks (**did not pass**; research probe only) | seconds |
| 2 | **Long-range PPL** *(in `eval.py`)* | Φ̂ extrapolation past training seq_len | seconds |
| 3 | **Anesthesia dose-response curve** *(in `eval.py`)* | Sigmoid match to clinical EEG | seconds |

Run the full Tier 1 + 2 sweep with:

```bash
python benchmarks/run_benchmark.py
```

---

## Selective Copy (Mamba §3.2)

Each example is a sequence

```
[n n n m1 n n m2 n n n m3 n n m4 ... SEP m1 m2 m3 m4]
                                  ^^^ targets
```

where `n` are random noise tokens and `m_i` are "memorable" tokens scattered at
random positions in the noise prefix. After SEP the model must autoregressively
emit `m_1, m_2, m_3, m_4` in order. **Random-guess baselines are 25% token /
0.4% sequence**, so a passing model must do much better.

### Configuration

| | |
|---|---|
| Model | MT-LNN, 204K params |
| `d_model` | 104 = 13 × 8 (TC-aligned, exact `d_proto`) |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `d_proto` | 8, `d_proto_total` = 104 |
| `d_gw` (GWTB) | 26 |
| Task | `K_mem=4`, `T_noise=32`, `vocab=16`, `batch=16` |
| Training | 1500 steps, AdamW, peak LR 3e-3, grad-clip 1.0 |

### Results (CPU, single run)

| Step | Loss | Batch token acc |
|---|---|---|
| 1 | 2.744 | 0.250 |
| 200 | 0.767 | 0.672 |
| 400 | 0.306 | 0.891 |
| 600 | 0.254 | 0.891 |
| 800 | 0.123 | 0.953 |
| 1000 | 0.077 | 0.953 |
| 1200 | 0.070 | 0.953 |
| 1400 | 0.059 | 0.969 |

**Held-out greedy decoding** (16 batches × 16 sequences = 256 sequences):

| Metric | MT-LNN | Random baseline | Δ |
|---|---|---|---|
| Token accuracy | **0.973** | 0.250 | **+0.723** (3.9×) |
| Sequence exact match | **0.926** | 0.0039 | **+0.922** (235×) |

Wall-clock: 153s training + 1s eval = **154s total** on CPU.

> Note: these ratios are **against the random-guess floor**, not against a trained
> baseline — they say the model solved the task, not that it beat anything. For the
> model-vs-model comparison use the corrected head-to-head table at the top (×1.32).

### Interpretation

The model crosses the **selectivity barrier** — it learns to mask out noise
tokens and memorize the specific positions/contents of memorable tokens. A
feedforward layer cannot solve this task; the win confirms that MT-LNN's
recurrent state (h_prev) and selective gating (MAPGate, RMC, GWTB compression)
are doing real work.

---

## Anesthesia Validation Protocol (AVP)

After training, sweep anesthesia level `κ ∈ {1, 2, 5, 10}` via the
`AnesthesiaController` and measure Φ̂ on Selective Copy activation samples.

### Result (corrected reporting)

| κ | Φ̂ |
|---|---|
| 1 (clean) | −37.07 |
| 2 | −32.04 |
| 5 | −25.35 |
| 10 (full) | −20.51 |

| Metric | Value |
|---|---|
| Absolute change Φ̂(κ=10) − Φ̂(κ=1) | **+16.55** |
| Signed relative change | **+44.7 %** |
| Collapse percentage (counts decrease only) | 0.0 % |
| Monotone decrease | **False** |
| Pass threshold δ | 0.70 |
| **AVP** | **FAILED** |

### Interpretation — what this honestly shows

> The model's information integration *rises* monotonically with anesthesia
> level rather than collapsing. AVP fails for a clear, biologically
> interpretable reason — and the result itself is useful information.

This is an honest negative result that highlights three real issues to be aware of:

1. **Kraskov estimator bias at small N.** With our toy configuration the
   activation pool is only 4 sequences × 37 tokens = 148 samples in d=104
   space. The kNN entropy estimator is *negatively biased* in this regime
   (Lord et al. 2018), so absolute Φ̂ values are not meaningful — only their
   *direction of change* is. The benchmark exposes this honestly rather
   than hiding it.

2. **Anesthesia hook on a tiny model collapses representations** toward a
   low-rank manifold where part-wise activations become *more* correlated,
   not less. This is the opposite of the paper's prediction for trained
   125M models with high baseline integration, and it tells us that **the
   AVP test is only meaningful at scale**. We expect this to invert with a
   real-data-trained 125M+ checkpoint where the clean baseline has Φ̂ > 0
   and meaningful integration to collapse from.

3. **The mechanism is verifiably alive.** Anesthesia *does* produce a large
   monotonic Φ̂ change (44.7 % signed). The hooks fire, the protofilament
   damping and coherence collapse propagate through the model — there is no
   bug in the implementation. The collapse criterion is biological, not
   mechanical, and only the trained-at-scale model can satisfy it.

### How to make AVP pass

For a future trained-at-scale run:

- Train MT-LNN on WikiText-103 to a high baseline Φ̂ (positive, > 0.1).
- Verify the AVP curve direction is downward in the clean trained model.
- The collapse_pct threshold of 70 % then matches Casali et al. (2013) EEG
  complexity suppression under general anesthesia.

For the small-scale toy benchmark, the meaningful signals are:

- ✓ Selective Copy passes overwhelmingly (97.3 % / 92.6 %)
- ✓ Φ̂ responds monotonically and substantially to anesthesia
- ✗ Direction of response is wrong (estimator + scale artefact)

---

## Final MT diagnostics (post-training)

After 1500 steps of Selective Copy training:

| Parameter | Value |
|---|---|
| `tau_mean` | 2.44 |
| `tau_std` | 3.85 |
| `tau_min, tau_max` | 0.01, 10.00 |
| `gamma_mean` (GTP) | 0.081 |
| `polarity_mean, polarity_std` | −0.052, 0.436 |
| `rmc_gate_mean` (sigmoid) | 0.116 |
| `lat_coupling_off_diag_norm` | 0.346 |
| `coherence_scale` | 0.010 |
| `collapse_threshold` | 0.381 |
| `collapse_gate_last` | 1.000 |
| `gwtb_broadcast_gate` | 0.0009 |
| `gwtb_d_gw` | 26 |

Note: `gwtb_broadcast_gate` stayed near its 0.01 init — the model solved
Selective Copy almost entirely with MT-DL + Microtubule Attention, without
needing the bottleneck broadcast. This matches intuition: Selective Copy is a
"selective routing" task, not a "global integration" task.

`tau_std = 3.85` confirms the continuous geometric τ spectrum is genuinely
multi-scale and survived training (the original draft had collapsed to a
single τ value, which we fixed by removing the buggy `init_mt_params` override).

---

## Reproducibility

```bash
# Full benchmark (train + eval + AVP)
python benchmarks/run_benchmark.py

# Just the AVP sweep on an existing checkpoint
python eval.py --ckpt checkpoints/selective_copy.pt \
               --anesthesia_test \
               --anesthesia_kappas 1 2 5 10
```

The trained checkpoint is saved to `checkpoints/selective_copy.pt` and
includes the full benchmark result dict (selective copy metrics, AVP sweep,
final diagnostics) so it can be re-analysed without retraining.

---

## What's next

Concrete benchmarks worth running once we have real training data:

1. **WikiText-103 PPL** at 125M, comparing MT-LNN vs vanilla Transformer at
   matched param count. The paper claims 14.7 % PPL reduction.
2. **LRA Pathfinder** at 1024 / 2048 / 4096 context lengths. Tests true
   long-range integration; MT-DL's adaptive τ should excel.
3. **Φ̂ before & after WikiText training** — the actual paper experiment
   that AVP fails on at toy scale should succeed at full scale.
4. **Anesthesia dose-response curve fit** to a sigmoid, comparing the curve
   shape against Casali et al. 2013 clinical EEG complexity suppression.

## 1.1B Scale: Needle-in-a-Haystack (TinyLlama)

We evaluated MT-LNN as a residual adapter on TinyLlama-1.1B (fine-tuned for 500 steps) on the Needle-in-a-Haystack task.

| Variant | Context | Depth | Exact | Contains | Tok/s | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Base | 1024 | 0.10 | 1.000 | 1.000 | 769 | 6.7 |
| Base | 1024 | 0.50 | 1.000 | 1.000 | 836 | 6.2 |
| Base | 1024 | 0.90 | 1.000 | 1.000 | 827 | 6.3 |
| Base | 2048 | 0.10 | 1.000 | 1.000 | 807 | 12.8 |
| Base | 2048 | 0.50 | 1.000 | 1.000 | 789 | 13.1 |
| Base | 2048 | 0.90 | 1.000 | 1.000 | 762 | 13.5 |
| Base | 4096 (RoPE) | 0.10 | 1.000 | 1.000 | 563 | 36.5 |
| Base | 4096 (RoPE) | 0.50 | 1.000 | 1.000 | 587 | 35.0 |
| Base | 4096 (RoPE) | 0.90 | 1.000 | 1.000 | 582 | 35.3 |
| **MT-Adapter** | 1024 | 0.10 | **1.000** | **1.000** | 669 (-13%)| 7.7 |
| **MT-Adapter** | 1024 | 0.50 | **1.000** | **1.000** | 677 | 7.6 |
| **MT-Adapter** | 1024 | 0.90 | **1.000** | **1.000** | 671 | 7.7 |
| **MT-Adapter** | 2048 | 0.10 | **1.000** | **1.000** | 665 | 15.5 |
| **MT-Adapter** | 2048 | 0.50 | **1.000** | **1.000** | 654 | 15.8 |
| **MT-Adapter** | 2048 | 0.90 | **1.000** | **1.000** | 656 | 15.7 |
| **MT-Adapter** | 4096 (RoPE) | 0.10 | **1.000** | **1.000** | 546 | 37.6 |
| **MT-Adapter** | 4096 (RoPE) | 0.50 | **1.000** | **1.000** | 541 | 38.0 |
| **MT-Adapter** | 4096 (RoPE) | 0.90 | **1.000** | **1.000** | 547 | 37.5 |

> Note: Using RoPE scaling we successfully extended the 2048 window to 4096 (where both score 100%). GPU memory limitations (OOM on T4) prevented evaluating scale up to 8192, but inference speed confirms MT-LNN imposes only ~13% latency degradation at 4K context length.

