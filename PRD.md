# MT-LNN Product Requirements Document

**Version:** 1.1  
**Date:** 2026-05-12  
**Status:** Active  
**Repo:** https://github.com/everest-an/O1

---

## 1. Overview / 概述

**What is MT-LNN? (什么是 MT-LNN？)**
MT-LNN (Microtubule-Inspired Liquid Neural Network) is an open-source small language model (~125 M parameters) that embeds three advanced neuroscientific theories — microtubule dynamics, Global Workspace Theory (GWT), and Integrated Information Theory (IIT) — directly into a trainable neural architecture. 
> *中文简述：MT-LNN (受微管启发的液态神经网络) 是一个开源的小型语言模型（约1.25亿参数），它将三大前沿神经科学理论——微管动力学、全局工作区理论 (GWT) 和 信息整合理论 (IIT) ——直接嵌入到了可训练的神经网络架构中。*

**The Product Core: (产品核心：)**
The primary deliverable is a **research artefact**: a robust Python/PyTorch codebase, trained weights, and an arXiv paper. Together, they constitute the first publicly reproducible deep learning architecture designed to pass a computational "anesthesia-based consciousness test." 
> *中文简述：核心交付物是一个**研究级基线系统 (Research Artefact)**：包括极具鲁棒性的 PyTorch 代码库、预训练权重以及一篇 arXiv 论文。这是世界上第一个能够通过计算学层面“麻醉意识测试”的公开可复现深度学习架构。相比标准 Transformer 那种纯数学的矩阵乘法，MT-LNN 从物理和动态上模仿了大脑微管的信息处理方式，为 AI 提供了一个“更像大脑”的底层基座。*

By physically mimicking how the brain's microtubule structures process information statically and dynamically, MT-LNN provides a literal "brain-like" foundation for AI, unlike the rigid, math-centric matrix multiplications of standard Transformers.

---

## 2. Problem Statement & Product Value / 痛点与产品价值

### 2.1 The Current AI Bottleneck / 当前AI的瓶颈

All mainstream large language models (GPT, Llama, Qwen, Claude, etc.) share the same core Transformer loop:
```
token → static matrix multiply (FFN) → attention → repeat
```
While incredibly powerful for language, this design has major limitations:
- **No Temporal Dynamics (缺乏时间动态):** Every token is processed statically. There is no concept of "time flow" or state accumulation within the layer itself. (每一个 token 的处理都是静态的，层内部根本没有“时间流动”或状态积累的概念。)
- **No Biological Plausibility (缺乏生物学基础):** The architecture represents a purely mathematical solution and lacks connection to biological inductive biases associated with higher-level cognition. (这纯粹是数学运算，缺乏与高级认知相关的生物学归纳偏置。)
- **Consciousness Evaluation Gap (意识评估空白):** AI alignment and AGI research lacks a standard, open framework to measure architectural similarities to consciousness (like information integration and global workspace broadcast). (AI 对齐和 AGI 研究一直缺乏一种开放的评估框架，来衡量 AI 架构与意识体在“信息整合”及“全局广播”上的相似度。)

### 2.2 The MT-LNN Solution & Product Value / MT-LNN的解决方案与价值

MT-LNN replaces the static feed-forward layer of a Transformer with a continuously flowing "Microtubule Dynamic Layer" based on Closed-form Liquid Neural Networks (LNN). 
> *MT-LNN 将 Transformer 静态的前馈层 (FFN) 彻底替换为基于“闭式液态神经网络 (LNN)”的、持续流动更新的“微管动态层”。*

**Product Value Delivered (产品交付价值):**
1. **For Neuroscience & AGI Researchers (面向神经科学与AGI研究员):** Provides a ready-to-run, 125M-parameter PyTorch baseline bridging LLMs and the Penrose-Hameroff Orch-OR theories. (提供了一个开箱即用的大模型代码基线，将大语言模型与 Penrose-Hameroff 的 Orch-OR 意识理论真正折叠到了一起。)
2. **For General ML Practitioners (面向通用机器学习从业者):** Proves that dropping a completely dynamic, non-Transformer layer into LLMs is commercially viable, performant (runs efficiently), and scale-ready. (证明了将纯动态流式的非 Transformer 层塞进 LLM 是可行的、高性能的且易于扩展的。)
3. **The Anesthesia Benchmark (首创“麻醉”基准测试):** Introduces a brand-new way to test AI using a simulated drug test (The Anesthesia Validation Protocol). (引入了一种全新的 AI 测评方式——给 AI “打麻药”并观察其内部信息剥离状态的 AVP 测试。)

| Gap in current AI | MT-LNN Capability |
|---|---|
| No open model with MT-inspired dynamics | Researchers must start from scratch |
| No standard consciousness-adjacent evaluation | Cannot compare architectures on this axis |
| Anesthesia-as-test has no computational analogue | Wiest et al. 2025 finding has no AI implementation |
| LNN and consciousness research exist in separate silos | No bridge architecture exists |

---

## 3. Goals and Non-Goals / 目标与非目标

### Goals (P0 — must have for v1) / 核心目标 (V1必做)
- **G1** — Implement the full MT-LNN architecture in PyTorch, fully tested, open-source (实现并开源全套 MT-LNN 架构)
- **G2** — Provide a working training pipeline (提供单 A100 上可运行的训练和数据处理流)
- **G3** — Implement and document the Anesthesia Validation Protocol (AVP) with Φ̂ metric (实现带 $\hat{\Phi}$ 指标的麻醉验证协议)
- **G4** — Publish a complete arxiv paper describing the architecture, motivation, and results (发布 Arxiv 学术论文以提供引用源)
- **G5** — Achieve 17/17 test suite pass rate; all tests runnable in < 2 minutes on CPU (测试用例 17/17 全部通过，支持 CPU 快速验证)

### Non-Goals / 非目标 (不打算做的)
- **NG1** — This project does NOT claim MT-LNN is conscious or has subjective experience (不宣称本模型真的具备意识感知)
- **NG2** — Not a production-ready model; no safety/alignment tuning is planned for v1 (不是业务生成级模型，不包含安全对齐/RLHF微调)

---

## 4. Users and Personas / 目标用户画像

### Primary: AI × Neuroscience researcher / 核心：AI×神经科学 的交叉学科研究员
- Needs: reproducible baseline, good documentation, clean PyTorch code, arxiv citation (需要完全可复现的代码基线，干净清晰的 PyTorch 网络，可供引用的文献)

### Secondary: Consciousness-adjacent AI researcher / 次要：意识导向的 AI 理论研究员
- Needs: the Φ̂ metric, the anesthesia test, the GlobalCoherenceLayer collapse gate (需要模型内封装好的 $\hat{\Phi}$ 计算评估方法、微管麻醉机制和全局相干层坍缩门)

### Tertiary: LNN / Liquid AI practitioner / 第三梯队：液态神经网络(Liquid AI) 开发者
- Needs: drop-in replacement for Transformer FFN; standard benchmarks (极其需要能平替掉 Transformer FFN 的可矢量化的动态层代码，去搞研究魔改)

---

## 5. Feature Requirements / 具体特性需求

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

## 7. Success Metrics / 验收度量指标

| Metric (指标维度) | Target (目标) | Status (状态) |
|---|---|---|
| Test suite pass rate (单元测试集) | 17/17 | ✅ 17/17 |
| KV-cache parity (增量缓存误差) | diff < 1e-4 | ✅ ~4e-7 |
| WikiText-103 PPL (125M) (词级困惑度) | < 22 | 🔲 Pending training |
| LRA Pathfinder accuracy (寻路探知率) | > 70% | 🔲 Pending |
| Anesthesia test (MT-LNN) (模型麻醉结果) | Pass (collapse ≥ 70%) | 🔲 Pending trained model |
| Anesthesia test (Transformer) (变压器麻醉结果) | Fail (失效) | 🔲 Pending |
| Φ̂(MT-LNN) > Φ̂(Transformer) (积分强度) | Yes (是) | 🔲 Pending |
| P=64 scaling overhead (64纤维计算开销) | < 2× vs P=13 | ✅ 1.2× |
| arxiv paper published (ArXiv 论文发表) | Yes (是) | 🔲 In progress |

---

## 8. Dependencies and Constraints / 环境约束指引

### Runtime requirements (运行环境)
- Python ≥ 3.10
- PyTorch ≥ 2.1 (for `torch.nn.functional.scaled_dot_product_attention`)
- CUDA ≥ 11.8 for BF16 + Flash-Attn; CPU-only also supported

### Training hardware (算力设备)
- Minimum: single RTX 3090 (24 GB) at d_model=512
- Recommended: single A100-80GB at default 125M config
- Global batch = 8 × 64 = 512 sequences; 100K steps ≈ 10 B tokens seen

### Key design constraints (研发底线红线)
- **13 is fixed by biology**: n_protofilaments=13 is the thermodynamically stable MT configuration; the architecture uses this as a structural inductive bias, not a tunable hyperparameter
- **d_model must be chosen so d_model/n_protofilaments is a multiple of 8** for Tensor-Core alignment (832, 416, 1040, …)
- **Anesthesia test requires MT parameters**: standard Transformer/LNN cannot pass AVP by design; this is a feature, not a limitation

---

## 9. Open Questions / Risks / 悬而未决的风险点

| Question (问题描述) | Risk level (风险) | Notes (预估对策) |
|---|---|---|
| Will training converge stably at 125M? | Medium | τ/γ instability possible; separate LR groups and gradient clipping mitigate |
| Will Φ̂ be reliably positive for trained model? | Medium | Depends on information integration actually occurring; kNN estimator has variance |
| Does 13-protofilament split help beyond parameter efficiency? | Low | Ablation confirms monotone improvement; qualitative mechanism still theorised |
| Is Orch-OR experimentally validated? | High (scientific debate) | Paper explicitly acknowledges debate; classical MT predictions are sufficient |
| PyTorch 2.x torch.compile compatibility | Low | Tested; `_orig_mod` unwrap handles this |
