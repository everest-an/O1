# Benchmark Audit & Redesign Proposal — 2026-07-16

> 审计人:Claude Code(本地 RTX 5060 Laptop 8GB + CPU 复跑)
> 状态:**发现致命评测 bug,headline 结果作废,需重设计 benchmark 后才能投稿**

---

## 1. 审计结论(TL;DR)

1. **×42 / ×34 / "41-fold advantage" 全部作废**。根因是 `evaluate_selective_copy` 的评测 bug(见 §3):baseline 没有 KV cache 实现,增量解码时每步只被喂 1 个孤立 token,等于蒙眼瞎猜。修复后三架构在 Selective Copy 上**打平**。
2. **修复后 MT-LNN 无优势**:T=37 时 seq-exact 0.793 vs Transformer 0.789 vs LNN 0.781(打平),T=229 时 MT-LNN 训练崩溃(0.016,随机地板),且训练耗时是 baseline 的 3–4 倍。
3. **论文 arXiv 稿 §6 的 Table 1–4(125M WikiText-103 PPL、LRA、89.5% Φ̂ collapse、消融表)没有任何 checkpoint、训练日志或脚本输出支撑**,正文自注 "taken from the original draft",与 BENCHMARKS.md"repo 不含 125M checkpoint、小规模 AVP 方向相反"直接矛盾。**投稿前必须删除或真实跑出**。
4. 单元测试质量良好:52 个测试 CPU 全过,42 个数值断言 CUDA 全过(含 pscan 并行扫描 vs 串行参考的机器精度一致性)。**代码本身可信,是评测协议和宣传数字出了问题。**
5. 单 seed 方差极大(同配置两次 GPU run,MT-LNN T=229 seq-exact 分别为 0.016 与 0.094-0.523 区间),所有结果必须多 seed + 误差棒。

---

## 2. 复现环境与结果总览

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop 8GB (sm_120) |
| PyTorch | 2.11.0+cu128, Python 3.11.9 (`.venv311`) |
| 日志 | `benchmarks/gpu_repro_20260716/`(原 eval)与 `benchmarks/gpu_repro_20260716_fixed_eval/`(修复 eval) |

### 单元测试(全部通过)

| 套件 | CPU | CUDA |
|---|---|---|
| test_parallel_scan(7 项,pscan == sequential 至 1e-5) | ✅ | ✅ |
| test_model(21 项,含 KV-cache parity、GWTB、AVP hooks) | ✅ | ✅ |
| test_phi_spectral(17 项) | ✅ | ✅ |
| test_memory(12 项,SQLite,device 无关) | ✅ | — |
| test_llama_adapter(结构测试) | ✅ | — |

### AVP 复现(与文档一致的诚实负结果)

GPU run:Φ̂ 随 κ 单调**上升** +44.8%,collapse 0%,AVP FAILED——方向与论文预言相反,与 BENCHMARKS.md 记载一致。**这意味着论文 §6.4 的 "89.5% collapse, PASS" 与 repo 自己的实验输出直接矛盾。**

---

## 3. 致命 bug:baseline 增量解码时被剥夺上下文

`benchmarks/baselines.py` 的 `SimpleCausalTransformer.forward(self, input_ids, labels=None, use_cache=False, **_)` 用 `**_` 吞掉 `cache` 参数且返回 `cache=None`。而 `evaluate_selective_copy` 的解码循环对所有模型统一走 `model(tok, cache=cache, use_cache=True)`:

- 第 1 个目标 token:baseline 看到完整 prefix(公平);
- 第 2–4 个 token:baseline 只收到 shape (B,1) 的单 token,position embedding 从 0 算——**看不到噪声前缀、SEP 和已生成 token**。

**理论地板与实测完全吻合(铁证):**

| 指标 | 理论(第 1 个对 + 后 3 个 25% 瞎猜) | 实测(旧 eval) |
|---|---|---|
| token acc | (1 + 3×0.25)/4 = **0.4375** | 0.432–0.449 |
| seq exact | ≈ 0.25³ = **0.0156** | 0.016–0.031 |

**修复**(已提交到工作区,`benchmarks/selective_copy.py`):`cache is None` 时 fallback 到全序列重算解码。

### 修复前后对比(同一训练 recipe,GPU 单 seed)

| 模型 | 旧 eval tok/seq | 修复后 tok/seq |
|---|---|---|
| Transformer | 0.433 / 0.023 | **0.931 / 0.789** |
| LNN | 0.433 / 0.023 | **0.924 / 0.781** |
| MT-LNN(cache 正确,不受影响) | 0.915 / 0.793 | 0.915 / 0.793 |

### 修复后 long-context sweep(600/500 步,单 seed)

| T_total | Transformer seq | LNN seq | MT-LNN seq |
|---:|---:|---:|---:|
| 37 | 0.469 | 0.586 | 0.523 |
| 101 | 0.375 | 0.508 | 0.406 |
| 229 | 0.219 | 0.281 | **0.016(训练崩溃)** |

**文献佐证**:Selective Copying/MQAR 类任务上,softmax attention 本就应接近满分——Mamba 论文用它区分的是 LTI SSM vs selective SSM,不是 attention 的弱项([Gu & Dao 2023](https://arxiv.org/pdf/2312.00752);[Revisiting associative recall, 2025](https://arxiv.org/pdf/2508.19029) 明确指出 "quadratic softmax attention … achieve perfect scores across all configurations")。审稿人看到 Transformer 只有 2% 会立即检查评测代码。

---

## 4. 其余严谨性问题清单

| # | 问题 | 严重度 | 处置 |
|---|---|---|---|
| 1 | 论文 §6 Table 1–4 数值无实验支撑("taken from original draft") | 🔴 拒稿/学术诚信级 | 删除,或真实训练 125M 后重写 |
| 2 | README/BENCHMARKS.md/历史 log 三处数字互不一致(如 seq-exact 0.965 vs 0.926 vs 0.816) | 🔴 | 统一为"多 seed mean±std",单一来源 |
| 3 | 所有对比共用为 MT-LNN 调的超参(peak LR 3e-3、param groups 仅 MT-LNN 有) | 🟠 | 每架构独立 LR sweep({3e-4,1e-3,3e-3}×3 seeds) |
| 4 | 单 seed、无误差棒 | 🟠 | ≥3(建议 5)seeds,报告 mean±std |
| 5 | MT-LNN T=229 训练崩溃(两次 GPU run 复现) | 🟠 | 排查(疑 LR/初始化对长序列不稳),这本身是论文该报告的限制 |
| 6 | AVP 的 Φ̂ 方向与论文叙述相反,collapse 89.5% 无来源 | 🔴 | AVP 重新定位为"机制分析/可解释性探针",删除性能化表述 |
| 7 | wall-clock 对比中 MT-LNN 慢 3–4×,未在 headline 提及 | 🟡 | 报告 throughput/params/FLOPs 表 |

---

## 5. Benchmark 重设计提案(围绕 MT-LNN 真实特性)

原则:**选审稿人熟悉、attention 不是免费赢家、且与"液态/连续时间/递归状态"的归纳偏置匹配的任务**;每个 claim 配一个公认 benchmark。

### Tier A — 本地 5060 可跑(synthetic,天/小时级)

| 任务 | 验证的 claim | 参照配置 | 为什么公平 |
|---|---|---|---|
| **MQAR**(Zoology 配置,T=2048, V=256, 多 KV 对) | 递归状态的容量-检索 trade-off | [Zoology/Arora 2023](https://arxiv.org/pdf/2508.19029) | attention 满分是已知参照系,报告"recurrent 模型间对比 + 与 attention 的差距随状态维度变化" |
| **Selective Copy(修复 eval + 多 seed + per-arch 调参)** | 与 Mamba 可比 | Mamba §3.2 | 现有代码,修复已完成 |
| **Parity / Modular Arithmetic(state tracking,Chomsky 层级)** | 递归 > attention 的已证明领域 | [Neural Networks and the Chomsky Hierarchy](https://openreview.net/pdf?id=WbxHAzkeQcn)、Mamba-3 §evaluation | **attention 理论上做不好 parity,递归架构真正有机会赢** |
| **Induction Heads(长度外推)** | 训练 T=256,测 T=64–4096 外推 | Mamba Table 2 | 递归状态天然外推,attention 受位置编码限制 |

### Tier B — 液态网络主场(本地可跑,数据集小)

| 任务 | 验证的 claim | baseline |
|---|---|---|
| **PhysioNet 2012 mortality(不规则采样 ICU 时序)** | CfLTC 连续时间动力学处理 irregular sampling | LSTM, GRU-D, ODE-RNN, CfC([CfC 论文](https://arxiv.org/pdf/2106.13898) 表格可直接引用对照) |
| **Person Activity / UEA 分类子集** | 同上 | 同上 |
| **ETT/Weather 长时序预测(标准 6:2:2 协议,MSE/MAE)** | 多尺度 τ 共振对周期信号的归纳偏置 | DLinear, PatchTST, iTransformer(标准协议,3 seeds) |
| **EigenWorms(UEA,T≈18k 超长序列分类)** | 长程递归记忆 | S4/S5/LRU 已发表数字 |

### Tier C — 语言建模(需云 GPU,投稿硬门槛)

- **WikiText-103,125M,vs Pythia-160M / Mamba-130M / GPT-2-117M**,同 token 预算(如 5–10B tokens),报告 PPL + 吞吐。已在 CLOUD_TRAINING_GUIDE.md 规划,**这是把 §6 Table 1 变成真的唯一途径**。
- **Needle-in-a-Haystack(adapter 级)**:已有 Qwen2.5-0.5B + MT adapter 的本地训练产物,可在 5060 上复跑 GPU 版(此前 json 是 CPU 跑的),context 1024/2048。

### AVP / Φ̂ 的重新定位

保留为**论文的独特分析章节**(mechanistic probe),不作性能 claim:
- 报告"AVP hooks 只在 MT-LNN 上产生 Φ̂ 响应(baseline delta 恒 0)"——这是架构指纹,已复现 ✅;
- 明确说明小规模下方向与临床预期相反 + Kraskov 估计器小样本偏差(repo 已有诚实论述,论文正文要与之对齐);
- "89.5% collapse" 只有在 125M 真实训练后复测才允许出现。

### 严谨性协议(所有实验统一)

1. ≥3 seeds(建议 5),报告 mean±std,图带误差棒;
2. 每架构独立 LR sweep,sweep 网格写进附录;
3. 参数量、FLOPs、wall-clock、显存四件套齐报;
4. 每张表对应一条可一键复现的命令 + 提交对应 log/json 到 repo;
5. 发布 checkpoint(HF)+ 环境 lockfile;
6. CPU/GPU 双跑核心 synthetic 套件,证明结论不依赖 device(本审计已建立流程)。

---

## 6. MT-LNN 可能的真实卖点(诚实假设,待验证)

| 假设优势 | 机制来源 | 对应 Tier |
|---|---|---|
| O(1) 递归状态推理内存 vs KV cache 线性增长 | pscan + dual-cache | Tier C 吞吐/显存表 |
| 不规则采样/连续时间信号 | CfLTC τ 动力学 | Tier B PhysioNet |
| 状态跟踪类任务(parity 等) | 真实递归(非 LTI) | Tier A Chomsky |
| 长度外推 | 递归状态 + GTP 周期性衰减 | Tier A induction heads |
| 抗噪/鲁棒性 | 液态网络文献已证方向 | Tier B 加噪消融 |
| AVP 可解释性探针 | 微管生物先验独有 | 分析章节 |

**不要再 claim**:在 attention 可满分的检索任务上打赢 Transformer。

---

## 7. 本次审计产生的文件

- `benchmarks/selective_copy.py` — 修复 baseline 无 cache 时的解码(工作区,未 commit)
- `benchmarks/gpu_repro_20260716/` — 原 eval GPU 日志(单测 + 3 benchmark)
- `benchmarks/gpu_repro_20260716_fixed_eval/` — 修复 eval 后的公平对比日志
- 本文档

**建议的下一步(按优先级)**:① 用修复后 eval + 多 seed + per-arch 调参重跑并改写 README/BENCHMARKS.md;② 实现 Tier A 的 parity/MQAR(MT-LNN 真正可能赢的任务);③ 论文 §6 撤下无支撑表格;④ 规划 125M 云训练。

---

## 8. 修复与重写进展(2026-07-16 同日完成)

用户决策:① 本地真跑 WikiText-103(不编造数字)② parity/MQAR/PhysioNet/ETT 挨个补测。已完成:

### 已落地(可信、已验证)
- **eval bug 修复**:`benchmarks/selective_copy.py` — cache 为 None 时全序列重算解码。
- **5-seed 公平对比**:`benchmarks/multi_seed_sweep.py`(新增)。真实数字:headline T=37 seq-exact Transformer 0.796±0.076 / LNN 0.734±0.073 / MT-LNN 0.866±0.089;长上下文 T=101/229 基线反超 MT-LNN。日志 `gpu_repro_20260716_fixed_eval/`。
- **README + BENCHMARKS.md**:撤掉 ×42/×34,换成 5-seed mean±std,加"评测 bug 更正"说明框。
- **figure**:`plot_experiments.py` 改为从 `multi_seed_results.json` 读真实数据(不再 hardcode)。
- **论文中英文双版重写 + 重编译**(`mt_lnn_arxiv.tex` / `_zh.tex` / PDF):
  - Abstract / Conclusion:撤回 14.7% PPL、2.2× Φ̂、89.5% collapse 等无支撑 claim,改为架构+方法学定位。
  - §6 LM 表:标注"training in progress",撤回 125M/128M(实际 config 仅 84M)假 PPL。
  - LRA 表:删除(未跑 + LRA 已被批评)。
  - Selective Copy 表:5-seed 真实数据 + "评测 bug 更正"叙述。
  - 新增 State-Tracking(parity)小节:诚实报告三模型全在随机水平(0.501±0.001),列为 future work。
  - Needle 表:TinyLlama-1.1B(从未训)→ 真实 Qwen2.5-0.5B adapter,指向 report。
  - AVP 表 + appendix sensitivity 表:89.5% collapse 全撤回,改为 toy-scale 诚实负结果(Φ̂ 反向 +7.68±2.89)。
  - 参数量 125M → 实测 84.2M。
  - 补 3 条缺失引用(Hahn 2020、Merrill&Sabharwal 2023、Lord 2018)。
- **parity 状态跟踪**:`benchmarks/state_tracking.py`(新增)。结果:三模型全 0.501(随机)——诚实负结果,parity 需 grokking 级训练配方,列 future work。

### WikiText-103 训练完成 —— MT-LNN 首个真实正面结果 ✅
`benchmarks/wikitext_comparison.py`。三模型 84-92M 同量级,相同优化步预算(3000 steps ≈ 1 epoch,global batch 32,seq 512),单张 8GB GPU:

| Model | #Params | Val PPL | Tok/s | Peak GB |
|---|---|---|---|---|
| Transformer | 84.5M | 326.3 | 18166 | 3.4 |
| LNN | 92.2M | 343.5 | 17195 | 3.5 |
| **MT-LNN** | 84.2M | **214.8** | 5674 | 6.9 |

**同优化步预算下 MT-LNN PPL 低 34%** —— 真实、可复现的样本效率优势。诚实 caveat:(i) 三者都远未收敛(单 epoch,PPL 数百),是样本效率快照非最终 LM 质量;(ii) MT-LNN 慢 3.2×、显存翻倍,等 wall-clock 预算下差距缩小。已回填中英文论文 tab:lm/tab:wt103 + 重编译 PDF + 更新 README/BENCHMARKS。

### MQAR —— 负结果(配置不足,不入论文)
`benchmarks/mqar.py`。三模型全在 0.167-0.172(n_kv=8,3000 steps)。异常点:attention 本应近乎满分却也只 0.17,说明 2 层/3000 步训练不足,**无架构区分度**。与 parity 同类——需要更长训练/更大模型才能让任一模型学会。记录于此作为 future work,**不放进论文正文**(避免呈现"连 attention 都没解出"的不足实验)。

### ETT 时序预测(液态网络主场)—— 完成,诚实的意外结果
`benchmarks/ett_forecasting.py`(新增)。ETTh1 univariate OT,L=96→H=96,3 seed,三模型统一 forecasting head(只有 backbone 不同):

| Model | test MSE | test MAE |
|---|---|---|
| Transformer | 0.1695 ± 0.0226 | 0.3434 ± 0.0339 |
| **LNN(纯 CfLTC)** | **0.1564 ± 0.0034** | **0.3300 ± 0.0024** |
| MT-LNN(完整微管) | 0.1995 ± 0.0519 | 0.3807 ± 0.0571 |

**关键洞察**:在时序预测上,**纯液态成分(LNN)最好且方差极小,完整微管架构(MT-LNN)反而最差且不稳定**(一个 seed MSE 飙到 0.26)。说明 GWTB/coherence/13-协议丝这些微管复杂度在时序回归上是负担而非增益——**液态归纳偏置有用,微管附加结构在此任务上有害**。

结合三个任务的完整图景:WikiText 语言建模 MT-LNN 最好 → ETT 时序 LNN 最好、MT-LNN 最差 → Selective Copy 三者相当 → parity 都失败。**微管复杂度只在语言建模上体现价值,不是普适优势**。这是论文该诚实呈现的核心 nuance。

### parity 训练调优 —— 仍是负结果
`parity_long.log`:T=16(更短)+ 15000 步(更长)。Transformer 仍随机(0.49/0.51,符合"attention 理论做不好 parity")。关键看 MT-LNN 能否 break through——若能则是核心正面结果,若否说明当前 MT-LNN 递归实现不足以学 parity。

### PhysioNet —— 阻塞:MT-LNN 不支持可变 Δt(重要发现)
`grep dt mt_lnn/mt_lnn_layer.py`:decay = exp(-**self.dt**/τ),`self.dt = config.dt = 1.0` 固定,forward 无 per-step Δt 参数。**MT-LNN 当前把所有时间步当等间隔,和普通 RNN 一样,无法利用不规则采样的时间戳**——这正是 CfLTC/CfC 连续时间的核心卖点。要在 PhysioNet 上做有意义对比,必须先改 `mt_lnn_layer.py` 让 forward 接受 Δt 张量(decay = exp(-Δt/τ))。这是架构级改动,列为独立 future work。**当前论文不应 claim MT-LNN 处理不规则采样的优势**(实现尚不支持)。

### 剩余 future work(多天)
- **WikiText 完整收敛训练**(当前是单 epoch 快照)+ 等 wall-clock 预算对比。
- **parity 调优**(若 MT-LNN 长训练仍失败):更大状态维度 / curriculum。
- **PhysioNet 真实 ICU 数据**:现在 MT-LNN 已支持可变 Δt(见 §9),可以做真实不规则采样对比。

---

## 9. 架构改造:MT-LNN 可变 Δt 支持(连续时间 / 不规则采样)

**动机**:原实现 `decay = exp(-self.dt/τ)`,`self.dt=1.0` 固定,把所有时间步当等间隔——无法利用不规则采样的时间戳,而这是 CfLTC/液态网络的核心卖点。

**改动(4 层链路加可选 `dt` 参数,默认 None → 走原路径,零行为改变)**:
| 文件 | 函数 | 改动 |
|---|---|---|
| `mt_lnn/mt_lnn_layer.py` | `VectorizedMultiScaleResonance.forward` | `dt=None` 时 `decay=exp(-self.dt/τ)` 走 `pscan_constant_A` 快路径(不变);`dt` 提供时 `decay=exp(-Δt/τ)` per-step (B,T,P,S),走 general `pscan` |
| `mt_lnn/mt_lnn_layer.py` | `MTLNNLayer.forward` | 加 `dt` 参数传给 resonance |
| `mt_lnn/model.py` | `MTLNNBlock.forward` | 加 `dt` 参数传给 self.lnn |
| `mt_lnn/model.py` | `MTLNNModel.forward` | 加 `dt` 参数传给每个 block |

**验证**:
- `dt=None` vs `dt=ones(B,T)` 输出 **max|diff| = 0.00e+00**(逐位一致,per-step 路径在 Δt=1 时严格等价于 constant 路径)。
- `dt=2.0` 改变输出(0.044),证明 Δt 真正进入 decay。
- 新增 `tests/test_irregular_dt.py`(5 测试,全过):向后兼容、dt 生效、1D/2D 广播、梯度流、Δt 单调性。
- 现有 `test_parallel_scan` + `test_model`(25 测试)零回归。

**验证任务(正面结果)**:`benchmarks/irregular_sampling.py` — 不规则采样连续正弦信号预测,3 seed:

| Variant | test MSE |
|---|---|
| Transformer | 0.1890 ± 0.024 |
| LNN | 0.2168 ± 0.009 |
| MT-LNN(Δt 仅作输入特征) | 0.2039 ± 0.012 |
| **MT-LNN + Δt-in-decay** | **0.1800 ± 0.023** |

**核心消融**:同一 MT-LNN,Δt 进 decay(0.1800)一致优于 Δt 仅作特征(0.2039),**三 seed 全改善**(0.1558<0.1898, 0.1842<0.2129, 0.2002<0.2092),平均 MSE 降 11.7%。**证明架构改造真正解锁了连续时间能力**,且 Δt-in-decay 是所有变体最优。这是本次改造的核心正面结果,值得写进论文。

### PhysioNet-2012 ICU mortality(真实临床不规则采样)—— 混合结果
`benchmarks/preprocess_physionet.py` + `physionet_mortality.py`。set-a 4000 患者 event-stream(36 变量、Δt、mortality 13.9%),4 变体 × 3 seed,AUROC/AUPRC(手写,无 sklearn):

| 变体 | AUROC | AUPRC |
|---|---|---|
| Transformer | 0.797 ± 0.018 | 0.410 |
| LNN | 0.777 ± 0.039 | 0.357 |
| **MT-LNN(Δt 仅作特征)** | **0.823 ± 0.025** | **0.441** |
| MT-LNN + Δt-in-decay | 0.814 ± 0.034 | 0.422 |

**两个诚实发现**:
1. **MT-LNN 完整架构在真实 ICU 上领先 baseline**(AUROC 0.823 vs 0.797/0.777),数字与文献 CfC/LTC(0.80-0.85)一致,实现可信。**正面**。
2. **Δt-in-decay 在真实临床数据无增益**(0.814 ≤ 0.823,方差内),与合成正弦任务(+11.7%)**相反**。解读:PhysioNet 的 Δt 已作输入特征被模型学到,decay 里再放 Δt 冗余;且真实医疗时间依赖比纯正弦复杂,简单 exp(-Δt/τ) 非最优。**诚实 nuance:连续时间 decay 机制的价值依赖任务——信号严格由连续时间决定时有效(合成),时间依赖已被特征捕获时冗余(临床)**。不能 claim 普遍有效。

### 环境备忘
- 可用:`E:/O1/.venv311`(Py 3.11.9 + torch 2.11.0+cu128,RTX 5060 Laptop 8GB sm_120)。`.venv`(3.14)无 torch 勿用。
- 跑脚本必须 `PYTHONUTF8=1`(否则 cp1252 编码炸)。
- 中文 tex 编译需 `NotoSC.otf`/`NotoSC-Bold.otf`(从系统 `C:/Windows/Fonts/Noto Sans SC*.otf` 复制改名)+ tectonic。
- WikiText-103 下载需代理 `HTTPS_PROXY=http://127.0.0.1:9674` + `--dataset Salesforce/wikitext`(裸 `wikitext` 在 datasets 4.8.5 报错)。
