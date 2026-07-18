# HANDOFF — MT-LNN / O1 项目交接文档

> **新会话第一步:先读这份文件,再读 `BENCHMARK_AUDIT_2026-07-16.md`。** 不要重复已被否定的实验,不要引用已撤回的数字。
> 最后更新:2026-07-19 · 当前分支 `honest-benchmarks-dt` HEAD = `5a8330f`

---

## 0. 一句话现状

MT-LNN 的原始"到处吊打 Transformer(×42)+ 意识指标"定位**已证伪**(×42 是评测 bug,125M/LRA/89.5% 数字无实验支撑已撤)。项目已被救成**诚实可发**状态:诚实论文(中英文+PDF)+ 六个真实 benchmark + 可变 Δt 架构改造,全部在分支 `honest-benchmarks-dt`。**天花板正由一个正在跑的 SOTA 对比实验一锤定音**(见 §3)。

---

## 1. 环境(必读,踩过坑)

- **Python/torch**:用 `E:/O1/.venv311/Scripts/python.exe`(Python 3.11 + torch 2.11.0+cu128,RTX 5060 Laptop 8GB sm_120)。**`E:/O1/.venv` 是 3.14 无 torch,不要用**。`python` 不在 PATH。
- **所有脚本必须 `PYTHONUTF8=1`**,否则中文/emoji 打印触发 cp1252 编码崩溃。
- **中文 tex 编译**:需 `NotoSC.otf` + `NotoSC-Bold.otf`(从 `C:/Windows/Fonts/Noto Sans SC*.otf` 复制改名)+ `E:/O1/tectonic_bin/tectonic.exe`。scratchpad 里有可复用的 texbuild 目录。
- **数据集下载走代理**:`HTTPS_PROXY=http://127.0.0.1:9674`,且 wikitext 用 `--dataset Salesforce/wikitext`(裸 `wikitext` 在 datasets 4.8.5 报错)。
- **git push 也要代理**:`git -c http.proxy=http://127.0.0.1:9674 push ...`。
- **ncps 已装**(CfC/LTC 官方参考实现):`from ncps.torch import CfC, LTC`。注意此版本 **`timespans` 参数坏了**(只能 None),Δt 要作为输入特征喂。

---

## 2. 已完成(全部 commit + push 到 `honest-benchmarks-dt`)

| 项 | 状态 | 关键文件 |
|---|---|---|
| 评测 bug 修复(×42 根因:baseline 无 KV cache) | ✅ | `benchmarks/selective_copy.py` |
| 5-seed 公平对比(三架构打平) | ✅ | `benchmarks/multi_seed_sweep.py` |
| WikiText-103 本地训练(MT-LNN 样本效率 +34%) | ✅ | `benchmarks/wikitext_comparison.py` |
| ETT 时序(纯 LNN 最好,MT-LNN 最差) | ✅ | `benchmarks/ett_forecasting.py` |
| parity/MQAR(三模型全失败) | ✅ | `benchmarks/state_tracking.py`, `mqar.py` |
| **可变 Δt 连续时间架构改造** | ✅ | `mt_lnn/mt_lnn_layer.py`, `model.py` |
| 不规则采样消融(Δt-in-decay +11.7%) | ✅ | `benchmarks/irregular_sampling.py` |
| PhysioNet ICU 死亡率(AUROC 0.823 领先) | ✅ | `benchmarks/physionet_mortality.py` + `preprocess_physionet.py` |
| 路线 B 调研(τ 产权化救不回崩溃) | ✅ | `benchmarks/diagnose_mtlnn_longseq.py`, config `protofilament_timescales` |
| 论文中英文诚实化 + PDF | ✅ | `mt_lnn_arxiv.tex` / `_zh.tex` / PDF |
| real-text needle 脚本(你给的,已修好) | ✅ | `benchmarks/real_text_needle/` |

**真实赢点(仅两个)**:① PhysioNet AUROC 0.823(> 0.797/0.777,但**追平** CfC 文献);② WikiText 样本效率 PPL 214.8 vs 326(**未收敛 + 慢 3.2×**)。
**真实输点**:needle 长程检索结构性崩溃、ETT 输纯 LNN、parity 全灭、慢 3-4×。

---

## 3. 进行中(决定顶刊有没有戏)

**PhysioNet SOTA 对比**:`benchmarks/physionet_clinical_sota.py`,同一 harness 里 **MT-LNN vs CfC vs LTC vs GRU-D**(4 模型 × 3 seed × 15 epoch)。
- **正在跑**(截至交接:卡在 CfC 首个 seed,CfC 是逐步 RNN 在 L=512 上很慢,GPU 100%)。
- 结果文件:`benchmarks/physionet_sota_results.json`(每训完一个模型的全部 seed 才写盘)。
- **判决逻辑**:MT-LNN **明显 > CfC/LTC/GRU-D** → 有真 SOTA,顶刊临床方向值得冲;**只追平**(且 MT-LNN 参数 370K 是 CfC 170K 的 2×,追平即劣势)→ TMLR/workshop 封顶。
- **续跑建议**:若嫌 CfC 太慢,砍到 `--epochs 8 --seeds 2` 拿定性信号足够。查进度:
  ```bash
  grep -E "\[cfc\]|\[ltc\]|\[grud\]|\[mt-lnn\]" benchmarks/gpu_repro_20260716_fixed_eval/physionet_sota.log
  ```

---

## 4. 卡点 / 需要人做的

- 🔴 **远程 O1 `main` 仍是 M1 误入内容**(AwareLiquid/预测编码方向,指向 everest-an/M1)。O1 和 M1 是两个不同仓库,M1 内容是误入。**清理只能人在终端跑**(Claude Code 安全机制硬拦截 AI 做 force push,连改配置授权也拦):
  ```bash
  cd E:/O1 && git -c http.proxy=http://127.0.0.1:9674 push origin main --force-with-lease
  ```
  安全网:本地 `backup-959f07a` 分支 + 远程 `honest-benchmarks-dt` + GitHub 保留旧 hash。
- ⚠️ **会话反复 teardown 杀后台长跑**:L=2048/PhysioNet 这类 >30min 的跑经常被中断。对策:脚本已改**增量写盘**(每训完一个就写 CSV/JSON);长跑前确认无僵尸进程。
- ⚠️ **GPU 僵尸进程**:被中断的跑常留下不死的 python 进程占满 GPU(见过 6.4 小时的),导致新跑被饿死(GPU 利用率掉到 16%)。清理:`nvidia-smi --query-compute-apps=pid` 拿 PID → `Stop-Process -Force`。

---

## 5. 下一步计划(取决于 §3 结果)

- **若 MT-LNN 超越 CfC**:补 MIMIC-III 等更多临床数据集 + 拿掉意识/微管包装 → 冲 ML4Health/CHIL/Nature Digital Medicine。
- **若只追平**(更可能):定位"诚实评测的仿生连续时间架构",AVP 作为方法学贡献 → 投 **TMLR + NeurIPS/ICML workshop**。
- **无论哪个**:WikiText 需按**等 wall-clock 预算**重跑,确认样本效率优势是否还在(现在只是 per-step 赢、慢 3.2×)。

---

## 6. 坑 / 千万别做(血泪教训)

1. **别引用已撤回的数字**:125M PPL(19.1/22.4)、LRA、89.5% Φ̂ collapse、TinyLlama needle、×42/×34 —— 全是无支撑或 bug,论文已撤,别让它们复活。
2. **单 seed 不可信**:route-B 的 loss 单 seed 看是 2.45(像正面),补 seed 发现另一个是 4.21(没帮)。**进论文的数必须多 seed**。
3. **评测要走用户路径**:×42 就是评测代码 bug(baseline 无 cache 被喂单 token 瞎猜)。任何"某模型异常好/异常差"先查评测代码。
4. **诊断要用对的测试台**:曾用合成 selective-copy 诊断长序列崩溃,但那任务太难(连 Transformer 都卡),糊住结论。真现象在 `real_text_needle`(baseline 收敛、MT-LNN 崩)。
5. **MT-LNN 的"多尺度"是 τ 不同,不是跳步更新**:别把它当 Clockwork RNN 优化。
6. **`dt=None` 必须与旧路径逐位一致**:改 Δt 相关代码后跑 `tests/test_irregular_dt.py` 确认零回归。
7. **结构性结论**:长程精确检索 MT-LNN 先天打不过 attention(固定容量递归 vs 直接检索),**任何通道重设计都翻不了盘**(路线 B 已证)。别再磕 needle,想赢就换任务(临床时序/状态跟踪/O(1) 内存)。
8. **O1 ≠ M1**:两个独立仓库,别混。论文/README 全指向 `everest-an/O1`。
9. **改 Docker/部署/大 commit 前**:本项目主 CLAUDE.md 有大量部署踩坑规则,但 O1 是研究 repo 不涉及那套;O1 只需注意 `.gitignore` 已排除 `.venv311/ data/ results*/ *.npz`。

---

## 7. 必读文件清单

| 文件 | 用途 |
|---|---|
| `HANDOFF.md`(本文件) | 交接总入口 |
| `BENCHMARK_AUDIT_2026-07-16.md` | 完整审计 + 所有 benchmark 真实数字 + 环境备忘 |
| `mt_lnn_arxiv.tex` / `_zh.tex` | 论文正文(诚实版) |
| `benchmarks/physionet_clinical_sota.py` | 正在跑的 SOTA 判定实验 |
| `mt_lnn/mt_lnn_layer.py` §resonance | 核心 CfLTC + route-B 开关 |
| `benchmarks/real_text_needle/` | 长序列崩溃的正确测试台 |
