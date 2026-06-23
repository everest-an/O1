"""O1 native MT-LNN pretrain -- 200M from-scratch causal LM.

Trains the native (no-Transformer) MT-LNN foundation model end to end on a
Kaggle T4. This is the O1 hardware/edge track: a from-scratch liquid-dynamics
language model, distinct from the M1 TinyLlama adapter.

Config (Tensor-Core aligned, all GWTB/proto constraints satisfied):
    d_model = 1248 = 13 x 96   -> d_proto = d_head = 96 (multiple of 8)
    n_layers = 17, n_heads = 13, GQA n_kv_heads = 1
    -> 203.2M parameters, all useful. The ~62.7M direct-extraction head
    (target_head) is disabled for from-scratch LM pretraining via
    config.use_target_head=False (train.py derives this automatically when no
    --train_target_head / --target_loss_weight is passed), so the whole budget
    funds the recurrent body instead of a head plain LM training never touches.

Corpus: wikitext-103-raw-v1 (gpt2 BPE, ~100M tokens). This first run validates
the full 200M pipeline end to end and produces coherent-text checkpoints; the
corpus can be scaled up on later runs. A single T4 session cannot fully pretrain
a 200M model -- the goal here is a working pipeline + flowing checkpoints.

T4 memory budget (why batch=2 / seq_len=512): a first launch at batch=6 /
seq_len=1024 OOM'd on the FIRST forward -- the model itself fits, but the liquid
recurrent activations (13 protofilaments x 5 time-scales, retained for backward
across 17 layers) plus the dense (B,H,S,S) attention-bias tensors filled all
15.9 GiB of a T4 before backward even ran. train.py already runs AMP (fp16 on
T4), so the only safe lever is batch x seq_len. batch=2, seq_len=512,
grad_accum=16 keeps an effective batch of 32 sequences (16,384 tokens/step) with
comfortable headroom; context length can scale up on a bigger GPU later. We also
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (recommended by the OOM
message) to avoid allocator fragmentation.

Output: /kaggle/working/out/*.pt  (download, drop into checkpoints/).
Run as a Kaggle script kernel (GPU T4, internet on).
"""

import os
import subprocess
import sys

# Reduce CUDA allocator fragmentation on the T4 (the OOM message recommends this).
# Set before torch initialises CUDA; inherited by the train.py subprocess.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = "https://github.com/everest-an/O1.git"
DIR = "/kaggle/working/O1"

if not os.path.exists(DIR):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, DIR])
os.chdir(DIR)
subprocess.check_call(["git", "log", "-1", "--oneline"])

# Kaggle's current default torch (2.10+cu128) dropped CUDA kernels for older
# GPUs such as the P100 (sm_60) that sessions are sometimes assigned, causing
# "CUDA error: no kernel image is available for execution on the device". Install
# a torch whose wheels ship sm_60..sm_90 so the job runs on whatever GPU Kaggle
# hands out (P100 / T4). Install torch LAST so it wins dependency resolution.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets", "transformers", "numpy",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
])

import torch  # noqa: E402

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
else:
    print("GPU: CPU")

os.makedirs("/kaggle/working/out", exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Pre-tokenise corpus -> data/{train,validation}.bin + meta.json
# ---------------------------------------------------------------------------
if not os.path.exists("data/meta.json"):
    subprocess.check_call([
        sys.executable, "prepare_data.py",
        "--dataset", "wikitext",
        "--config", "wikitext-103-raw-v1",
        "--tokenizer", "gpt2",
        "--out_dir", "data",
    ])

# ---------------------------------------------------------------------------
# 2) Train the 200M native model
# ---------------------------------------------------------------------------
cmd = [
    sys.executable, "train.py",
    "--d_model", "1248",
    "--n_layers", "17",
    "--n_heads", "13",
    "--n_kv_heads", "1",
    "--seq_len", "512",
    "--batch", "2",
    "--grad_accum", "16",
    "--lr", "6e-4",
    "--warmup_steps", "1000",
    "--steps", "24000",
    "--save_every", "2000",
    "--eval_every", "1000",
    "--ckpt_dir", "/kaggle/working/out",
    "--data_dir", "data",
]
print("RUN:", " ".join(cmd))
subprocess.check_call(cmd)

print("=== output checkpoints ===")
subprocess.check_call(["ls", "-la", "/kaggle/working/out"])
print("O1_PRETRAIN_DONE")
