"""O1 native MT-LNN pretrain -- 200M from-scratch causal LM.

Trains the native (no-Transformer) MT-LNN foundation model end to end on a
Kaggle T4. This is the O1 hardware/edge track: a from-scratch liquid-dynamics
language model, distinct from the M1 TinyLlama adapter.

Config (Tensor-Core aligned, all GWTB/proto constraints satisfied):
    d_model = 1248 = 13 x 96   -> d_proto = d_head = 96 (multiple of 8)
    n_layers = 18, n_heads = 13, GQA n_kv_heads = 1
    -> 200.8M parameters.

Corpus: wikitext-103-raw-v1 (gpt2 BPE, ~100M tokens). This first run validates
the full 200M pipeline end to end and produces coherent-text checkpoints; the
corpus can be scaled up on later runs. A single T4 session cannot fully pretrain
a 200M model -- the goal here is a working pipeline + flowing checkpoints.

Output: /kaggle/working/out/*.pt  (download, drop into checkpoints/).
Run as a Kaggle script kernel (GPU T4, internet on).
"""

import os
import subprocess
import sys

REPO = "https://github.com/everest-an/O1.git"
DIR = "/kaggle/working/O1"

if not os.path.exists(DIR):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, DIR])
os.chdir(DIR)
subprocess.check_call(["git", "log", "-1", "--oneline"])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets", "transformers", "numpy",
])

import torch  # noqa: E402

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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
    "--n_layers", "18",
    "--n_heads", "13",
    "--n_kv_heads", "1",
    "--seq_len", "1024",
    "--batch", "6",
    "--grad_accum", "8",
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
