"""Unit-test train.prune_old_checkpoints -- the fix for the Kaggle disk-full
crash (v4 trained fine to step 18000 then failed writing a checkpoint with
"No space left on device" because every 2.4 GB ckpt was kept).

Verifies the REAL function (imported from train.py), not a copy:
  - keeps exactly the K newest ckpt_<step>.pt files
  - removes the oldest ones (zero-padded names => lexical order == step order)
  - never touches final.pt or unrelated files
  - keep_last=0 disables pruning; fewer files than K is a no-op

Run:  python _test_ckpt_prune.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import prune_old_checkpoints

R = {}


def make(d, name):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"x")          # content irrelevant; the function keys off names
    return p


def names(d):
    return sorted(os.listdir(d))


# --- case 1: 8 step-ckpts + final.pt + an unrelated file, keep_last=3 ---
d = tempfile.mkdtemp()
steps = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
for s in steps:
    make(d, f"ckpt_{s:06d}.pt")
make(d, "final.pt")
make(d, "notes.txt")

removed = prune_old_checkpoints(d, keep_last=3)
left = names(d)
R["case1_removed"] = sorted(os.path.basename(p) for p in removed)
R["case1_left"] = left
R["case1_ok"] = (
    R["case1_removed"] == [f"ckpt_{s:06d}.pt" for s in (2000, 4000, 6000, 8000, 10000)]
    and left == ["ckpt_012000.pt", "ckpt_014000.pt", "ckpt_016000.pt",
                 "final.pt", "notes.txt"]
)

# --- case 2: keep_last=0 disables pruning ---
d2 = tempfile.mkdtemp()
for s in (2000, 4000):
    make(d2, f"ckpt_{s:06d}.pt")
removed2 = prune_old_checkpoints(d2, keep_last=0)
R["case2_ok"] = (removed2 == [] and len(names(d2)) == 2)

# --- case 3: fewer files than keep_last -> no-op ---
d3 = tempfile.mkdtemp()
make(d3, "ckpt_002000.pt")
removed3 = prune_old_checkpoints(d3, keep_last=3)
R["case3_ok"] = (removed3 == [] and names(d3) == ["ckpt_002000.pt"])

import json
print(json.dumps(R, ensure_ascii=False, indent=2))
ok = R["case1_ok"] and R["case2_ok"] and R["case3_ok"]
print("PRUNE_PASS" if ok else "PRUNE_FAIL")
sys.exit(0 if ok else 1)
