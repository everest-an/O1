"""
benchmarks/preprocess_physionet.py — parse PhysioNet-2012 set-a into an
event-stream cache for irregular-sampling mortality prediction.

Each patient record is a list of (time, variable, value) measurements taken at
IRREGULAR times over the first 48h in ICU. We keep the raw event stream (not a
fixed-time-bin grid) precisely so a continuous-time model can consume the real
inter-event gaps Δt. Output is a padded .npz:

    values   (N, L) float32   raw measurement value (standardised at train time)
    var_ids  (N, L) int64     variable index 0..V-1 (pad = V)
    dt       (N, L) float32   hours elapsed since previous event (dt[0]=0)
    lengths  (N,)   int64     true sequence length before padding
    labels   (N,)   int64     in-hospital mortality (0/1)
    record_ids (N,) int64

Static fields (Age/Gender/Height/ICUType/Weight/RecordID) are dropped from the
stream; the task is deliberately about the time-series dynamics.

Usage:
    python benchmarks/preprocess_physionet.py
"""

import csv
import json
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "physionet")
STATIC = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}
MAX_LEN = 512


def parse_time(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)          # minutes


def main():
    set_a = os.path.join(DATA, "set-a")
    files = sorted(os.listdir(set_a))

    # Fixed variable vocabulary (sorted for determinism)
    var_list = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol',
                'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
                'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP',
                'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2',
                'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'pH']
    var_id = {v: i for i, v in enumerate(var_list)}
    V = len(var_list)

    # Labels
    lab = {}
    with open(os.path.join(DATA, "Outcomes-a.txt")) as f:
        for row in csv.DictReader(f):
            lab[row["RecordID"]] = int(row["In-hospital_death"])

    N = len(files)
    values = np.zeros((N, MAX_LEN), dtype=np.float32)
    var_ids = np.full((N, MAX_LEN), V, dtype=np.int64)      # pad id = V
    dt = np.zeros((N, MAX_LEN), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)
    labels = np.zeros(N, dtype=np.int64)
    record_ids = np.zeros(N, dtype=np.int64)

    n_trunc = 0
    for i, fn in enumerate(files):
        rid = fn.replace(".txt", "")
        events = []
        with open(os.path.join(set_a, fn)) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3 or parts[0] == "Time":
                    continue
                t, p, v = parts
                if p in STATIC or p not in var_id:
                    continue
                try:
                    val = float(v)
                except ValueError:
                    continue
                if val < 0:                                 # -1 = missing
                    continue
                events.append((parse_time(t), var_id[p], val))
        events.sort(key=lambda e: e[0])
        if len(events) > MAX_LEN:
            events = events[:MAX_LEN]                       # keep earliest MAX_LEN
            n_trunc += 1
        L = len(events)
        lengths[i] = L
        labels[i] = lab.get(rid, 0)
        record_ids[i] = int(rid)
        prev_t = None
        for j, (tm, vid, val) in enumerate(events):
            values[i, j] = val
            var_ids[i, j] = vid
            dt[i, j] = 0.0 if prev_t is None else (tm - prev_t) / 60.0  # hours
            prev_t = tm

    out = os.path.join(DATA, "cache.npz")
    np.savez_compressed(out, values=values, var_ids=var_ids, dt=dt,
                        lengths=lengths, labels=labels, record_ids=record_ids)
    with open(os.path.join(DATA, "vars.json"), "w") as f:
        json.dump({"var_list": var_list, "V": V, "max_len": MAX_LEN}, f)
    print(f"N={N}  V={V}  max_len={MAX_LEN}  truncated={n_trunc}  "
          f"mortality={labels.mean():.3f}")
    print(f"cache -> {out}")


if __name__ == "__main__":
    main()
