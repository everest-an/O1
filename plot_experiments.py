"""
plot_experiments.py — Figure 2 (fig_experiments) from REAL benchmark data.

Reads benchmarks/multi_seed_results.json (produced by
`python benchmarks/multi_seed_sweep.py`) and renders three panels:

  Left:   Selective Copy held-out seq-exact, 5-seed mean +/- std (headline run)
  Center: Long-context sweep — seq-exact vs T_total, mean +/- std
  Right:  AVP response — MT-LNN Phi_hat at kappa=1 vs kappa=10 (baselines: no hooks, delta = 0)

Every number in the figure comes from the JSON; nothing is hard-coded.
"""

import json
import statistics

import matplotlib.pyplot as plt
import numpy as np

# Nature-style configuration
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42
})

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "ours_base":  "#E4CCD8",
    "ours_large": "#F0C0CC",
    "red_strong": "#B64342",
    "green_strong": "#2E9E44",
}
MODEL_COLORS = {
    "Transformer": PALETTE["baseline_dark"],
    "LNN":         PALETTE["baseline_mid"],
    "MT-LNN":      PALETTE["red_strong"],
}
MODELS = ["Transformer", "LNN", "MT-LNN"]

with open("benchmarks/multi_seed_results.json", encoding="utf-8") as f:
    R = json.load(f)


def ms(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


fig = plt.figure(figsize=(7.2, 2.5))

# ---------------------------------------------------------------------------
# Panel 1: headline Selective Copy seq-exact (mean +/- std over seeds)
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(131)
means, stds = [], []
for m in MODELS:
    mu, sd = ms([r["seq_exact"] for r in R["headline_T37"][m]])
    means.append(mu)
    stds.append(sd)

x = np.arange(len(MODELS))
ax1.bar(x, means, 0.55, yerr=stds, capsize=3,
        color=[MODEL_COLORS[m] for m in MODELS])
ax1.axhline(0.25 ** 4, ls=":", color="#A8A8A8", lw=1)
ax1.text(0.02, 0.25 ** 4 + 0.015, "random", fontsize=6, color="#888888")
ax1.set_xticks(x)
ax1.set_xticklabels(MODELS, rotation=12)
ax1.set_ylabel("Held-out seq-exact")
ax1.set_ylim(0, 1.0)
ax1.set_title("Selective Copy (T=37, 1500 steps)")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ---------------------------------------------------------------------------
# Panel 2: long-context sweep, seq-exact vs T_total
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(132)
sweep = [("longctx_T37", 37), ("longctx_T101", 101), ("longctx_T229", 229)]
for m in MODELS:
    mus, sds = [], []
    for key, _t in sweep:
        mu, sd = ms([r["seq_exact"] for r in R[key][m]])
        mus.append(mu)
        sds.append(sd)
    ts = [t for _k, t in sweep]
    ax2.errorbar(ts, mus, yerr=sds, marker="o", capsize=3,
                 label=m, color=MODEL_COLORS[m])
ax2.set_xlabel("$T_\\mathrm{total}$")
ax2.set_ylabel("Held-out seq-exact")
ax2.set_xscale("log")
ax2.set_xticks([37, 101, 229])
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_title("Long-context sweep (600/500 steps)")
ax2.legend(frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ---------------------------------------------------------------------------
# Panel 3: AVP — Phi_hat response to anesthesia (MT-LNN only has hooks)
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(133)
mt_runs = R["headline_T37"]["MT-LNN"]
phi1_mu, phi1_sd = ms([r["phi_clean"] for r in mt_runs])
phi10_mu, phi10_sd = ms([r["phi_full"] for r in mt_runs])

ax3.errorbar([1, 10], [phi1_mu, phi10_mu], yerr=[phi1_sd, phi10_sd],
             marker="o", capsize=3, color=MODEL_COLORS["MT-LNN"],
             label="MT-LNN (hooks active)")
# Baselines have no anesthesia hooks: delta is exactly 0 by construction.
delta_ref = phi1_mu
ax3.plot([1, 10], [delta_ref, delta_ref], ls="--", color="#A8A8A8",
         label="no-hook reference (flat)")
ax3.set_xlabel("Anesthesia level $\\kappa$")
ax3.set_ylabel("$\\hat{\\Phi}$")
ax3.set_xticks([1, 10])
ax3.set_title("AVP response (toy scale)")
ax3.legend(frameon=False, fontsize=6)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_experiments.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_experiments.png', bbox_inches='tight', dpi=300)
print("Saved fig_experiments.pdf and fig_experiments.png (from multi_seed_results.json)")
