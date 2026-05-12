import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# Nature-style configuration
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42
})

# NMI Pastel Palette + custom
PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "bg_lilac": "#E0E0F0",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "magenta":"#EA84DD",
    "green_1": "#DDF3DE",
    "neutral_mid":   "#767676",
    "neutral_light": "#CFCECE",
}

fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

def add_block(x, y, w, h, text, color, text_color='black', fontsize=8, alpha=1.0):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=2", 
                                  linewidth=1, edgecolor=PALETTE['neutral_mid'], facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')

def add_arrow(p1, p2, color=PALETTE['baseline_dark'], style='-|>', lw=1.5):
    arrow = patches.FancyArrowPatch(posA=p1, posB=p2, arrowstyle=style, color=color, mutation_scale=10, linewidth=lw)
    ax.add_patch(arrow)

# 1. Input Layer
add_block(5, 5, 90, 6, "Input Sequence ($X$)", PALETTE['neutral_light'])

# 2. MT-DL Layer (Microtubule Dynamic Layer)
add_block(5, 18, 90, 14, "", PALETTE['bg_lilac'], alpha=0.5)
ax.text(10, 29, "Microtubule Dynamic Layer (MT-DL)", fontweight='bold', color=PALETTE['baseline_dark'])

# Inside MT-DL: 13 Protofilaments
for i in range(13):
    add_block(8 + i*6.5, 20, 5, 5, f"P{i+1}", PALETTE['green_1'], fontsize=6)

ax.text(50, 16, "13 Parallel CfLTC Channels with Lateral Coupling", ha='center', fontsize=7, color=PALETTE['neutral_mid'])

# 3. Microtubule Attention & GWTB
add_block(5, 38, 40, 8, "Microtubule Attention\n(GTP-decay gate)", PALETTE['teal'], text_color='white')
add_block(55, 38, 40, 8, "Global Workspace Bottleneck\n(Compress to $d_{gw}$ & Broadcast)", PALETTE['gold'])

# 4. Global Coherence / Output
add_block(5, 52, 90, 6, "Global Coherence Layer (Orch-OR) & Output Projection", PALETTE['magenta'], text_color='white')

# Arrows
add_arrow((50, 11), (50, 18))  # Input to MT-DL
add_arrow((25, 32), (25, 38))  # MT-DL to Attention
add_arrow((75, 32), (75, 38))  # MT-DL to GWTB
add_arrow((45, 42), (55, 42), style='<|-|>', color=PALETTE['neutral_mid']) # Attn <-> GWTB
add_arrow((25, 46), (25, 52))  # Attn to Coherence
add_arrow((75, 46), (75, 52))  # GWTB to Coherence

plt.tight_layout()
plt.savefig('fig_architecture.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_architecture.png', bbox_inches='tight', dpi=300)
print("Saved fig_architecture.pdf and .png")
