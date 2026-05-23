import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Use aesthetic fonts if available, fallback to sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aeonik', 'Helvetica', 'Arial']

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
cloud_color = "#e0f2fe"
cloud_edge = "#0284c7"
cloud_box = "#bae6fd"
cloud_text = "#0369a1"

edge_color = "#fdf2f8"
edge_edge = "#be185d"
edge_box = "#fce7f3"
edge_text = "#831843"
m1_box = "#ffe4e6"
m1_edge = "#e11d48"

# Draw Cloud
cloud = patches.FancyBboxPatch((7.0, 3.2), 3.5, 2.5, boxstyle="round,pad=0.2", ec=cloud_edge, fc=cloud_color, alpha=0.9, lw=2)
ax.add_patch(cloud)
ax.text(8.75, 5.5, "Cloud (Knowledge Oracle)", ha="center", va="center", fontsize=14, fontweight='bold', color=cloud_text)

db = patches.Rectangle((7.4, 4.3), 2.7, 0.7, ec=cloud_edge, fc=cloud_box, lw=1.5, rx=0.1)
ax.add_patch(db)
ax.text(8.75, 4.65, "Vector DB (Facts/RAG)", ha="center", va="center", fontsize=12, color=cloud_text)

gemini = patches.Rectangle((7.4, 3.4), 2.7, 0.7, ec=cloud_edge, fc=cloud_box, lw=1.5, rx=0.1)
ax.add_patch(gemini)
ax.text(8.75, 3.75, "Gemini 3.1 / GPT-4o\n(LLM API)", ha="center", va="center", fontsize=11, color=cloud_text)


# Draw Edge
edge = patches.FancyBboxPatch((0.5, 0.5), 5.0, 5.0, boxstyle="round,pad=0.2", ec=edge_edge, fc=edge_color, alpha=0.9, lw=2)
ax.add_patch(edge)
ax.text(3.0, 5.2, "Edge / Local Device (Air-gapped)", ha="center", va="center", fontsize=14, fontweight='bold', color=edge_text)

m1_panel = patches.FancyBboxPatch((0.8, 0.8), 4.4, 4.0, boxstyle="round,pad=0.1", ec=m1_edge, fc=edge_box, lw=1.5)
ax.add_patch(m1_panel)
ax.text(3.0, 4.5, "M1 (MT-LNN) Engine", ha="center", va="center", fontsize=12, fontweight='bold', color=edge_text)

capsule = patches.FancyBboxPatch((1.1, 3.5), 3.8, 0.6, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(capsule)
ax.text(3.0, 3.8, "Personal State Capsule (4.1KB $h_{prev}$)\nLifelong memory logic", ha="center", va="center", fontsize=10, color=edge_text)

monitor = patches.FancyBboxPatch((1.1, 2.7), 3.8, 0.5, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(monitor)
ax.text(3.0, 2.95, "Predictive Error Monitor (Active UI/UX)", ha="center", va="center", fontsize=10, color=edge_text)

loop = patches.FancyBboxPatch((1.1, 1.2), 1.7, 1.2, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(loop)
ax.text(1.95, 1.8, "Latent Loop\n\n10s Silent\nDeep Deduction", ha="center", va="center", fontsize=10, color=edge_text)

router = patches.FancyBboxPatch((3.2, 1.2), 1.7, 1.2, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(router)
ax.text(4.05, 1.8, "Cloud Oracle\nRouter\n\n(Fact Queries)", ha="center", va="center", fontsize=10, color=edge_text)

# Hardware Text
ax.text(3.0, 1.0, "CPU / NPU inference only", ha="center", va="center", fontsize=9, style='italic', color=edge_text)

# Connections
ax.annotate("", xy=(7.0, 3.75), xytext=(4.95, 1.8), 
            arrowprops=dict(arrowstyle="<|-|>", color="#94a3b8", lw=3.0, ls='dashed', shrinkA=5, shrinkB=5))
ax.text(6.0, 2.6, "1. Submits concise query\n2. Consumes raw facts", ha="center", va="center", fontsize=10, rotation=37, color="#475569", fontweight='bold')

plt.savefig("fig_awareness_network.pdf", bbox_inches='tight')
plt.savefig("fig_awareness_network.png", bbox_inches='tight', dpi=300)
print("Saved fig_awareness_network.pdf")
