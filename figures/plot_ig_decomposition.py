#!/usr/bin/env python3
"""Draw the decomposition figure (figures/ig_decomposition.pdf in the paper).

Splits the DS gain of CSN into information content (Flat Text - baseline) and
causal structure (CSN - Flat Text) for both weight configurations, from
results/multitown_ablation/. Output: figures/out/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects
import json
from pathlib import Path

import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "multitown_ablation"
OUT_DIR = ROOT / "figures" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mean_ds(prefix):
    """Mean DS over the five repetitions of one configuration."""
    reps = []
    for k in range(1, 6):
        recs = json.load(open(RESULT_DIR / f"{prefix}_rep{k}.json"))["_checkpoint"]["records"]
        reps.append(np.mean([r["scores"]["score_composed"] for r in recs]))
    return float(np.mean(reps))


# --- Data ---
orig_base = mean_ds("original_lmdrive")
orig_info = mean_ds("original_flat_text") - orig_base
orig_struct = mean_ds("original_csn") - orig_base - orig_info
orig_total = orig_info + orig_struct

pl_base = mean_ds("baseline")
pl_info = mean_ds("flat_text") - pl_base
pl_struct = mean_ds("csn_only") - pl_base - pl_info
pl_total = pl_info + pl_struct

# --- Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})
sns.set_theme(
    style="ticks",
    font_scale=1.25,
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.6,
    },
)

color_base = "#7B9DC2"   # soft steel blue
color_info = "#E2A542"   # warm amber
color_struct = "#D16B6B" # soft coral

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False,
                         gridspec_kw={"wspace": 0.22})

for ax, (base, info, struct, total, title) in zip(axes, [
    (orig_base, orig_info, orig_struct, orig_total, "(a) Original LMDrive"),
    (pl_base, pl_info, pl_struct, pl_total, "(b) PL-DPO-NLL"),
]):
    labels = ["Base", "$+\\,$Info\nContent", "$+\\,$Causal\nStructure", "CSN"]
    bottoms = [0, base, base + info, 0]
    heights = [base, info, struct, base + total]

    bar_w = 0.6

    # Draw bars
    for i, (b, h, c) in enumerate(zip(bottoms, heights,
            [color_base, color_info, color_struct, color_base])):
        if i == 3:
            # Total: stacked composite
            ax.bar(i, base, bottom=0, color=color_base, alpha=0.25,
                   width=bar_w, edgecolor="none")
            ax.bar(i, info, bottom=base, color=color_info, alpha=0.5,
                   width=bar_w, edgecolor="none")
            ax.bar(i, struct, bottom=base + info, color=color_struct, alpha=0.5,
                   width=bar_w, edgecolor="none")
            ax.bar(i, h, bottom=0, color="none", width=bar_w,
                   edgecolor="#666666", linewidth=1.0, linestyle="-")
        else:
            alpha = 0.55 if i == 0 else 0.85
            ax.bar(i, h, bottom=b, color=c, alpha=alpha,
                   width=bar_w, edgecolor="#666666", linewidth=0.7)

    # Connector arrows
    for i in range(2):
        top = bottoms[i] + heights[i] if i > 0 else heights[i]
        ax.annotate("", xy=(i + 0.72, top), xytext=(i + 0.28 + 1, top),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                   linewidth=0.8, linestyle="--"))

    # --- Labels ---

    # Base
    ax.text(0, base / 2, f"{base:.1f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#FFFFFF",
            path_effects=[matplotlib.patheffects.withStroke(
                linewidth=2.5, foreground="#555555")])

    # Info Content
    ax.text(1, base + info / 2, f"+{info:.1f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#FFFFFF",
            path_effects=[matplotlib.patheffects.withStroke(
                linewidth=2.5, foreground="#8B6914")])

    # Percentage badges (above each incremental bar)
    info_pct = info / total * 100
    struct_pct = struct / total * 100

    ax.text(1, base + info + 1.5,
            f"{info_pct:.0f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color_info,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                     edgecolor=color_info, alpha=0.95, linewidth=1.0))

    # Causal Structure — label inside if tall enough, below bar otherwise
    if struct > 2:
        ax.text(2, base + info + struct / 2, f"+{struct:.1f}",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="#FFFFFF",
                path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=2.5, foreground="#8B3333")])
    else:
        ax.text(2, base + info - 0.8, f"+{struct:.1f}",
                ha="center", va="top",
                fontsize=12, fontweight="bold", color=color_struct)

    ax.text(2, base + info + struct + 1.5,
            f"{struct_pct:.0f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color_struct,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                     edgecolor=color_struct, alpha=0.95, linewidth=1.0))

    # Total label
    csn_val = base + total
    ax.text(3, csn_val + 1.5, f"{csn_val:.1f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#333333")

    # Delta DS badge
    ax.text(3, csn_val + 5.5,
            f"$\\Delta$DS$\\,=\\,+${total:.1f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5F5F5",
                     edgecolor="#999999", alpha=0.95, linewidth=0.8))

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#333333")
    ax.set_ylabel("Driving Score" if ax == axes[0] else "",
                  fontsize=13, color="#333333")
    ax.set_ylim(0, 54)
    ax.yaxis.grid(True, linewidth=0.3, color="#E5E5E5")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11)

# Legend
legend_patches = [
    mpatches.Patch(color=color_base, alpha=0.55, label="Baseline DS"),
    mpatches.Patch(color=color_info, alpha=0.85,
                   label="$\\mathrm{Utility}_{\\mathrm{info}}$"
                         " (information content)"),
    mpatches.Patch(color=color_struct, alpha=0.85,
                   label="$\\mathrm{Utility}_{\\mathrm{struct}}$"
                         " (causal structure)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=11, frameon=True, fancybox=True,
           edgecolor="#CCCCCC", facecolor="white",
           bbox_to_anchor=(0.5, -0.01))

sns.despine()
plt.tight_layout()
fig.subplots_adjust(bottom=0.16)

fig.savefig(OUT_DIR / "ig_decomposition.pdf", format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "ig_decomposition.png", format="png", dpi=300, bbox_inches="tight")
print("Saved", OUT_DIR / "ig_decomposition.pdf")
plt.close()
