"""Draw the decomposition figure (Fig. 6 of the paper) from results/multitown_ablation; output figures/out/.
causal structure (CSN - Flat Text), ICLR-draft style (no boxes; plain labels). Data = multitown_ablation."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, seaborn as sns

ROOT = Path(__file__).resolve().parents[1] / "results" / "multitown_ablation"
OUT = Path(__file__).resolve().parents[1] / "figures" / "out"; OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="ticks", context="paper")
SERIF = ["Nimbus Roman", "Liberation Serif", "STIX Two Text", "STIXGeneral", "DejaVu Serif"]
plt.rcParams.update({"font.family": SERIF, "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
                     "axes.edgecolor": "0.3", "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.unicode_minus": False})
P = sns.color_palette("deep"); BLUE, ORANGE, RED = P[0], P[1], P[3]

def mean_ds(prefix):
    return float(np.mean([np.mean([r["scores"]["score_composed"] for r in json.load(open(ROOT / f"{prefix}_rep{k}.json"))["_checkpoint"]["records"]]) for k in range(1, 6)]))
panels = []
for title, b, f, c in [("Original LMDrive", "original_lmdrive", "original_flat_text", "original_csn"), ("PL-DPO-NLL", "baseline", "flat_text", "csn_only")]:
    base = mean_ds(b); info = mean_ds(f) - base; struct = mean_ds(c) - base - info; panels.append((title, base, info, struct))

fig = plt.figure(figsize=(5.5, 2.2), dpi=300)
gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.085, right=0.99, top=0.90, bottom=0.30)
for k, (title, base, info, struct) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k]); total = info + struct; W = 0.62
    ax.bar(0, base, width=W, color=BLUE, alpha=0.75, edgecolor="0.3", lw=0.5)
    ax.bar(1, info, bottom=base, width=W, color=ORANGE, edgecolor="0.3", lw=0.5)
    ax.bar(2, struct, bottom=base + info, width=W, color=RED, edgecolor="0.3", lw=0.5)
    ax.bar(3, base, width=W, color=BLUE, alpha=0.30, lw=0); ax.bar(3, info, bottom=base, width=W, color=ORANGE, alpha=0.45, lw=0)
    ax.bar(3, struct, bottom=base + info, width=W, color=RED, alpha=0.45, lw=0); ax.bar(3, base + total, width=W, color="none", edgecolor="0.3", lw=0.6)
    for x0, y in [(0, base), (1, base + info)]:
        ax.plot([x0 + W / 2, x0 + 1 - W / 2], [y, y], color="0.6", lw=0.6, ls="--", zorder=1)
    ax.text(0, base / 2, f"{base:.1f}", ha="center", va="center", fontsize=7, color="white")
    ax.text(1, base + info / 2, f"+{info:.1f}", ha="center", va="center", fontsize=7, color="white")
    if struct > 2.2: ax.text(2, base + info + struct / 2, f"+{struct:.1f}", ha="center", va="center", fontsize=7, color="white")
    else: ax.text(2, base + info - 0.9, f"+{struct:.1f}", ha="center", va="top", fontsize=7, color=RED)
    ax.text(1, base + info + 1.2, f"{info / total * 100:.0f}%", ha="center", va="bottom", fontsize=6.8, color=ORANGE)
    ax.text(2, base + info + max(struct, 0) + 1.2, f"{struct / total * 100:.0f}%", ha="center", va="bottom", fontsize=6.8, color=RED)
    ax.text(3, base + total + 1.2, f"{base + total:.1f}", ha="center", va="bottom", fontsize=7, color="0.2")
    ax.text(3, base + total + 5.6, rf"$\Delta$DS $= +{total:.1f}$", ha="center", va="bottom", fontsize=6.8, color="0.3")
    ax.set_xticks(range(4)); ax.set_xticklabels(["base", "+ info\ncontent", "+ causal\nstructure", "CSN"])
    ax.set_ylim(0, 56); ax.set_yticks([0, 10, 20, 30, 40, 50]); ax.set_xlabel(title, labelpad=2)
    ax.set_title(f"({'ab'[k]})", loc="left", fontweight="bold", pad=3)
    if k == 0: ax.set_ylabel("driving score")
    else: ax.set_yticklabels([])
    sns.despine(ax=ax)
fig.legend(handles=[mpatches.Patch(color=BLUE, alpha=0.75, label="baseline DS"), mpatches.Patch(color=ORANGE, label=r"$\mathrm{Utility}_{\mathrm{info}}$ (information content)"),
                    mpatches.Patch(color=RED, label=r"$\mathrm{Utility}_{\mathrm{struct}}$ (causal structure)")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.005))
fig.savefig(OUT / "ig_decomposition.pdf"); fig.savefig(OUT / "ig_decomposition.png", dpi=300)
print("saved", [(t, round(b, 2), round(i, 2), round(s, 2)) for t, b, i, s in panels])
