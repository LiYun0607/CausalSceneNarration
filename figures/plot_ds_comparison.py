#!/usr/bin/env python3
"""Draw the Driving Score comparison figure (figures/ds_comparison.pdf in the paper).

Bar charts for the two weight configurations and a raincloud of the individual
runs, from results/multitown_ablation/. Output: figures/out/.
"""

import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from scipy import stats
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = str(ROOT / "results" / "multitown_ablation")
OUT_DIR = ROOT / "figures" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ds(pattern):
    files = sorted(glob.glob(f"{RESULT_DIR}/{pattern}"))
    vals = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        p = d["_checkpoint"]["progress"]
        if not p or p[0] != p[1]:
            continue
        records = d["_checkpoint"]["records"]
        vals.append(np.mean([r["scores"]["score_composed"] for r in records]))
    return vals

configs = [
    ("LMDrive",    "original_lmdrive_rep*.json",          "Original LMDrive"),
    ("+CSN",       "original_csn_rep*.json",              "Original LMDrive"),
    ("+Flat Text", "original_flat_text_rep*.json",        "Original LMDrive"),
    ("+Safety",    "original_semantic_safety_rep*.json",   "Original LMDrive"),
    ("PL-DPO-NLL", "baseline_*.json",                     "PL-DPO-NLL"),
    ("+TTC",       "ttc_safety_*.json",                   "PL-DPO-NLL"),
    ("+Safety",    "semantic_safety_*.json",              "PL-DPO-NLL"),
    ("+CSN",       "csn_only_*.json",                    "PL-DPO-NLL"),
    ("+CSN+Safety","csn_semantic_safety_*.json",          "PL-DPO-NLL"),
    ("+Flat Text", "flat_text_*.json",                   "PL-DPO-NLL"),
]

rows = []
for label, pat, group in configs:
    vals = load_ds(pat)
    for v in vals:
        rows.append({"Configuration": label, "Group": group, "DS": v})
df = pd.DataFrame(rows)

# Full names for right panel
name_map = {}
for label, _, group in configs:
    key = (label, group)
    if key not in name_map:
        if group == "Original LMDrive" and label != "LMDrive":
            name_map[key] = f"Orig. {label}"
        elif group == "PL-DPO-NLL" and label != "PL-DPO-NLL":
            name_map[key] = f"PL {label}"
        else:
            name_map[key] = label
df["FullName"] = df.apply(lambda r: name_map[(r["Configuration"], r["Group"])], axis=1)

# --- Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})
sns.set_theme(
    style="ticks",
    font_scale=1.2,
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.6,
    },
)

fig, axes = plt.subplots(
    1, 3, figsize=(20, 5.5),
    gridspec_kw={"width_ratios": [4, 6, 7], "wspace": 0.15},
)
ax_bar1, ax_bar2, ax_rain = axes
ax_bar2.sharey(ax_bar1)

color_map = {
    "LMDrive":    "#6C8EBF",
    "PL-DPO-NLL": "#6C8EBF",
    "+CSN":       "#D4A03C",
    "+Flat Text": "#E8C56D",
    "+Safety":    "#7BAE7F",
    "+TTC":       "#C75B5B",
    "+CSN+Safety":"#9B7DC2",
}

# ---- Left two panels: bar charts ----
for ax, group_name in zip([ax_bar1, ax_bar2], ["Original LMDrive", "PL-DPO-NLL"]):
    group_df = df[df["Group"] == group_name]
    order = [c[0] for c in configs if c[2] == group_name]
    seen = set()
    order_unique = []
    for o in order:
        if o not in seen:
            order_unique.append(o)
            seen.add(o)
    palette = [color_map[c] for c in order_unique]

    sns.barplot(
        data=group_df, x="Configuration", y="DS", order=order_unique,
        palette=palette, ax=ax, errorbar=("ci", 95), capsize=0.12,
        err_kws={"linewidth": 1.2, "color": "#444444"},
        edgecolor="#555555", linewidth=0.5, saturation=0.95,
        width=0.72,
    )

    for i, bar in enumerate(ax.patches):
        if bar.get_height() > 0:
            sub = group_df[group_df["Configuration"] == order_unique[i]]
            m = sub["DS"].mean()
            ci_upper = np.percentile(
                [np.mean(np.random.default_rng(s).choice(sub["DS"].values, len(sub), replace=True))
                 for s in range(5000)], 97.5)
            y_pos = max(ci_upper, bar.get_height()) + 1.8
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=9,
                    fontweight="semibold", color="#222222")

    ax.set_title(group_name, fontsize=12, fontweight="bold", pad=10, color="#333333")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9, rotation=0)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.grid(True, linewidth=0.4, color="#E0E0E0")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

# ---- Right panel: Raincloud plot (half-violin + jittered strip + mean diamond) ----
rain_order = [
    "LMDrive", "Orig. +CSN", "Orig. +Flat Text", "Orig. +Safety",
    "PL-DPO-NLL", "PL +TTC", "PL +Safety", "PL +CSN", "PL +CSN+Safety", "PL +Flat Text",
]

strip_color_map = {
    "LMDrive":         "#6C8EBF",
    "Orig. +CSN":      "#D4A03C",
    "Orig. +Flat Text": "#E8C56D",
    "Orig. +Safety":   "#7BAE7F",
    "PL-DPO-NLL":      "#6C8EBF",
    "PL +TTC":         "#C75B5B",
    "PL +Safety":      "#7BAE7F",
    "PL +CSN":         "#D4A03C",
    "PL +CSN+Safety":  "#9B7DC2",
    "PL +Flat Text":   "#E8C56D",
}

# Draw half-violin (KDE) on the upper side, strip on the lower side
for idx, name in enumerate(rain_order):
    sub = df[df["FullName"] == name]["DS"].values
    color = strip_color_map[name]

    # Half-violin (KDE) — drawn above the center line
    if len(sub) >= 2 and np.std(sub) > 0.01:
        kde_x = np.linspace(sub.min() - 3, sub.max() + 3, 200)
        kde = stats.gaussian_kde(sub, bw_method=0.5)
        kde_y = kde(kde_x)
        # Normalize height
        kde_y = kde_y / kde_y.max() * 0.35
        # Draw as filled polygon above the line (idx - kde_y)
        ax_rain.fill_between(kde_x, idx - kde_y, idx,
                             color=color, alpha=0.35, zorder=2)
        ax_rain.plot(kde_x, idx - kde_y,
                     color=color, linewidth=0.8, alpha=0.7, zorder=2)

    # Jittered strip below center line
    rng = np.random.default_rng(42 + idx)
    jitter = rng.uniform(0.05, 0.25, size=len(sub))
    ax_rain.scatter(sub, idx + jitter,
                    color=color, s=40, alpha=0.85,
                    edgecolors="#444444", linewidths=0.5, zorder=4)

    # Mean marker (diamond)
    m = np.mean(sub)
    ax_rain.scatter([m], [idx], marker="D", color="white",
                    edgecolors="#333333", s=50, linewidths=1.2, zorder=5)

    # Thin CI whisker
    ci_lo, ci_hi = np.percentile(
        [np.mean(rng.choice(sub, len(sub), replace=True)) for _ in range(5000)],
        [2.5, 97.5])
    ax_rain.plot([ci_lo, ci_hi], [idx, idx],
                 color="#555555", linewidth=1.5, zorder=3, solid_capstyle="round")

# Separator between groups
ax_rain.axhline(y=3.5, color="#BBBBBB", linestyle="--", linewidth=0.7, zorder=1)

# Reference line
base_orig = df[(df["Configuration"] == "LMDrive") & (df["Group"] == "Original LMDrive")]["DS"].mean()
ax_rain.axvline(x=base_orig, color="#999999", linestyle=":", linewidth=0.9, zorder=0)

# Y-axis labels on the RIGHT side
ax_rain.set_yticks(range(len(rain_order)))
ax_rain.set_yticklabels(rain_order, fontsize=9)
ax_rain.yaxis.tick_right()
ax_rain.yaxis.set_label_position("right")
ax_rain.invert_yaxis()

ax_rain.set_title("Distribution of Individual Runs", fontsize=12, fontweight="bold",
                   pad=10, color="#333333")
ax_rain.set_xlabel("Driving Score", fontsize=11, color="#333333")
ax_rain.set_ylabel("")
ax_rain.tick_params(axis="x", labelsize=10)
ax_rain.xaxis.grid(True, linewidth=0.4, color="#E0E0E0")
ax_rain.yaxis.grid(False)
ax_rain.set_axisbelow(True)

# Reference line on bar panels
for ax in [ax_bar1, ax_bar2]:
    ax.axhline(y=base_orig, color="#999999", linestyle=":", linewidth=0.9, zorder=0)

ax_bar1.set_ylabel("Driving Score", fontsize=12, color="#333333")
ax_bar2.set_ylabel("")
ax_bar1.set_ylim(0, 54)

sns.despine(right=False)
# Keep right spine for rain panel only
ax_bar1.spines["right"].set_visible(False)
ax_bar2.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "ds_comparison.pdf", format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "ds_comparison.png", format="png", dpi=300, bbox_inches="tight")
print("Saved", OUT_DIR / "ds_comparison.pdf")
plt.close()
