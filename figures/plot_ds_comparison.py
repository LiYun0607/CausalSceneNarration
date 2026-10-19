"""Draw the Driving Score comparison figure (Fig. 5 of the paper) from results/multitown_ablation; output figures/out/.
(seaborn ticks, serif, mathtext, bold (a)(b)(c) at the left, no text boxes). Data = the paper's
results (release/CausalSceneNarration/results/multitown_ablation). Bootstrap CIs: 5,000 draws, seed 0."""
import glob, json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / "results" / "multitown_ablation"
OUT = Path(__file__).resolve().parents[1] / "figures" / "out"; OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="ticks", context="paper")
SERIF = ["Nimbus Roman", "Liberation Serif", "STIX Two Text", "STIXGeneral", "DejaVu Serif"]
plt.rcParams.update({"font.family": SERIF, "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
                     "axes.edgecolor": "0.3", "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.unicode_minus": False})
P = sns.color_palette("deep")
BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW = P[0], P[1], P[2], P[3], P[4], P[8]
COL = {"LMDrive": BLUE, "PL-DPO-NLL": BLUE, "+CSN": ORANGE, "+Flat Text": YELLOW, "+Safety": GREEN, "+TTC": RED, "+CSN+Safety": PURPLE}

def load(pattern):
    vals = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        d = json.load(open(f)); p = d["_checkpoint"]["progress"]
        if not p or p[0] != p[1]: continue
        vals.append(np.mean([r["scores"]["score_composed"] for r in d["_checkpoint"]["records"]]))
    return np.array(vals)

GROUPS = {"Original LMDrive": [("LMDrive", "original_lmdrive_rep*.json"), ("+CSN", "original_csn_rep*.json"),
                               ("+Flat Text", "original_flat_text_rep*.json"), ("+Safety", "original_semantic_safety_rep*.json")],
          "PL-DPO-NLL": [("PL-DPO-NLL", "baseline_*.json"), ("+TTC", "ttc_safety_*.json"), ("+Safety", "semantic_safety_*.json"),
                         ("+CSN", "csn_only_*.json"), ("+CSN+Safety", "csn_semantic_safety_*.json"), ("+Flat Text", "flat_text_*.json")]}
data = {g: [(lab, load(pat)) for lab, pat in cfg] for g, cfg in GROUPS.items()}
rng = np.random.default_rng(0)
def ci(v, n=5000):
    m = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)]); return np.percentile(m, [2.5, 97.5])
base = data["Original LMDrive"][0][1].mean()

fig = plt.figure(figsize=(6.5, 2.4), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.75], wspace=0.42, left=0.07, right=0.995, top=0.90, bottom=0.29)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, (gname, panel) in zip(axes[:2], [("Original LMDrive", "(a)"), ("PL-DPO-NLL", "(b)")]):
    rows = data[gname]
    for i, (lab, v) in enumerate(rows):
        lo, hi = ci(v); m = v.mean()
        ax.bar(i, m, width=0.68, color=COL[lab], edgecolor="0.3", linewidth=0.5, zorder=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="0.25", elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
        ax.text(i, hi + 1.2, f"{m:.1f}", ha="center", va="bottom", fontsize=6.3, color="0.2")
    ax.axhline(base, color="0.55", ls=":", lw=0.7, zorder=1)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlabel(gname, labelpad=2); ax.set_ylim(0, 56); ax.set_title(panel, loc="left", fontweight="bold", pad=3)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
axes[0].set_ylabel("driving score"); axes[1].set_yticklabels([])
sns.despine(ax=axes[0]); sns.despine(ax=axes[1])

ax = axes[2]
order = [(g, lab, v) for g in GROUPS for lab, v in data[g]]
names = [lab if lab in ("LMDrive", "PL-DPO-NLL") else ("Orig. " if g == "Original LMDrive" else "PL ") + lab for g, lab, v in order]
for idx, (g, lab, v) in enumerate(order):
    c = COL[lab]
    if len(v) >= 2 and v.std() > 0.01:
        xs = np.linspace(v.min() - 3, v.max() + 3, 200); ys = stats.gaussian_kde(v, bw_method=0.5)(xs); ys = ys / ys.max() * 0.38
        ax.fill_between(xs, idx - ys, idx, color=c, alpha=0.30, lw=0, zorder=2); ax.plot(xs, idx - ys, color=c, lw=0.6, zorder=2)
    jit = np.random.default_rng(42 + idx).uniform(0.06, 0.30, len(v))
    ax.scatter(v, idx + jit, s=9, color=c, edgecolors="0.3", linewidths=0.4, zorder=4)
    lo, hi = ci(v); ax.plot([lo, hi], [idx, idx], color="0.25", lw=1.0, solid_capstyle="round", zorder=3)
    ax.scatter([v.mean()], [idx], marker="D", s=14, color="white", edgecolors="0.2", linewidths=0.7, zorder=5)
ax.axhline(3.5, color="0.6", ls="--", lw=0.6, zorder=1); ax.axvline(base, color="0.55", ls=":", lw=0.7, zorder=0)
ax.set_yticks(range(len(order))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_ylim(len(order) - 0.4, -0.9)
ax.set_xlabel("driving score"); ax.set_xlim(15, 52); ax.set_title("(c)", loc="left", fontweight="bold", pad=3)
sns.despine(ax=ax)
fig.savefig(OUT / "ds_comparison.pdf"); fig.savefig(OUT / "ds_comparison.png", dpi=300)
print("saved", OUT / "ds_comparison.pdf", {g: [(l, round(v.mean(), 2), len(v)) for l, v in data[g]] for g in data})
