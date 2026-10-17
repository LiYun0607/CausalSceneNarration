#!/usr/bin/env python3
"""Recompute the numbers reported in the paper from the files in results/.

Prints each table next to the value printed in the paper. Needs only numpy and
the Python standard library:

    python analysis/reproduce_paper_numbers.py
    python analysis/reproduce_paper_numbers.py --results runs   # your own runs

For your own runs, first build the token and intervention summaries with
analysis/summarize_logs.py. Sections whose result files are missing are skipped.

The bootstrap procedures, resample counts and random seeds are the ones used to
produce the paper, so the intervals match to the printed precision (the last
digit of a few half-widths can differ by one because of rounding).
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 10000
SEED = 42


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_rep(path):
    """Return per-route scores of one repetition (16 routes) or None if missing or incomplete."""
    if not Path(path).exists():
        return None
    d = json.load(open(path))
    ck = d["_checkpoint"]
    prog = ck["progress"]
    recs = ck["records"]
    if not prog or prog[0] != prog[1] or len(recs) != 16:
        return None
    return {
        r["route_id"]: {
            "ds": r["scores"]["score_composed"],
            "rc": r["scores"]["score_route"],
            "is": r["scores"]["score_penalty"],
        }
        for r in recs
    }


def load_config(folder, prefix, reps=range(1, 6)):
    out = []
    for k in reps:
        rep = load_rep(RES / folder / f"{prefix}_rep{k}.json")
        if rep is not None:
            out.append(rep)
    return out


def rep_means(reps, key="ds"):
    return [statistics.mean(v[key] for v in rep.values()) for rep in reps]


# --------------------------------------------------------------------------
# Statistics (identical to the scripts that produced the paper)
# --------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    """Mean and percentile 95% CI of the mean over repetitions."""
    v = np.array(values)
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    return float(v.mean()), float(lo), float(hi)


def half_width(values):
    _, lo, hi = bootstrap_ci(values)
    return (hi - lo) / 2


def per_route_mean(reps):
    routes = sorted(reps[0])
    return {rt: statistics.mean(r[rt]["ds"] for r in reps) for rt in routes}


def per_route_paired(reps_a, reps_b, threshold=0.5, seed=SEED):
    a, b = per_route_mean(reps_a), per_route_mean(reps_b)
    common = sorted(set(a) & set(b))
    diffs = [a[rt] - b[rt] for rt in common]
    wtl = (sum(d > threshold for d in diffs),
           sum(abs(d) <= threshold for d in diffs),
           sum(d < -threshold for d in diffs))
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
    p_pos = sum(m > 0 for m in boots) / N_BOOT
    return wtl, statistics.mean(diffs), (lo, hi), p_pos


def struct_fraction_bootstrap(b, f, c, seed=SEED):
    """Paired rep-level bootstrap of the structure fraction (same index for b, f, c)."""
    rng = random.Random(seed + 2)
    n = min(len(b), len(f), len(c))
    pcts = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        bm = sum(b[i] for i in idx) / n
        fm = sum(f[i] for i in idx) / n
        cm = sum(c[i] for i in idx) / n
        total = cm - bm
        if abs(total) > 0.01:
            pcts.append((cm - fm) / total * 100)
    pcts.sort()
    lo, hi = pcts[int(0.025 * len(pcts))], pcts[int(0.975 * len(pcts))]
    p_pos = sum(p > 0 for p in pcts) / len(pcts)
    return (lo, hi), p_pos


def asymmetry_bootstrap(orig, pl, seed=SEED):
    """Bootstrap of struct_frac(Original) - struct_frac(PL-DPO-NLL)."""
    rng = random.Random(seed + 3)

    def one(b, f, c):
        n = min(len(b), len(f), len(c))
        idx = [rng.randrange(n) for _ in range(n)]
        bm = sum(b[i] for i in idx) / n
        fm = sum(f[i] for i in idx) / n
        cm = sum(c[i] for i in idx) / n
        total = cm - bm
        if abs(total) < 0.01:
            return None
        return (cm - fm) / total * 100

    diffs = []
    for _ in range(N_BOOT):
        x, y = one(*orig), one(*pl)
        if x is not None and y is not None:
            diffs.append(x - y)
    diffs.sort()
    n = len(diffs)
    return (diffs[int(0.025 * n)], diffs[int(0.975 * n)]), sum(d > 0 for d in diffs) / n


def normal_ci(values):
    m, s, n = statistics.mean(values), statistics.stdev(values), len(values)
    hw = 1.96 * s / math.sqrt(n)
    return m, s, (m - hw, m + hw)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
MULTITOWN = [
    # label, file prefix, paper DS, RC, IS
    ("LMDrive (original)",           "original_lmdrive",         "32.54+-3.00", "48.3+-2.6", "0.729+-0.034"),
    ("  + CSN",                      "original_csn",             "42.67+-2.74", "56.5+-1.7", "0.787+-0.028"),
    ("  + Flat Text",                "original_flat_text",       "38.71+-1.44", "48.7+-1.9", "0.828+-0.025"),
    ("  + Semantic Safety",          "original_semantic_safety", "34.10+-2.25", "45.5+-2.1", "0.785+-0.038"),
    ("  + PL-DPO-NLL",               "baseline",                 "32.49+-3.34", "44.9+-2.7", "0.754+-0.021"),
    ("    + TTC Safety",             "ttc_safety",               "22.02+-4.07", "33.7+-3.4", "0.658+-0.037"),
    ("    + Semantic Safety",        "semantic_safety",          "33.17+-1.42", "44.5+-2.5", "0.787+-0.022"),
    ("    + CSN",                    "csn_only",                 "40.45+-3.79", "51.9+-4.3", "0.789+-0.026"),
    ("    + CSN+Safety",             "csn_semantic_safety",      "35.74+-1.37", "48.6+-2.1", "0.754+-0.020"),
    ("    + Flat Text",              "flat_text",                "39.38+-2.66", "49.0+-1.3", "0.823+-0.047"),
]


def section(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def table_multitown():
    section("Table: multi-town ablation (16 routes, 8 towns, N=5); mean +- 95% bootstrap CI half-width")
    print(f"{'Configuration':<24}{'DS':>14}{'(paper)':>14}{'RC %':>12}{'(paper)':>12}"
          f"{'IS':>16}{'(paper)':>15}{'dDS_orig':>10}{'dDS_base':>10}")
    data = {}
    for label, prefix, *_ in MULTITOWN:
        data[prefix] = load_config("multitown_ablation", prefix)
    orig = statistics.mean(rep_means(data["original_lmdrive"])) if data["original_lmdrive"] else None
    base = statistics.mean(rep_means(data["baseline"])) if data["baseline"] else None
    for label, prefix, p_ds, p_rc, p_is in MULTITOWN:
        reps = data[prefix]
        if not reps:
            print(f"{label:<24}  no complete repetition of {prefix} found, skipped")
            continue
        ds, rc, is_ = rep_means(reps, "ds"), rep_means(reps, "rc"), rep_means(reps, "is")
        m_ds, m_rc, m_is = statistics.mean(ds), statistics.mean(rc), statistics.mean(is_)
        if prefix == "original_lmdrive" or orig is None:
            d_orig = "---"
        else:
            d_orig = f"{(m_ds - orig) / orig * 100:+.1f}%"
        pl_rows = ("ttc_safety", "semantic_safety", "csn_only", "csn_semantic_safety", "flat_text")
        d_base = f"{(m_ds - base) / base * 100:+.1f}%" if prefix in pl_rows and base is not None else "---"
        print(f"{label:<24}{m_ds:>8.2f}+-{half_width(ds):<4.2f}{p_ds:>14}"
              f"{m_rc:>6.1f}+-{half_width(rc):<4.1f}{p_rc:>12}"
              f"{m_is:>9.3f}+-{half_width(is_):<5.3f}{p_is:>15}{d_orig:>10}{d_base:>10}"
              f"   (N={len(reps)})")
    return data


def decomposition(data):
    section("Information vs structure decomposition (Eq. 3) with paired rep-level bootstrap")
    paper = {
        "Original LMDrive": ("+6.17", "+3.96", "39.1%", "[6.6%, 71.4%]", "0.986"),
        "PL-DPO-NLL": ("+6.88", "+1.08", "13.5%", "[-244.0%, 41.9%]", "0.664"),
    }
    triples = {}
    for label, b, f, c in [
        ("Original LMDrive", "original_lmdrive", "original_flat_text", "original_csn"),
        ("PL-DPO-NLL", "baseline", "flat_text", "csn_only"),
    ]:
        bm, fm, cm = rep_means(data[b]), rep_means(data[f]), rep_means(data[c])
        triples[label] = (bm, fm, cm)
        B, F, C = statistics.mean(bm), statistics.mean(fm), statistics.mean(cm)
        total, info, struct = C - B, F - B, C - F
        (lo, hi), p = struct_fraction_bootstrap(bm, fm, cm)
        pi = paper[label]
        print(f"[{label}]  baseline {B:.2f}, +Flat Text {F:.2f}, +CSN {C:.2f}, total gain {total:+.2f}")
        print(f"  Utility_info   {info:+.2f}   (paper {pi[0]})")
        print(f"  Utility_struct {struct:+.2f}   (paper {pi[1]})")
        print(f"  structure fraction {struct / total * 100:.1f}%  (paper {pi[2]})")
        print(f"  95% CI [{lo:.1f}%, {hi:.1f}%]  (paper {pi[3]}),  P(>0) = {p:.3f}  (paper {pi[4]})")
    (lo, hi), p = asymmetry_bootstrap(triples["Original LMDrive"], triples["PL-DPO-NLL"])
    print(f"Asymmetry (Original - PL-DPO-NLL): 95% CI [{lo:+.1f}%, {hi:+.1f}%]  (paper [-23.8%, 306.4%]),"
          f"  P(>0) = {p:.3f}")

    section("Per-route paired comparison, CSN vs Flat Text (margin 0.5 DS)")
    for label, c, f, pw, pc in [
        ("Original LMDrive", "original_csn", "original_flat_text", "5/4/7", "[-3.86, 13.41]"),
        ("PL-DPO-NLL", "csn_only", "flat_text", "5/6/5", "[-4.44, 6.99]"),
    ]:
        wtl, mean_diff, (lo, hi), p = per_route_paired(data[c], data[f])
        print(f"[{label}]  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]} (paper {pw}),  mean diff {mean_diff:+.2f} DS,"
              f"  95% CI [{lo:+.2f}, {hi:+.2f}] (paper {pc})")


BASELINES = [
    ("DriveVLM-style compressed", "original_drivevlm", "31.02+-4.95", "47.3", "0.672", "-1.5", "-4.7%"),
    ("DriveVLM-style verbatim", "original_drivevlm_faithful", "8.43+-0.79", "18.4", "0.472", "-24.1", "-74.1%"),
    ("DriveLM-style compressed", "original_drivelm", "28.82+-3.30", "47.8", "0.632", "-3.7", "-11.4%"),
    ("DriveLM-style verbatim", "original_drivelm_faithful", "5.63+-0.42", "13.7", "0.504", "-26.9", "-82.7%"),
]


def py_bootstrap_hw(values, rng, n_boot=N_BOOT):
    """Half-width of a percentile bootstrap CI drawn from a shared random.Random."""
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n_boot))
    return (means[int(0.975 * n_boot)] - means[int(0.025 * n_boot)]) / 2


def table_baselines(data):
    section("Table: DriveVLM-style and DriveLM-style text on Original LMDrive (N=5)")
    orig = statistics.mean(rep_means(data["original_lmdrive"]))
    # The paper's half-widths for this table come from one random.Random(42)
    # stream consumed in this order.
    rng = random.Random(SEED)
    hw = {}
    for prefix in ["original_drivevlm", "original_drivelm", "original_drivevlm_faithful",
                   "original_drivelm_faithful"]:
        hw[prefix] = py_bootstrap_hw(rep_means(load_config("baseline_comparison", prefix)), rng)
    print(f"{'Configuration':<28}{'DS':>14}{'(paper)':>14}{'RC %':>8}{'(paper)':>9}{'IS':>8}{'(paper)':>9}"
          f"{'dDS':>8}{'(paper)':>9}{'rel.':>9}{'(taxonomy table)':>18}")
    for label, prefix, p_ds, p_rc, p_is, p_d, p_rel in BASELINES:
        reps = load_config("baseline_comparison", prefix)
        ds = rep_means(reps)
        m = statistics.mean(ds)
        rc = statistics.mean(rep_means(reps, "rc"))
        is_ = statistics.mean(rep_means(reps, "is"))
        print(f"{label:<28}{m:>8.2f}+-{hw[prefix]:<4.2f}{p_ds:>14}{rc:>8.1f}{p_rc:>9}{is_:>8.3f}{p_is:>9}"
              f"{m - orig:>+8.1f}{p_d:>9}{(m - orig) / orig * 100:>+8.1f}%{p_rel:>18}   (N={len(reps)})")


def token_stats():
    section("Token estimates per frame (results/token_stats/token_stats_summary.json)")
    if not (RES / "token_stats" / "token_stats_summary.json").exists():
        print("not found, skipped")
        return
    tok = json.load(open(RES / "token_stats" / "token_stats_summary.json"))["per_config"]
    print(f"{'Configuration':<32}{'first string':>14}{'second string':>15}{'both':>8}{'frames':>9}")
    for k, v in tok.items():
        print(f"{k:<32}{v['instruction_mean']:>14.1f}{v['notice_mean']:>15.1f}{v['total_mean']:>8.1f}{v['frames']:>9}")
    print("The agent sends only the first string to LMDrive. No token log was recorded for CSN runs.")
    print("The counts are the renderer's estimate (len(text) + 2) // 4, not LLaMA tokens.")


NOISE = [
    ("Clean (privileged)", None, "42.3+-3.5", "[39.5, 45.0]"),
    ("Mild", "noise_mild", "45.2+-2.9", "[42.6, 47.8]"),
    ("Moderate", "noise_moderate", "45.4+-2.7", "[43.1, 47.6]"),
    ("Severe", "noise_severe", "46.0+-4.1", "[42.5, 49.6]"),
    ("Extreme", "noise_extreme", "45.1+-2.7", "[42.7, 47.4]"),
    ("Flat Text + Severe", "flat_noise_severe", "37.7+-2.4", "[35.6, 39.7]"),
]


def table_noise(data):
    section("Table: perception noise on Original LMDrive + CSN; mean +- SD and 95% CI")
    print("Clean and Flat Text rows: sample SD and normal-approximation CI.")
    print("Mild to Extreme rows: population SD (ddof=0) and percentile bootstrap CI, as printed in the paper.")
    for label, prefix, p_ms, p_ci in NOISE:
        if prefix is None:
            # clean: the five Original + CSN repetitions plus one extra clean repetition
            vals = rep_means(data["original_csn"]) + rep_means([load_rep(RES / "noise_ablation" / "clean_rep6.json")])
        else:
            vals = rep_means(load_config("noise_ablation", prefix))
        if prefix is None or prefix == "flat_noise_severe":
            m, s, (lo, hi) = normal_ci(vals)
        else:
            m, lo, hi = bootstrap_ci(vals)
            s = float(np.std(vals))
        print(f"{label:<22} N={len(vals)}  {m:5.1f}+-{s:<4.1f} [{lo:5.1f}, {hi:5.1f}]   (paper {p_ms} {p_ci})")

    flat_clean = rep_means(data["original_flat_text"])
    flat_noise = rep_means(load_config("noise_ablation", "flat_noise_severe"))
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOT):
        a = statistics.mean(rng.choice(flat_noise) for _ in flat_noise)
        b = statistics.mean(rng.choice(flat_clean) for _ in flat_clean)
        diffs.append(a - b)
    diffs.sort()
    print(f"Flat Text clean {statistics.mean(flat_clean):.2f}+-{statistics.stdev(flat_clean):.2f} (SD), "
          f"Severe {statistics.mean(flat_noise):.2f}+-{statistics.stdev(flat_noise):.2f}; "
          f"difference {statistics.mean(flat_noise) - statistics.mean(flat_clean):+.2f} DS")
    print(f"  bootstrap 95% CI [{diffs[250]:+.2f}, {diffs[9750]:+.2f}] (paper [-3.35, +1.38]), "
          f"P(diff>0) = {sum(d > 0 for d in diffs) / N_BOOT:.2f} (paper 0.19)")
    print("  (the two groups are resampled independently)")


def interventions():
    section("Runtime monitor interventions (results/intervention_logging/intervention_counts.json)")
    if not (RES / "intervention_logging" / "intervention_counts.json").exists():
        print("not found, skipped")
        return
    runs = json.load(open(RES / "intervention_logging" / "intervention_counts.json"))["runs"]
    n_routes = n_int = n_dc = n_stuck = 0
    for name in ["baseline_safety_rep1", "baseline_safety_rep2", "baseline_safety_rep3",
                 "csn_safety_rep1", "csn_safety_rep2", "csn_safety_rep3"]:
        for r in runs.get(name, []):
            if "total_checks" in r:
                n_routes += 1
                n_int += r["total_interventions"]
                n_dc += r["direction_conflict_count"]
                n_stuck += r["stuck_detected_count"]
    print(f"Repetitions 1-3 of baseline+Safety and CSN+Safety: {n_routes} route checks (paper 96), "
          f"{n_int} interventions, {n_dc} direction conflicts, {n_stuck} stuck detections")
    extra = sum(r.get("total_interventions", 0) for k, v in runs.items() for r in v)
    done = sum(1 for v in runs.values() for r in v if "total_checks" in r)
    print(f"All logged runs including incomplete repetitions: {done} routes, {extra} interventions")


def main():
    global RES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(RES),
                    help="folder with multitown_ablation/, baseline_comparison/, noise_ablation/ (default: results/)")
    RES = Path(ap.parse_args().results).resolve()
    data = table_multitown()
    for fn, args in [(decomposition, (data,)), (table_baselines, (data,)), (token_stats, ()),
                     (table_noise, (data,)), (interventions, ())]:
        try:
            fn(*args)
        except (AttributeError, KeyError, IndexError, ZeroDivisionError, TypeError,
                statistics.StatisticsError) as e:
            print(f"skipped: result files for this section are missing or incomplete ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
