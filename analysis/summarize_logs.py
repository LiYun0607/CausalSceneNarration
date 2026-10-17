#!/usr/bin/env python3
"""Build the token and intervention summaries from your own closed-loop runs.

    python analysis/summarize_logs.py runs

<runs> is the out_dir given to scripts/run_paper_experiments.sh. The script
writes the two files that analysis/reproduce_paper_numbers.py reads:

  <runs>/token_stats/token_stats_summary.json
      from <runs>/baseline_comparison/token_stats/<config>_rep<k>.jsonl
      (written by leaderboard/team_code/text_token_logger.py)
  <runs>/intervention_logging/intervention_counts.json
      from <runs>/intervention_logging/frames/<config>_rep<k>/<route>/intervention_summary.json
      (written by the agent at the end of each route when SAVE_PATH is set)

Both files have the same layout as the ones in results/. Needs only the
Python standard library.
"""

import argparse
import json
import re
from pathlib import Path

INTERVENTION_FIELDS = ["total_checks", "total_interventions", "direction_conflict_count",
                       "stuck_detected_count"]


def summarize_tokens(runs):
    src = runs / "baseline_comparison" / "token_stats"
    logs = sorted(src.glob("*_rep*.jsonl")) if src.is_dir() else []
    if not logs:
        print(f"token logs: none found in {src}, skipped")
        return
    per_run, pooled = {}, {}
    for path in logs:
        name = path.stem
        config = re.sub(r"_rep\d+$", "", name)
        n = i_sum = n_sum = t_sum = 0
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                n += 1
                i_sum += rec.get("instruction_tokens", 0)
                n_sum += rec.get("notice_tokens", 0)
                t_sum += rec.get("total_tokens", 0)
        if n == 0:
            continue
        per_run[name] = {"frames": n, "instruction_mean": round(i_sum / n, 3),
                         "notice_mean": round(n_sum / n, 3), "total_mean": round(t_sum / n, 3)}
        acc = pooled.setdefault(config, [0, 0, 0, 0])
        acc[0] += n
        acc[1] += i_sum
        acc[2] += n_sum
        acc[3] += t_sum
    per_config = {
        c: {"frames": a[0], "instruction_mean": round(a[1] / a[0], 3),
            "notice_mean": round(a[2] / a[0], 3), "total_mean": round(a[3] / a[0], 3)}
        for c, a in sorted(pooled.items())
    }
    out = runs / "token_stats" / "token_stats_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "description": "Per-frame token estimates, (len(text)+2)//4, from text_token_logger.py. "
                       "'instruction' is the first string returned by generate(), which the agent "
                       "sends to LMDrive through the notice slot; 'notice' is the second string. "
                       "per_config pools all frames of a configuration.",
        "per_run": per_run,
        "per_config": per_config,
    }, open(out, "w"), indent=1)
    print(f"token logs: {len(per_run)} runs -> {out}")


def summarize_interventions(runs):
    src = runs / "intervention_logging" / "frames"
    run_dirs = sorted(p for p in src.glob("*_rep*") if p.is_dir()) if src.is_dir() else []
    if not run_dirs:
        print(f"intervention summaries: no run folders in {src}, skipped")
        return
    out_runs = {}
    for run_dir in run_dirs:
        # One folder per route, named by the route start time, so sorted order is run order.
        routes = sorted(p for p in run_dir.iterdir() if p.is_dir())
        entries = []
        for k, route in enumerate(routes):
            summary_path = route / "intervention_summary.json"
            if summary_path.exists():
                s = json.load(open(summary_path))
                entries.append({"route_index": k, **{f: s.get(f, 0) for f in INTERVENTION_FIELDS}})
            else:
                entries.append({"route_index": k, "summary": None})
        out_runs[run_dir.name] = entries
    out = runs / "intervention_logging" / "intervention_counts.json"
    json.dump({
        "description": "Per-route intervention summaries written by the runtime monitor at the end "
                       "of each route. Routes are listed in the order they were run; a route "
                       "without a summary did not finish.",
        "runs": out_runs,
    }, open(out, "w"), indent=1)
    print(f"intervention summaries: {len(out_runs)} runs -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", help="out_dir of scripts/run_paper_experiments.sh")
    runs = Path(ap.parse_args().runs).resolve()
    summarize_tokens(runs)
    summarize_interventions(runs)


if __name__ == "__main__":
    main()
