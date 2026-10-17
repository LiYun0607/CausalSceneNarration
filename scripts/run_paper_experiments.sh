#!/usr/bin/env bash
# Re-run the closed-loop experiments of the paper, one group at a time.
#
#   bash scripts/run_paper_experiments.sh <group> [out_dir]
#
# Groups:
#   multitown      multi-town ablation table, 10 configurations x 5 repetitions
#   baselines      DriveVLM-style and DriveLM-style text, 4 configurations x 5
#   noise          perception noise table: 5 noise configurations x 5, plus
#                  one extra clean repetition (the clean row reuses original_csn)
#   interventions  runtime monitor logging, 4 configurations x 3 repetitions;
#                  saves per-frame images under <out_dir>/intervention_logging/frames/,
#                  where the monitor also writes its per-route summaries
#
# Each run takes 40 to 100 minutes on one GPU. Environment variables are the
# same as for scripts/run_eval.sh. Afterwards:
#   python analysis/summarize_logs.py <out_dir>
#   python analysis/reproduce_paper_numbers.py --results <out_dir>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:-}"
OUT="${2:-runs}"

run() { bash "${HERE}/run_eval.sh" "$1" "$2" "${OUT}/$3"; }

case "${GROUP}" in
    multitown)
        for c in original_lmdrive original_csn original_flat_text original_semantic_safety \
                 baseline ttc_safety semantic_safety csn_only csn_semantic_safety flat_text; do
            for r in 1 2 3 4 5; do run "$c" "$r" multitown_ablation; done
        done ;;
    baselines)
        for c in original_drivevlm original_drivelm original_drivevlm_faithful original_drivelm_faithful; do
            for r in 1 2 3 4 5; do run "$c" "$r" baseline_comparison; done
        done ;;
    noise)
        for c in noise_mild noise_moderate noise_severe noise_extreme flat_noise_severe; do
            for r in 1 2 3 4 5; do run "$c" "$r" noise_ablation; done
        done
        run clean 6 noise_ablation ;;
    interventions)
        for c in baseline_safety csn_safety baseline_nosafety csn_nosafety; do
            for r in 1 2 3; do
                SAVE_PATH="${OUT}/intervention_logging/frames/${c}_rep${r}" \
                    bash "${HERE}/run_eval.sh" "$c" "$r" "${OUT}/intervention_logging"
            done
        done ;;
    *)
        sed -n '2,18p' "$0"
        exit 1 ;;
esac
