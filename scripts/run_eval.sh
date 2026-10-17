#!/usr/bin/env bash
# Run one repetition of one paper configuration on the 16-route benchmark.
#
#   bash scripts/run_eval.sh <config> <rep> [out_dir]
#
# Example:
#   bash scripts/run_eval.sh original_csn 1 runs/
#
# The result is written to <out_dir>/<config>_rep<rep>.json in the CARLA
# Leaderboard format, the same format as the files in results/.
#
# Required environment variables:
#   CARLA_ROOT            CARLA 0.9.10.1 installation (with AdditionalMaps)
#   LMDRIVE_ROOT          LMDrive checkout with this repository installed
#                         (scripts/install_into_lmdrive.sh)
#   LMDRIVE_LLM_MODEL     liuhaotian/llava-v1.5-7b directory
#   LMDRIVE_VISION_CKPT   vision-encoder-r50.pth.tar
#   LMDRIVE_LMDRIVE_CKPT  llava-v1.5-checkpoint.pth
#   CSN_LORA_DIR          LoRA adapter directory (only for the PL-DPO-NLL configs)
# Optional:
#   PYTHON (default: python), PORT (2000), TM_PORT (8000),
#   CARLA_WAIT (seconds to wait for CARLA to start, 30),
#   CARLA_ARGS (CARLA window flags, default "-windowed -ResX=1280 -ResY=720";
#               CARLA needs an X display, set DISPLAY),
#   SAVE_PATH (directory for per-frame images; "none" to disable, default none;
#              the runtime monitor writes its per-route intervention summary
#              there, so set it for the intervention-logging runs),
#   DRY_RUN=1 (print the settings of the configuration and exit)
#
# The script stops if the result file already exists. The Leaderboard
# evaluator parses --resume=False as true (argparse type=bool), so an existing
# file would be continued instead of replaced. The paper's runs passed the
# same flag.
set -uo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,33p' "$0"
    exit 1
fi

CONFIG="$1"
REP="$2"
OUT_DIR="${3:-runs}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

: "${CARLA_ROOT:?set CARLA_ROOT}"
: "${LMDRIVE_ROOT:?set LMDRIVE_ROOT}"
: "${LMDRIVE_LLM_MODEL:?set LMDRIVE_LLM_MODEL}"
: "${LMDRIVE_VISION_CKPT:?set LMDRIVE_VISION_CKPT}"
: "${LMDRIVE_LMDRIVE_CKPT:?set LMDRIVE_LMDRIVE_CKPT}"
PYTHON="${PYTHON:-python}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
CARLA_WAIT="${CARLA_WAIT:-30}"
CARLA_ARGS="${CARLA_ARGS:--windowed -ResX=1280 -ResY=720}"

# ---------------------------------------------------------------------------
# Configuration table (see README, "Names used in the paper and in the files")
# ---------------------------------------------------------------------------
LORA="none"; SAFETY="false"; SAFETY_TYPE="disabled"
TEXT="false"; STYLE="csn"; FLAT="false"
NOISE_DIST=0; NOISE_SPEED=0; NOISE_MISS=0

pl() { LORA="${CSN_LORA_DIR:?set CSN_LORA_DIR for PL-DPO-NLL configs}"; }

case "${CONFIG}" in
    # Multi-town ablation, Original LMDrive weights
    original_lmdrive)          ;;
    original_csn|clean)        TEXT=true ;;
    original_flat_text)        TEXT=true; FLAT=true ;;
    original_semantic_safety)  SAFETY=true; SAFETY_TYPE=advanced ;;
    # Multi-town ablation, PL-DPO-NLL LoRA
    baseline|baseline_nosafety)          pl ;;
    ttc_safety)                          pl; SAFETY=true; SAFETY_TYPE=ttc ;;
    semantic_safety|baseline_safety)     pl; SAFETY=true; SAFETY_TYPE=advanced ;;
    csn_only|csn_nosafety)               pl; TEXT=true ;;
    csn_semantic_safety|csn_safety)      pl; TEXT=true; SAFETY=true; SAFETY_TYPE=advanced ;;
    flat_text)                           pl; TEXT=true; FLAT=true ;;
    # DriveVLM-style and DriveLM-style text, Original LMDrive weights
    original_drivevlm)          TEXT=true; STYLE=drivevlm ;;
    original_drivevlm_faithful) TEXT=true; STYLE=drivevlm_faithful ;;
    original_drivelm)           TEXT=true; STYLE=drivelm ;;
    original_drivelm_faithful)  TEXT=true; STYLE=drivelm_faithful ;;
    # Perception noise on CSN inputs, Original LMDrive weights
    noise_mild)        TEXT=true; NOISE_DIST=1.0; NOISE_SPEED=0.10; NOISE_MISS=0.0 ;;
    noise_moderate)    TEXT=true; NOISE_DIST=2.0; NOISE_SPEED=0.20; NOISE_MISS=0.0 ;;
    noise_severe)      TEXT=true; NOISE_DIST=5.0; NOISE_SPEED=0.20; NOISE_MISS=0.10 ;;
    noise_extreme)     TEXT=true; NOISE_DIST=5.0; NOISE_SPEED=0.30; NOISE_MISS=0.20 ;;
    flat_noise_severe) TEXT=true; FLAT=true; NOISE_DIST=5.0; NOISE_SPEED=0.20; NOISE_MISS=0.10 ;;
    *) echo "unknown config: ${CONFIG}" >&2; exit 1 ;;
esac

export LMDRIVE_LORA_PATH="${LORA}"
export LMDRIVE_SAFETY_ENABLED="${SAFETY}"
export LMDRIVE_SAFETY_TYPE="${SAFETY_TYPE}"
export LMDRIVE_TEXT_ENRICHMENT="${TEXT}"
export LMDRIVE_TEXT_STYLE="${STYLE}"
export LMDRIVE_TEXT_FLAT="${FLAT}"
export CSN_NOISE_DIST_STD="${NOISE_DIST}"
export CSN_NOISE_SPEED_REL="${NOISE_SPEED}"
export CSN_NOISE_MISS_RATE="${NOISE_MISS}"
export LMDRIVE_HEADLESS=true
export SDL_VIDEODRIVER=dummy
SAVE_PATH="${SAVE_PATH:-none}"
case "${SAVE_PATH}" in
    ""|none|None|NONE) ;;
    *) mkdir -p "${SAVE_PATH}"; SAVE_PATH="$(cd "${SAVE_PATH}" && pwd)" ;;
esac
export SAVE_PATH
export ROUTES="${LMDRIVE_ROOT}/langauto/benchmark_short.xml"
export CARLA_ROOT
export LEADERBOARD_ROOT="${LMDRIVE_ROOT}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${LMDRIVE_ROOT}/leaderboard:${LMDRIVE_ROOT}/leaderboard/team_code:${LMDRIVE_ROOT}/scenario_runner:${PYTHONPATH:-}"
# Token statistics (the paper logged them for the DriveVLM/DriveLM runs)
case "${STYLE}" in
    drivevlm*|drivelm*) export LMDRIVE_TOKEN_LOG_PATH="${OUT_DIR}/token_stats/${CONFIG}_rep${REP}.jsonl" ;;
esac

if [ "${DRY_RUN:-0}" = "1" ]; then
    # print the settings of this configuration and stop
    env | grep -E '^(LMDRIVE_|CSN_NOISE_|ROUTES=|SAVE_PATH=)' | sort
    exit 0
fi

RESULT="${OUT_DIR}/${CONFIG}_rep${REP}.json"
LOG="${OUT_DIR}/${CONFIG}_rep${REP}.log"
if [ -e "${RESULT}" ]; then
    echo "error: ${RESULT} exists; delete it or choose another out_dir" >&2
    exit 1
fi
echo "[$(date +%H:%M:%S)] ${CONFIG} rep ${REP}: lora=${LORA} safety=${SAFETY}/${SAFETY_TYPE} text=${TEXT}/${STYLE} flat=${FLAT} noise=${NOISE_DIST}/${NOISE_SPEED}/${NOISE_MISS}"

# A fresh CARLA server for every run, rendering at High quality as in the paper.
# shellcheck disable=SC2086
setsid bash "${CARLA_ROOT}/CarlaUE4.sh" --world-port="${PORT}" -quality-level=High \
    ${CARLA_ARGS} > "${OUT_DIR}/carla_${CONFIG}_rep${REP}.log" 2>&1 &
CARLA_PID=$!
stop_carla() { kill -TERM -- "-${CARLA_PID}" 2>/dev/null || kill -TERM "${CARLA_PID}" 2>/dev/null; sleep 3; }
trap stop_carla EXIT
sleep "${CARLA_WAIT}"

"${PYTHON}" -u "${LMDRIVE_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py" \
    --scenarios="${LMDRIVE_ROOT}/leaderboard/data/official/all_towns_traffic_scenarios_public.json" \
    --routes="${ROUTES}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${RESULT}" \
    --agent="${LMDRIVE_ROOT}/leaderboard/team_code/lmdriver_agent.py" \
    --agent-config="${LMDRIVE_ROOT}/leaderboard/team_code/csn_agent_config.py" \
    --debug=1 \
    --resume=False \
    --port="${PORT}" \
    --trafficManagerPort="${TM_PORT}" \
    2>&1 | tee "${LOG}"

echo "[$(date +%H:%M:%S)] wrote ${RESULT}"
