#!/usr/bin/env bash
# Copy the CSN agent files into an LMDrive checkout.
#
#   bash scripts/install_into_lmdrive.sh /path/to/LMDrive
#
# The original leaderboard/team_code/lmdriver_agent.py is kept as
# lmdriver_agent.py.orig the first time this script runs.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 /path/to/LMDrive" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LMDRIVE="$(cd "$1" && pwd)"
DST="${LMDRIVE}/leaderboard/team_code"

if [ ! -f "${DST}/planner.py" ] || [ ! -f "${LMDRIVE}/langauto/benchmark_short.xml" ]; then
    echo "error: ${LMDRIVE} does not look like an LMDrive checkout" >&2
    exit 1
fi

if [ -f "${DST}/lmdriver_agent.py" ] && [ ! -f "${DST}/lmdriver_agent.py.orig" ]; then
    cp "${DST}/lmdriver_agent.py" "${DST}/lmdriver_agent.py.orig"
fi

mkdir -p "${DST}/safety"
cp "${REPO}"/leaderboard/team_code/*.py "${DST}/"
cp "${REPO}"/leaderboard/team_code/safety/*.py "${DST}/safety/"

echo "Installed into ${DST}:"
(cd "${REPO}/leaderboard/team_code" && ls *.py safety/*.py)
