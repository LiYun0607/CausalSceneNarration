#!/usr/bin/env bash
# Build the weights archive attached to the GitHub release.
#
#   bash scripts/pack_lora_release.sh /path/to/pldpo-v5-turn-focused-r32-final [out.zip]
#
# The archive holds adapter_model.safetensors from the given directory and the
# adapter_config.json of weights/ (training-server path removed). The SHA-256 of
# both files is checked against weights/MANIFEST.json.
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,8p' "$0"
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(cd "$1" && pwd)"
OUT="${2:-CausalSceneNarration_lora_v1.zip}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${STAGE}/CausalSceneNarration_lora_v1"
cp "${SRC}/adapter_model.safetensors" "${STAGE}/CausalSceneNarration_lora_v1/"
cp "${REPO}/weights/adapter_config.json" "${REPO}/weights/MANIFEST.json" "${STAGE}/CausalSceneNarration_lora_v1/"

"${PYTHON:-python3}" - "${STAGE}/CausalSceneNarration_lora_v1" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
man = json.load(open(d / "MANIFEST.json"))["files"]
for name, info in man.items():
    h = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if h != info["sha256"]:
        sys.exit(f"checksum mismatch for {name}: {h}")
    print(f"ok  {name}  {h}")
EOF

(cd "${STAGE}" && zip -r -q - CausalSceneNarration_lora_v1) > "${OUT}"
echo "wrote ${OUT}"
