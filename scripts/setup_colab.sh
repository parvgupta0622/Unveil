#!/usr/bin/env bash
# Installs Unveil's dependencies and fetches the TripoSR source. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

TRIPOSR_COMMIT=107cefdc244c39106fa830359024f6a2f1c78871

echo "==> Installing Python dependencies"
pip install -q -r requirements.txt

if [ ! -f third_party/TripoSR/tsr/system.py ]; then
  echo "==> Cloning TripoSR (pinned to ${TRIPOSR_COMMIT:0:7})"
  mkdir -p third_party
  git clone -q https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR
  git -C third_party/TripoSR checkout -q "$TRIPOSR_COMMIT"
fi

echo "==> Checking imports"
python - <<'EOF'
import torch
from unveil.reconstruction.triposr import ensure_triposr_importable, install_torchmcubes_shim
ensure_triposr_importable()
import tsr.system  # noqa: F401
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
      + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print("marching cubes backend:", install_torchmcubes_shim())
print("Unveil setup OK")
EOF
