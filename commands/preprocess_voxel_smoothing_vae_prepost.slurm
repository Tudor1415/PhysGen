#!/bin/bash
#SBATCH --job-name=voxel_smoothing_pre
#SBATCH -A wbw@v100
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/voxel_smoothing_pre_%j.out
#SBATCH --error=slurm_logs/voxel_smoothing_pre_%j.err

set +u
set -eo pipefail
set -x

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1

PHYSGEN_VENV="${PHYSGEN_VENV:-${WORK}/PhysGen/physgen}"

if [ -z "$PHYSGEN_VENV" ] || [ ! -f "$PHYSGEN_VENV/bin/activate" ]; then
  echo "ERROR: physgen virtualenv not found."
  echo "Set PHYSGEN_VENV to the venv root containing bin/activate."
  exit 1
fi

# shellcheck disable=SC1091
source "$PHYSGEN_VENV/bin/activate"

PYTHON_BIN="$(command -v python)"
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/voxel_smoothing}"
PROC_ROOT="${PROC_ROOT:-${SCRATCH}/voxel_smoothing_vae}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: data directory not found: $DATA_ROOT"
  exit 1
fi

mkdir -p "$PROC_ROOT"

"$PYTHON_BIN" preprocess/voxel_smoothing_preprocess.py \
  --data_root "$DATA_ROOT" \
  --output_root "$PROC_ROOT"

export PROC_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

proc_root = Path(os.environ["PROC_ROOT"]).expanduser().resolve()
processed_path = proc_root / "processed_meshes.json"
invalid_path = proc_root / "invalid_meshes.json"

processed = json.loads(processed_path.read_text()) if processed_path.exists() else []
invalid = json.loads(invalid_path.read_text()) if invalid_path.exists() else []

print(f"processed={len(processed)}")
print(f"invalid={len(invalid)}")
print(f"dataset_root={proc_root}")
print(f"split_files={proc_root / 'train.json'} {proc_root / 'val.json'} {proc_root / 'test.json'}")
PY
