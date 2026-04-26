#!/bin/bash
#SBATCH --job-name=voxel_smoothing_pre
#SBATCH -A wbw@v100
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --output=slurm_logs/voxel_smoothing_pre_%j.out
#SBATCH --error=slurm_logs/voxel_smoothing_pre_%j.err

set +u
set -eo pipefail
set -x

module purge
module load gcc/14.2.0

GCC_LIBSTDCPP="$(g++ -print-file-name=libstdc++.so.6)"
export LD_PRELOAD="$GCC_LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$(dirname "$GCC_LIBSTDCPP")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL
unset PYTHONPATH

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
echo "LD_PRELOAD=$LD_PRELOAD"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

"$PYTHON_BIN" - <<'PY'
import pysdf
print("pysdf import OK")
PY

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/seg_data}"
PROC_ROOT="${PROC_ROOT:-${SCRATCH}/seg_data_voxel_smoothing}"

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
