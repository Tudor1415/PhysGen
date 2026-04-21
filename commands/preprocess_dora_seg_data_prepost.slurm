#!/bin/bash
#SBATCH --job-name=dora_preprocess
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/dora_preprocess_%j.out
#SBATCH --error=slurm_logs/dora_preprocess_%j.err

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

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/seg_data}"
PROC_ROOT="${PROC_ROOT:-${SCRATCH}/seg_data_dora}"
RESOLUTION="${RESOLUTION:-512}"
POINT_NUMBER="${POINT_NUMBER:-371088}"
ANGLE_THRESHOLD="${ANGLE_THRESHOLD:-15}"
TRAIN_RATIO="${TRAIN_RATIO:-0.90}"
VAL_RATIO="${VAL_RATIO:-0.05}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: data directory not found: $DATA_ROOT"
  exit 1
fi

mkdir -p "$PROC_ROOT"
mkdir -p "$PROC_ROOT/sharp_ply"
mkdir -p "$PROC_ROOT/samples"

OBJ_JSON="$PROC_ROOT/meshes_obj.json"

"$PYTHON_BIN" preprocess/detect_path.py \
  --directory_to_search "$DATA_ROOT" \
  --json_file_path "$OBJ_JSON" \
  --file_type .obj

"$PYTHON_BIN" preprocess/sharp_sample.py \
  --json_file_path "$OBJ_JSON" \
  --point_number "$POINT_NUMBER" \
  --angle_threshold "$ANGLE_THRESHOLD" \
  --sharp_point_path "$PROC_ROOT/sharp_ply" \
  --sample_path "$PROC_ROOT/samples"

export PROC_ROOT TRAIN_RATIO VAL_RATIO
"$PYTHON_BIN" - <<'PY'
import json
import os
import random
from pathlib import Path

proc_root = Path(os.environ["PROC_ROOT"]).expanduser().resolve()
train_ratio = float(os.environ["TRAIN_RATIO"])
val_ratio = float(os.environ["VAL_RATIO"])

samples_root = proc_root / "samples"
files = sorted(str(path.resolve()) for path in samples_root.glob("*.npz"))

if not files:
    raise RuntimeError(f"No .npz files found under {samples_root}")

random.Random(42).shuffle(files)

n = len(files)
n_train = int(train_ratio * n)
n_val = int(val_ratio * n)
n_train = max(1, min(n_train, n))
n_val = max(0, min(n_val, n - n_train))

train = files[:n_train]
val = files[n_train:n_train + n_val]
test = files[n_train + n_val:]

if not test:
    test = val[-1:] if val else train[-1:]
    if val:
        val = val[:-1]
    else:
        train = train[:-1]

(proc_root / "train.json").write_text(json.dumps(train, indent=2))
(proc_root / "val.json").write_text(json.dumps(val, indent=2))
(proc_root / "test.json").write_text(json.dumps(test, indent=2))

print(f"Prepared dataset at {proc_root}")
print(f"train={len(train)} val={len(val)} test={len(test)}")
PY

echo "Preprocessing finished."
echo "Dataset root: $PROC_ROOT"
echo "Splits: $PROC_ROOT/train.json $PROC_ROOT/val.json $PROC_ROOT/test.json"
echo "Samples: $PROC_ROOT/samples"
