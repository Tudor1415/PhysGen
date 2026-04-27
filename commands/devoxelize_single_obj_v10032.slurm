#!/bin/bash
#SBATCH --job-name=devoxelize_obj
#SBATCH -A wbw@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/devoxelize_obj_%j.out
#SBATCH --error=slurm_logs/devoxelize_obj_%j.err

set +u
set -eo pipefail
set -x

module purge
module load cuda/12.8.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

INPUT_OBJ="${INPUT_OBJ:-$HOME/surgical_44_45_46_voxels.obj}"
CKPT_PATH="${CKPT_PATH:-}"
CONFIG_PATH="${CONFIG_PATH:-configs/shape-autoencoder/Dora-VAE-test-online-obj.yaml}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
GPU_IDS="${GPU_IDS:-0}"
USE_FLASH="${USE_FLASH:-false}"
SAMPLE_POSTERIOR="${SAMPLE_POSTERIOR:-false}"
SAVE_MESH="${SAVE_MESH:-true}"
EXP_ROOT_DIR="${EXP_ROOT_DIR:-outputs}"
EXP_NAME="${EXP_NAME:-ShapeVAE}"
RUN_TAG="${RUN_TAG:-devoxelize_$(basename "${INPUT_OBJ%.*}")}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-_$(date +%Y%m%d-%H%M%S)}"
TRIAL_DIR="${EXP_ROOT_DIR}/${EXP_NAME}/${RUN_TAG}${RUN_TIMESTAMP}"

if [ ! -f "$INPUT_OBJ" ]; then
  echo "ERROR: input OBJ not found: $INPUT_OBJ"
  exit 1
fi

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/devoxelize_obj.XXXXXX")"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

export INPUT_OBJ TMP_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

input_obj = Path(os.environ["INPUT_OBJ"]).expanduser().resolve()
tmp_root = Path(os.environ["TMP_ROOT"]).expanduser().resolve()

tmp_root.mkdir(parents=True, exist_ok=True)
(tmp_root / "test.json").write_text(json.dumps([str(input_obj)], indent=2))

print(f"Selected input mesh: {input_obj}")
print(f"Temporary dataset root: {tmp_root}")
PY

echo "Resolved input_obj: $INPUT_OBJ"
echo "Resolved checkpoint: $CKPT_PATH"
echo "Resolved config: $CONFIG_PATH"
echo "Expected trial_dir: $TRIAL_DIR"

"$PYTHON_BIN" -u launch_shapevae.py \
  --config "$CONFIG_PATH" \
  --test \
  --gpu "$GPU_IDS" \
  exp_root_dir="$EXP_ROOT_DIR" \
  name="$EXP_NAME" \
  tag="$RUN_TAG" \
  timestamp="\"$RUN_TIMESTAMP\"" \
  data.root_dir="$TMP_ROOT" \
  data.batch_size=1 \
  data.num_workers="$NUM_WORKERS" \
  system.shape_model.pretrained_model_name_or_path="$CKPT_PATH" \
  system.shape_model.use_flash="$USE_FLASH" \
  system.sample_posterior="$SAMPLE_POSTERIOR" \
  system.save_mesh="$SAVE_MESH" \
  trainer.limit_test_batches=1

echo "Devoxelization finished."
echo "Outputs are under ${TRIAL_DIR}/save/"
