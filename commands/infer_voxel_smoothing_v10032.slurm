#!/bin/bash
#SBATCH --job-name=voxel_infer
#SBATCH -A wbw@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/voxel_infer_%j.out
#SBATCH --error=slurm_logs/voxel_infer_%j.err

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

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/seg_data_voxel_smoothing}"
CONFIG_PATH="${CONFIG_PATH:-configs/shape-autoencoder/Voxel-Smoothing-VAE-train.yaml}"
CKPT_PATH="${CKPT_PATH:-}"
MODE="${MODE:-validate}"
LIMIT_BATCHES="${LIMIT_BATCHES:-10}"
EVAL_VARIANT_INDEX="${EVAL_VARIANT_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
GPU_IDS="${GPU_IDS:-0}"
USE_FLASH="${USE_FLASH:-false}"
SAMPLE_POSTERIOR="${SAMPLE_POSTERIOR:-false}"
SAVE_MESH="${SAVE_MESH:-true}"
EXP_ROOT_DIR="${EXP_ROOT_DIR:-outputs}"
EXP_NAME="${EXP_NAME:-ShapeVAE}"
RUN_TAG="${RUN_TAG:-voxel_infer_${MODE}_${SLURM_JOB_ID:-local}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-@$(date +%Y%m%d-%H%M%S)}"

case "$MODE" in
  validate)
    STAGE_FLAG="--validate"
    LIMIT_KEY="trainer.limit_val_batches"
    SPLIT_FILE="val.json"
    ;;
  test)
    STAGE_FLAG="--test"
    LIMIT_KEY="trainer.limit_test_batches"
    SPLIT_FILE="test.json"
    ;;
  *)
    echo "ERROR: MODE must be either 'validate' or 'test' (got '$MODE')"
    exit 1
    ;;
esac

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: voxel dataset root not found: $DATA_ROOT"
  exit 1
fi

if [ ! -f "$DATA_ROOT/$SPLIT_FILE" ]; then
  echo "ERROR: split file not found: $DATA_ROOT/$SPLIT_FILE"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 1
fi

TRIAL_DIR="${EXP_ROOT_DIR}/${EXP_NAME}/${RUN_TAG}${RUN_TIMESTAMP}"

echo "Resolved data_root: $DATA_ROOT"
echo "Resolved split file: $DATA_ROOT/$SPLIT_FILE"
echo "Resolved checkpoint: $CKPT_PATH"
echo "Resolved mode: $MODE"
echo "Resolved limit batches: $LIMIT_BATCHES"
echo "Resolved eval_variant_index: $EVAL_VARIANT_INDEX"
echo "Expected trial_dir: $TRIAL_DIR"

"$PYTHON_BIN" -u launch_shapevae.py \
  --config "$CONFIG_PATH" \
  "$STAGE_FLAG" \
  --gpu "$GPU_IDS" \
  exp_root_dir="$EXP_ROOT_DIR" \
  name="$EXP_NAME" \
  tag="$RUN_TAG" \
  timestamp="\"$RUN_TIMESTAMP\"" \
  data.root_dir="$DATA_ROOT" \
  data.num_workers="$NUM_WORKERS" \
  data.eval_variant_index="$EVAL_VARIANT_INDEX" \
  system.shape_model.pretrained_model_name_or_path="$CKPT_PATH" \
  system.shape_model.use_flash="$USE_FLASH" \
  system.sample_posterior="$SAMPLE_POSTERIOR" \
  system.save_mesh="$SAVE_MESH" \
  "${LIMIT_KEY}=${LIMIT_BATCHES}"

echo "Voxel inference finished."
echo "Meshes are under ${TRIAL_DIR}/save/"
