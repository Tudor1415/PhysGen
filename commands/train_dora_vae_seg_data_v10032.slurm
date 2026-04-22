#!/bin/bash
#SBATCH --job-name=dora_vae_train
#SBATCH -A wbw@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/dora_vae_train_%j.out
#SBATCH --error=slurm_logs/dora_vae_train_%j.err

set +u
set -eo pipefail
set -x

module purge
module load cuda/12.8.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

PHYSGEN_VENV="${PHYSGEN_VENV:-${WORK}/PhysGen/physgen}"
if [ ! -f "$PHYSGEN_VENV/bin/activate" ]; then
  echo "ERROR: physgen virtualenv not found: $PHYSGEN_VENV"
  exit 1
fi

# shellcheck disable=SC1091
source "$PHYSGEN_VENV/bin/activate"

PYTHON_BIN="$(command -v python)"
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/seg_data_dora}"
CONFIG_PATH="${CONFIG_PATH:-configs/shape-autoencoder/Dora-VAE-train.yaml}"
RUN_TAG="${RUN_TAG:-shapevae_train_seg_data_${SLURM_JOB_ID:-local}}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
INIT_CKPT="${INIT_CKPT:-./outputs/Dora-VAE-1.1/dora_vae_1_1.ckpt}"
EXP_ROOT_DIR="${EXP_ROOT_DIR:-${WORK}/PhysGen}"
EXP_NAME="${EXP_NAME:-ckpts}"
RUN_TIMESTAMP="$(date +@%Y%m%d-%H%M%S)"
TRIAL_DIR="${EXP_ROOT_DIR}/${EXP_NAME}/${RUN_TAG}${RUN_TIMESTAMP}"
CKPT_DIR="${TRIAL_DIR}/ckpts"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: preprocessed dataset root not found: $DATA_ROOT"
  exit 1
fi

for split_file in train.json val.json test.json; do
  if [ ! -f "$DATA_ROOT/$split_file" ]; then
    echo "ERROR: missing split file: $DATA_ROOT/$split_file"
    exit 1
  fi
done

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi

echo "Resolved exp_root_dir: $EXP_ROOT_DIR"
echo "Resolved experiment name: $EXP_NAME"
echo "Resolved run tag: $RUN_TAG"
echo "Expected trial_dir: $TRIAL_DIR"
echo "Expected checkpoint dir: $CKPT_DIR"

srun "$PYTHON_BIN" -u launch_shapevae.py \
  --config "$CONFIG_PATH" \
  --train \
  --gpu "$GPU_IDS" \
  exp_root_dir="$EXP_ROOT_DIR" \
  name="$EXP_NAME" \
  timestamp="$RUN_TIMESTAMP" \
  data.root_dir="$DATA_ROOT" \
  data.batch_size="$BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  tag="$RUN_TAG" \
  system.shape_model.pretrained_model_name_or_path="$INIT_CKPT"
