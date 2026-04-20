#!/bin/bash
#SBATCH --job-name=shape_recon_10
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=shape_recon_10_%j.out
#SBATCH --error=shape_recon_10_%j.out

set -euo pipefail
set -x

module purge
module load anaconda3 || true
module load cuda/12.1 || true

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate physgen
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/seg_data}"
CKPT_PATH="${CKPT_PATH:-outputs/ShapeVAE/shapevae_output/ckpts/last.ckpt}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
RUN_TAG="${RUN_TAG:-shape_recon_10_random_${SLURM_JOB_ID:-local}}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: data directory not found: $DATA_ROOT"
  exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: pretrained ShapeVAE checkpoint not found: $CKPT_PATH"
  echo "Expected a decoder checkpoint such as outputs/ShapeVAE/shapevae_output/ckpts/last.ckpt"
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/shape_recon10.XXXXXX")"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

export DATA_ROOT TMP_ROOT NUM_SAMPLES SEED
python - <<'PY'
import json
import os
import random
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
tmp_root = Path(os.environ["TMP_ROOT"]).expanduser().resolve()
num_samples = int(os.environ["NUM_SAMPLES"])
seed = int(os.environ["SEED"])

candidates = sorted(
    str(path.resolve())
    for path in data_root.rglob("*.npz")
    if path.is_file()
)

if not candidates:
    raise RuntimeError(f"No .npz files found under {data_root}")

rng = random.Random(seed)
picked = rng.sample(candidates, k=min(num_samples, len(candidates)))

tmp_root.mkdir(parents=True, exist_ok=True)
(tmp_root / "test.json").write_text(json.dumps(picked, indent=2))

print(f"Selected {len(picked)} models from {data_root}:")
for path in picked:
    print(path)
PY

if [ ! -s "$TMP_ROOT/test.json" ]; then
  echo "ERROR: failed to create non-empty test.json in $TMP_ROOT"
  exit 1
fi

# Decoder-only reconstruction path:
# - launch_shapevae.py / Dora-VAE-test.yaml
# - no PhysDec, DragDec, or joint model
# - deterministic latent path via sample_posterior=false
python -u launch_shapevae.py \
  --config configs/shape-autoencoder/Dora-VAE-test.yaml \
  --gpu 0 \
  --test \
  data.root_dir="$TMP_ROOT" \
  data.batch_size=1 \
  data.num_workers="$NUM_WORKERS" \
  system.shape_model.pretrained_model_name_or_path="$CKPT_PATH" \
  system.shape_model.use_flash=false \
  system.sample_posterior=false \
  system.save_mesh=true \
  trainer.limit_test_batches="$NUM_SAMPLES" \
  tag="$RUN_TAG"

echo "Reconstruction finished."
echo "Meshes are under outputs/ShapeVAE/${RUN_TAG}/save/"
