#!/bin/bash
#SBATCH --job-name=shape_recon_10
#SBATCH -A wbw@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/shape_recon_10_%j.out
#SBATCH --error=slurm_logs/shape_recon_10_%j.err

set +u
set -eo pipefail
set -x

module purge
module load cuda/12.8.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

RAW_DATA_ROOT="${RAW_DATA_ROOT:-${SCRATCH}/seg_data}"
SPLIT_JSON="${SPLIT_JSON:-${SCRATCH}/seg_data_dora/train.json}"
CKPT_PATH="${CKPT_PATH:-}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
RUN_TAG="${RUN_TAG:-shape_recon_train_${SLURM_JOB_ID:-local}}"

if [ ! -d "$RAW_DATA_ROOT" ]; then
  echo "ERROR: raw mesh directory not found: $RAW_DATA_ROOT"
  exit 1
fi

if [ ! -f "$SPLIT_JSON" ]; then
  echo "ERROR: split json not found: $SPLIT_JSON"
  exit 1
fi

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/shape_recon_train.XXXXXX")"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

export RAW_DATA_ROOT SPLIT_JSON TMP_ROOT NUM_SAMPLES SAMPLE_OFFSET
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

raw_root = Path(os.environ["RAW_DATA_ROOT"]).expanduser().resolve()
split_json = Path(os.environ["SPLIT_JSON"]).expanduser().resolve()
tmp_root = Path(os.environ["TMP_ROOT"]).expanduser().resolve()
num_samples = int(os.environ["NUM_SAMPLES"])
sample_offset = int(os.environ["SAMPLE_OFFSET"])

entries = json.loads(split_json.read_text())
if not entries:
    raise RuntimeError(f"No entries found in {split_json}")

window = entries[sample_offset:sample_offset + num_samples]
if not window:
    raise RuntimeError(
        f"Requested samples [{sample_offset}:{sample_offset + num_samples}] but split has only {len(entries)} entries"
    )

picked = []
for sample in window:
    sample_path = Path(sample)
    stem = sample_path.stem
    matches = sorted(raw_root.rglob(f"{stem}.obj"))
    if not matches:
        raise RuntimeError(f"No raw OBJ found for train sample {sample_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple raw OBJs found for {stem}: {matches}")
    picked.append(str(matches[0].resolve()))

tmp_root.mkdir(parents=True, exist_ok=True)
(tmp_root / "test.json").write_text(json.dumps(picked, indent=2))

print(f"Selected {len(picked)} meshes from train split {split_json}:")
for path in picked:
    print(path)
PY

if [ ! -s "$TMP_ROOT/test.json" ]; then
  echo "ERROR: failed to create non-empty test.json in $TMP_ROOT"
  exit 1
fi

"$PYTHON_BIN" -u launch_shapevae.py \
  --config configs/shape-autoencoder/Dora-VAE-test-online-obj.yaml \
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
echo "Meshes and renders are under outputs/ShapeVAE/${RUN_TAG}/save/"
