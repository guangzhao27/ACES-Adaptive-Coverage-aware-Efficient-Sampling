#!/bin/bash
# Reproduces the component ablation in Figure 4 (PoolBoiling2D): Uniform,
# Fixed + Weighting (fixed grid + residual-aware weighting), Partition Only
# (Unbiased) (adaptive partition, no weighting), and full ACES (adaptive
# partition + residual-aware weighting).
#
# Setup:
#   export CONDA_ENV_PATH=/path/to/your/conda/env
#   export DATA_ROOT=/path/to/data                      # directory containing PoolBoiling-SubCooled-FC72-2D/
#   export WANDB_PROJECT=aces-inr-sampling
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source ~/.bashrc
  conda activate "$CONDA_ENV_PATH"
fi
wandb offline 2>/dev/null || true

DATA_PATH="${DATA_ROOT:?set DATA_ROOT}/PoolBoiling-SubCooled-FC72-2D"
PROJECT="${WANDB_PROJECT:-aces-inr-sampling}"

common=(
  data.dataset_name=PoolBoiling2D data.data_path="$DATA_PATH" data.data_type=other
  data.poolboiling_condition=100 data.poolboiling_key=temperature data.poolboiling_sample_idx=0
  "data.split_ratios=[1,0.01,0.01]"
  inr.model_type=siren inr.depth=3 inr.hidden_dim=155 inr.w0=30 inr.latent_dim=256
  optim.optimizer=sgd optim.sgd_nesterov=True optim.sgd_momentum=0.9
  optim.batch_size=2 optim.inner_steps=6 optim.epochs=5000 optim.evo_every_epochs=100
  optim.lr_inr=1e-3
  sampling.rate=2e-3
  save_sampled_frames=False
  wandb.use_wandb=True wandb.project="$PROJECT"
)

for time_frame in 30 70 110 150 190; do
  for seed in 42 43 44; do
    common_run=("${common[@]}" data.seed=$seed data.single_time_frame=$time_frame)

    echo "=== Uniform t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=random \
        wandb.name="pool2d_ablation_uniform_t${time_frame}_seed${seed}"

    echo "=== Fixed + Weighting t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=2d_grid_linear_weighted sampling.use_weight_function=True \
        wandb.name="pool2d_ablation_fixedweight_t${time_frame}_seed${seed}"

    echo "=== Partition Only (Unbiased) t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=2d_grid_adaptive sampling.adaptive_mode=loss_variance_score \
        sampling.adaptive_iterations=5 sampling.subdivision_percentage=20.0 \
        sampling.adaptive_equal_cell_topk=True sampling.adaptive_equal_cell_topk_count_mode=same \
        sampling.adaptive_equal_cell_topk_weight_mode=unbiased_weight \
        sampling.adaptive_weight_mode=unbiased_weight sampling.power_for_loss_as_weight=1.0 \
        wandb.name="pool2d_ablation_unbiased_t${time_frame}_seed${seed}"

    echo "=== ACES t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=2d_grid_adaptive sampling.adaptive_mode=loss_variance_score \
        sampling.adaptive_iterations=4 sampling.subdivision_percentage=20.0 \
        sampling.adaptive_grid_update_interval=500 \
        sampling.adaptive_equal_cell_topk=True sampling.adaptive_equal_cell_topk_count_mode=same \
        sampling.adaptive_equal_cell_topk_weight_mode=loss_powered_weight \
        sampling.adaptive_weight_mode=none sampling.power_for_loss_as_weight=0.25 \
        wandb.name="pool2d_ablation_aces_t${time_frame}_seed${seed}"
  done
done
