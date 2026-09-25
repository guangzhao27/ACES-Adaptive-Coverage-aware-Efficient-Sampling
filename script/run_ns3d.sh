#!/bin/bash
# Reproduces the NS3D row of Table 1 (joint (x, y, t) spatiotemporal reconstruction
# over 64 consecutive frames): INT, Uniform, Fixed Stratified, and ACES.
#
# Setup:
#   export CONDA_ENV_PATH=/path/to/your/conda/env
#   export DATA_ROOT=/path/to/data   # directory containing NS2d/ns_data_res2048_re10000_7.npy
#   export WANDB_PROJECT=aces-inr-sampling
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  source ~/.bashrc
  conda activate "$CONDA_ENV_PATH"
fi
wandb offline 2>/dev/null || true

DATA_PATH="${DATA_ROOT:?set DATA_ROOT}/NS2d/ns_data_res2048_re10000_7.npy"
PROJECT="${WANDB_PROJECT:-aces-inr-sampling}"

common=(
  inr.model_type=siren inr.depth=10 inr.hidden_dim=384 inr.w0=60 inr.latent_dim=512
  data.dataset_name=NS3D data.data_type=other data.data_path="$DATA_PATH"
  data.space_factor=2 data.volume_time_start=100 data.volume_num_frames=64
  "data.split_ratios=[1,0.01,0.01]"
  optim.optimizer=sgd optim.sgd_nesterov=True optim.sgd_momentum=0.9
  optim.batch_size=2 optim.inner_steps=6 optim.epochs=1500 optim.evo_every_epochs=20
  sampling.rate=7e-4
  save_sampled_frames=False
  wandb.use_wandb=True wandb.project="$PROJECT"
)

for seed in 42 43 44; do
  echo "=== INT seed=$seed ==="
  python inr_sample/single_image_inr.py "${common[@]}" \
      data.seed=$seed optim.lr_inr=1e-4 \
      sampling.type=NMT \
      wandb.name="ns3d_int_seed${seed}"

  echo "=== Uniform seed=$seed ==="
  python inr_sample/single_image_inr.py "${common[@]}" \
      data.seed=$seed optim.lr_inr=2e-3 \
      sampling.type=random \
      wandb.name="ns3d_uniform_seed${seed}"

  echo "=== Fixed Stratified seed=$seed ==="
  python inr_sample/single_image_inr.py "${common[@]}" \
      data.seed=$seed optim.lr_inr=2e-3 \
      sampling.type=3d_grid_stratified sampling.stratified_allocation=proportional \
      "sampling.stratified_n_bins=[16,16,12]" sampling.stratified_min_alloc_frac=0.1 \
      sampling.stratified_update_interval=100 sampling.stratified_pilot_per_cell=16 \
      wandb.name="ns3d_stratified_seed${seed}"

  echo "=== ACES seed=$seed ==="
  python inr_sample/single_image_inr.py "${common[@]}" \
      data.seed=$seed optim.lr_inr=2e-3 \
      sampling.type=3d_grid_adaptive sampling.adaptive_mode=loss_std_score \
      sampling.adaptive_iterations=5 sampling.subdivision_percentage=10 \
      sampling.adaptive_initial_grid_size=8 sampling.adaptive_grid_update_interval=100 \
      sampling.adaptive_equal_cell_topk=False sampling.adaptive_equal_cell_topk_count_mode=same \
      sampling.adaptive_equal_cell_topk_weight_mode=none sampling.adaptive_weight_mode=none \
      wandb.name="ns3d_aces_seed${seed}"
done
