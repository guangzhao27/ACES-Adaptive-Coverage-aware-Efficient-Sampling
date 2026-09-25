#!/bin/bash
# Reproduces the NS2D (Navier-Stokes, Re=10,000) curves in Figure 3a:
# Uniform, INT, EVOS, Fixed Stratified, and ACES, at matched sampling
# rate and iteration budget, across five temporal snapshots and three seeds.
#
# Setup:
#   export CONDA_ENV_PATH=/path/to/your/conda/env      # or just `conda activate` your own env
#   export DATA_ROOT=/path/to/data                      # directory containing NS2d/ns_data_res2048_re10000_7.npy
#   export WANDB_PROJECT=aces-inr-sampling               # optional, for logging
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
  data.dataset_name=NS data.data_path="$DATA_PATH" data.data_type=other
  data.space_factor=1 "data.split_ratios=[1,0.01,0.01]"
  inr.model_type=siren inr.depth=3 inr.hidden_dim=155 inr.w0=30 inr.latent_dim=256
  optim.optimizer=sgd optim.sgd_nesterov=True optim.sgd_momentum=0.9
  optim.batch_size=2 optim.inner_steps=6 optim.epochs=5000 optim.evo_every_epochs=100
  optim.lr_inr=1e-5
  sampling.rate=2e-3
  save_sampled_frames=False
  wandb.use_wandb=True wandb.project="$PROJECT"
)

for time_frame in 100 120 140 160 180 200; do
  for seed in 42 43 44; do
    common_run=("${common[@]}" data.seed=$seed data.single_time_frame=$time_frame)

    echo "=== Uniform t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=random \
        wandb.name="ns2d_uniform_t${time_frame}_seed${seed}"

    echo "=== INT t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=NMT \
        wandb.name="ns2d_int_t${time_frame}_seed${seed}"

    echo "=== EVOS t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=EVOS \
        sampling.n_clusters_2d_start=11 sampling.n_clusters_2d_end=128 \
        sampling.sample_num_schedular=constant sampling.mutation_method=constant \
        sampling.profile_interval_method=lin_dec sampling.profile_guide=value \
        wandb.name="ns2d_evos_t${time_frame}_seed${seed}"

    echo "=== Fixed Stratified t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=2d_grid_stratified sampling.stratified_allocation=neyman \
        sampling.stratified_n_bins=23 sampling.stratified_min_alloc_frac=0.1 \
        sampling.stratified_update_interval=500 sampling.stratified_pilot_per_cell=16 \
        wandb.name="ns2d_stratified_t${time_frame}_seed${seed}"

    echo "=== ACES t=$time_frame seed=$seed ==="
    python inr_sample/single_image_inr.py "${common_run[@]}" \
        sampling.type=2d_grid_adaptive sampling.adaptive_mode=loss_variance_score \
        sampling.adaptive_iterations=7 sampling.subdivision_percentage=22 \
        sampling.adaptive_grid_update_interval=200 sampling.adaptive_equal_cell_topk=False \
        sampling.adaptive_weight_mode=loss_powered_weight sampling.power_for_loss_as_weight=1.0 \
        wandb.name="ns2d_aces_t${time_frame}_seed${seed}"
  done
done
