#!/bin/bash

# Lightweight 6-agent Graph-CBF RMAPPO profile for an 8 GB laptop GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

seed_max=1
ep_lens=50

for seed in $(seq "${seed_max}"); do
    echo "seed: ${seed}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python ../onpolicy/scripts/train_mpe.py \
        --use_valuenorm --use_popart \
        --project_name "GS_GP_laptop" \
        --env_name "GSMPE" \
        --algorithm_name "rmappo" \
        --seed "${seed}" \
        --experiment_name "graph_cbf_6agts_laptop" \
        --scenario_name "graph_navigation_6agts" \
        --max_edge_dist 1.0 \
        --action_grid_size 9 \
        --use_local_dtcbf_shield "True" \
        --use_joint_dtcbf_shield "True" \
        --joint_shield_mode "decentralized_priority" \
        --cbf_max_accel 0.5 --cbf_max_speed 1.0 \
        --cbf_alpha 0.3 \
        --dtcbf_horizon 3 \
        --dtcbf_predict_mode "constant_action" \
        --dtcbf_min_margin 0.0 \
        --dtcbf_early_brake_buffer 0.05 \
        --predictive_hard_neighbors 1 \
        --min_joint_domain_size 5 \
        --predictive_soft_penalty_coef 2.0 \
        --use_soft_predictive_penalty "True" \
        --use_joint_repair "True" \
        --joint_repair_top_k 8 \
        --joint_repair_max_cluster_size 4 \
        --joint_repair_include_blockers "True" \
        --no_safe_action_strategy "backup" \
        --backup_action_mode "brake" \
        --clip_param 0.2 --gamma 0.99 \
        --hidden_size 64 --layer_N 1 \
        --num_target 6 --num_agents 6 --num_obstacle 6 --num_dynamic_obs 0 \
        --gp_type "navigation" \
        --save_data "False" \
        --use_policy "False" \
        --use_curriculum "False" \
        --guide_cp 0.4 --cp 0.4 --js_ratio 0.0 \
        --entropy_coef 0.02 --cost_value_loss_coef 1 --safety_bound 1.0 \
        --lamda_lagr 0.1 --lagrangian_coef_rate 5e-5 --lamda_scale 0.3 \
        --use_wandb "False" \
        --n_training_threads 4 --n_rollout_threads 4 \
        --use_lstm "False" \
        --episode_length "${ep_lens}" \
        --num_env_steps 500000 \
        --ppo_epoch 3 --use_ReLU --gain 0.01 \
        --lr 1e-4 --critic_lr 1e-4 --cost_critic_lr 1e-4 \
        --user_name "local" \
        --use_cent_obs "False" \
        --graph_feat_type "relative" \
        --use_att_gnn "False" \
        --gnn_hidden_size 16 --gnn_layer_N 1 --gnn_num_heads 1 \
        --embed_hidden_size 16 --embed_layer_N 1 \
        --split_batch "True" --max_batch_size 128 \
        --auto_mini_batch_size "True" --target_mini_batch_size 128 \
        --train_metric_window 30 \
        --best_model_cost_limit 1.0 \
        --log_interval 1
done
