#!/bin/bash
set -e
# Run the script
seed_max=1
n_agents=6
ep_lens=100
use_curriculum="False"

for seed in $(seq ${seed_max});
do
    echo "seed: ${seed}"
    # execute the script with different params
    python ../onpolicy/scripts/eval_mpe.py \
    --project_name "GS_GP" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --seed ${seed} \
    --experiment_name "check" \
    --scenario_name "graph_navigation_6agts" \
    --hidden_size 128 \
    --layer_N 2 \
    --use_wandb "False" \
    --save_gifs "False" \
    --use_render "True" \
    --save_data "True" \
    --use_curriculum "False" \
    --use_policy "False" \
    --gp_type "navigation" \
    --render_file_name "6agt-AERL" \
    --num_target 6 \
    --num_agents 6 \
    --num_obstacle 6 \
    --num_dynamic_obs 0 \
    --n_rollout_threads 1 \
    --use_lstm "True" \
    --episode_length ${ep_lens} \
    --use_ReLU --gain 0.01 \
    --user_name "finleygou" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --use_att_gnn "True" \
    --monte_carlo_test "True" \
    --render_episodes 100 \
    --model_dir "/data/goufandi_space/Projects/GS-MARL-GP/GS-MARL-GP/onpolicy/results/GraphMPE/graph_navigation_6agts/rmappo/check/wandb/run-20250904_150321-2gormrpm/files/"
done