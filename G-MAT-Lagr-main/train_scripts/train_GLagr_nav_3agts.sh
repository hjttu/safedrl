#!/bin/bash

# Run the script
seed_max=1
n_agents=3
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=100
export WANDB_BASE_URL=https://api.bandw.top

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
CUDA_VISIBLE_DEVICES='0' python  ../onpolicy/scripts/train_mpe.py \
--use_valuenorm --use_popart \
--project_name "GS_GP" \
--env_name "GSMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_navigation_3agts" \
--max_edge_dist 1 \
--clip_param 0.2 --gamma 0.99 \
--hidden_size 128 --layer_N 2 \
--num_target 3 --num_agents 3 --num_obstacle 3 --num_dynamic_obs 0 \
--gp_type "navigation" \
--save_data "True" \
--reward_file_name "r_navigation_3agts_GLag_noGP-v2" \
--cost_file_name "c_navigation_3agts_GLag_noGP-v2" \
--use_policy "False" \
--use_curriculum "False" \
--guide_cp 0.4 --cp 0.4 --js_ratio 0.0 \
--entropy_coef 0.02 --cost_value_loss_coef 1 --safety_bound 1.0 \
--lamda_lagr 0.5 --lagrangian_coef_rate 5e-5 --lamda_scale 0.3 \
--use_wandb "True" \
--n_training_threads 16 --n_rollout_threads 32 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--num_env_steps 3000000 \
--ppo_epoch 15 --use_ReLU --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --cost_critic_lr 1e-4 \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "False" \
--split_batch "True" --max_batch_size 512 \
--auto_mini_batch_size "True" --target_mini_batch_size 512
done

# &> $logs_folder/out_${ep_lens}_${seed} \
# --num_mini_batch 64 \