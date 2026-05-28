#!/bin/bash

seed_max=1
n_agents=3
ep_lens=100

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
CUDA_VISIBLE_DEVICES='0' python ../onpolicy/scripts/train_mpe.py \
--project_name "GMAT_CBF_ActionMask" \
--env_name "GSMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "cbf_action_mask" \
--scenario_name "graph_navigation_3agts" \
--max_edge_dist 1.0 \
--clip_param 0.2 --gamma 0.99 \
--hidden_size 128 --layer_N 2 \
--num_target 3 --num_agents ${n_agents} --num_obstacle 3 --num_dynamic_obs 0 \
--gp_type "navigation" \
--save_data "True" \
--reward_file_name "r_navigation_3agts_CBF_ActionMask" \
--cost_file_name "c_navigation_3agts_CBF_ActionMask" \
--use_policy "False" \
--use_curriculum "False" \
--guide_cp 0.4 --cp 0.4 --js_ratio 0.0 \
--entropy_coef 0.02 --cost_value_loss_coef 1 --safety_bound 1.0 \
--lamda_lagr 0.5 --lagrangian_coef_rate 5e-5 --lamda_scale 0.3 \
--use_wandb "False" \
--n_training_threads 16 --n_rollout_threads 16 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--num_env_steps 1000000 \
--ppo_epoch 10 --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --cost_critic_lr 1e-4 \
--user_name "autodl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "False" \
--split_batch "True" --max_batch_size 256 \
--auto_mini_batch_size "True" --target_mini_batch_size 256 \
--use_cbf_filter "False" \
--use_cbf_action_mask "True" \
--use_continuous_cbf_filter "False" \
--use_attention_priority "False" \
--use_responsibility "False" \
--use_trm "False" \
--use_cbf_guide "False" \
--cbf_k1 2.0 --cbf_k2 2.0 \
--d_safe_agent 0.20 --d_safe_obstacle 0.25 \
--action_mask_hard "True" \
--action_mask_soft_penalty "True" \
--h_keep 0.05 --tau_ttc 1.0 --lambda_soft_mask 1.0 \
--empty_mask_fallback "min_violation" \
--neighbor_action_mode "zero"
done
