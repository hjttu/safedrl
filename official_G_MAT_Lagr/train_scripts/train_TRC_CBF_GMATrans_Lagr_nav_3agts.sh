#!/bin/bash

seed_max=1
n_agents=3
ep_lens=100

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
CUDA_VISIBLE_DEVICES='0' python ../onpolicy/scripts/train_mpe.py \
--project_name "TRC_CBF_GMATrans_Lagr" \
--env_name "GSMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "trc_cbf" \
--scenario_name "graph_navigation_3agts" \
--max_edge_dist 1 \
--clip_param 0.2 --gamma 0.99 \
--hidden_size 128 --layer_N 2 \
--num_target 3 --num_agents ${n_agents} --num_obstacle 3 --num_dynamic_obs 0 \
--gp_type "navigation" \
--use_policy "False" \
--use_curriculum "False" \
--entropy_coef 0.02 --cost_value_loss_coef 1 --safety_bound 1.0 \
--lamda_lagr 0.5 --lagrangian_coef_rate 5e-5 --lamda_scale 0.3 \
--use_wandb "False" \
--n_training_threads 16 --n_rollout_threads 32 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--num_env_steps 3000000 \
--ppo_epoch 15 --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --cost_critic_lr 1e-4 \
--user_name "autodl" \
--use_cent_obs "False" \
--graph_feat_type "global" \
--use_att_gnn "False" \
--split_batch "True" --max_batch_size 512 \
--auto_mini_batch_size "True" --target_mini_batch_size 512 \
--use_cbf_filter "True" \
--use_attention_priority "True" \
--use_responsibility "True" \
--use_trm "True" \
--use_cbf_guide "True" \
--cbf_k1 2.0 --cbf_k2 2.0 \
--d_safe_agent 0.25 --d_safe_obstacle 0.30 \
--slack_penalty_base 100.0 --slack_penalty_priority 500.0 \
--beta_min 0.2 --beta_max 0.95 --memory_reset_steps 20 \
--lambda_int 0.1 --lambda_trm 0.05 --lambda_guide_init 0.2 \
--guide_schedule_type "s_curve" \
--eval_hard_filter "True"
done
