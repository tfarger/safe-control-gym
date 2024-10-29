#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='sac'
# ALGO='td3'
# ALGO='safe_explorer_ppo'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

NS=1
# RL Experiment
#for NS in {0,1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}
#do
for SEED in {0..0}
do
  python3 ./rl_experiment.py \
      --task ${SYS_NAME} \
      --algo ${ALGO} \
      --use_gpu \
      --overrides \
          ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
          ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
      --kv_overrides \
          algo_config.training=False \
          task_config.randomized_init=True \
          task_config.task_info.num_cycles=2 \
          task_config.task_info.ilqr_ref=False \
          task_config.task_info.ilqr_traj_data='../lqr/ilqr_ref_traj.npy' \
          task_config.noise_scale=${NS}
#      --pretrain_path ./Results/Benchmark_data/ilqr_ref/${SYS}_${ALGO}_data/${SEED}
done
#done