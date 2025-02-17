#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
# SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_NAME='test'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

Q=(15.0,0.1,15.0,0.1,0.1,0.001)

# Train the unsafe controller/agent.
for SEED in {1..1}
do
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${SYS}_${ALGO}_data00 \
        --seed ${SEED} \
        --use_gpu \
        --kv_overrides \
            task_config.randomized_init=True \
            task_config.normalized_rl_action_space=False\
            task_config.rew_state_weight=${Q}
done
