#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
# SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

# ALGO='ppo'
ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_NAME='final'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

TRAIN_LIST=('nominal' 'generalization' 'robustness_ob5' 'robustness_ob10' 'robustness_ps3' 'robustness_ps5' 'robustness_ob5ps3' 'robustness_ob10ps5' 'robustness_pm')
# shellcheck disable=SC2054
Q=(3.0,0.1,3.0,0.1,0.1,0.001)

# Train the unsafe controller/agent.
for TRAIN in "${TRAIN_LIST[@]}"; do
    if [ "${TRAIN}" == 'nominal' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}.yaml"
    elif [ "${TRAIN}" == 'generalization' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_gen.yaml"
    elif [ "${TRAIN}" == 'robustness_ob5' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5.yaml"
    elif [ "${TRAIN}" == 'robustness_ob10' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob10.yaml"
    elif [ "${TRAIN}" == 'robustness_ps3' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ps3.yaml"
    elif [ "${TRAIN}" == 'robustness_ps5' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ps5.yaml"
    elif [ "${TRAIN}" == 'robustness_ob5ps3' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5_ps3.yaml"
    elif [ "${TRAIN}" == 'robustness_ob10ps5' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob10_ps5.yaml"
    elif [ "${TRAIN}" == 'robustness_pm' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_pm.yaml"
    fi

    for SEED in {0..4}; do
        python3 ../../safe_control_gym/experiments/train_rl_controller.py \
            --algo ${ALGO} \
            --task ${SYS_NAME} \
            --overrides \
                "${CONFIG1}" \
                "${CONFIG2}" \
            --output_dir ./Results/${EXP_NAME}/${TRAIN} \
            --tag ${SYS}_${ALGO}_data \
            --seed "${SEED}" \
            --use_gpu \
            --kv_overrides \
                task_config.randomized_init=True \
                task_config.normalized_rl_action_space=False\
                task_config.rew_state_weight=${Q} &
    done
    wait
done
