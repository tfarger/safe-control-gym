#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

# ALGO='ppo'
ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_DATA='final/robustness_ps5'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

NS=1
T=11
H=0
EVAL_LIST=('performance' 'robustness_ob' 'robustness_ps' 'robustness_dw' 'generalization')
# EVAL_LIST=('robustness_ps')

# RL Experiment
for EVAL in "${EVAL_LIST[@]}"; do
    if [ "${EVAL}" == 'robustness_ob' ]; then
        EXTERNAL_PARAM=(0 1 2 3 4 5 10 15 20 25 30 40 50 60 70 80 90 100)
    elif [ "${EVAL}" == 'robustness_ps' ]; then
        EXTERNAL_PARAM=(0 1 2 3 4 5 10 15 20 25 30 40 50 60 70 80 90 100)
    elif [ "${EVAL}" == 'robustness_dw' ]; then
        EXTERNAL_PARAM=(1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.5 4.0 4.5 5.0)
    elif [ "${EVAL}" == 'generalization' ]; then
        EXTERNAL_PARAM=(9 10 11 12 13 14 15)
    else
        EXTERNAL_PARAM=(1)
    fi

    for EP in "${EXTERNAL_PARAM[@]}"; do
        for SEED in {0..4}; do
            python3 ./rl_experiment.py \
                --task ${SYS_NAME} \
                --algo ${ALGO} \
                --use_gpu \
                --overrides \
                    ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                    ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                --experiment_type ${EVAL} \
                --seed ${SEED} \
                --kv_overrides \
                    algo_config.training=False \
                    task_config.normalized_rl_action_space=False \
                    task_config.randomized_init=True \
                    task_config.task_info.num_cycles=2 \
                    task_config.episode_len_sec=${T} \
                    task_config.noise_scale=${NS} \
                    task_config.downwash_height=${H} \
                    task_config.external_param=${EP} \
                --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data/seed${SEED}_*/ &
        done
        wait
    done
done