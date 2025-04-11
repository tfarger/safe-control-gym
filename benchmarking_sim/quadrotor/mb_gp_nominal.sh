
GP_BASE_TAG='100_200'
# for RAND_TYPE in '_dw_h=1dot5' '_dw_h=2' '_dw_h=2dot5' '_dw_h=3'
# for RAND_TYPE in '_ob_ns=5' '_ob_ns=15' '_ob_ns=25'  
# for RAND_TYPE in  '_proc_ns=25' # '_proc_ns=5' '_proc_ns=15'
# for RAND_TYPE in '_tr'

# get the time
START_TIME=$(date +%s)

# for RAND_TYPE in ''
# for RAND_TYPE in '_param'
# for RAND_TYPE in '' '_param' '_tr'\
#                  '_ob_ns=5_proc_ns=3' \
#                  '_ob_ns=10_proc_ns=5' \
#                  '_ob_ns=15_proc_ns=7' \
#                  '_ob_ns=20_proc_ns=10' \
#                  '_ob_ns=5' '_ob_ns=10' '_ob_ns=15' '_ob_ns=20' \
#                  '_proc_ns=3' '_proc_ns=5' '_proc_ns=7' '_proc_ns=10'
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP' $RAND_TYPE
    
#     # python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG
#     # python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
#     # python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
#     # python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
#     # for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
#     # do
#     #     for algo in 'gpmpc_acados_TP'
#     #     do
#     #         python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
#     #     done
#     # done
# done

# nominal
for RAND_TYPE in ''
do
    GP_TAG=$GP_BASE_TAG$RAND_TYPE
    python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG
    python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
    python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
    python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
    for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
    do
        for algo in 'gpmpc_acados_TP'
        do
            python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
        done
    done
done

# python3 ../del_acados_files.py
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Elapsed time: $ELAPSED_TIME seconds"