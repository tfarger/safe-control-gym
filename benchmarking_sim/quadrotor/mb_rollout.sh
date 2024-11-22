

# for STARTSEED in 1 21 41 61 81
for STARTSEED in 40
do 
    for ADDITIOANL in '_11' # '_10' '_12' '_13' '_14'
    do
        python3 mb_experiment_rollout.py $STARTSEED 20 $ADDITIOANL 'gpmpc_acados'
    done
done

# for algo in 'mpc_acados' # 'linear_mpc' 
# do
#     for STARTSEED in 21 41 61 81
#     do 
#         for ADDITIOANL in  '_13' '_14' # '_10' '_12'
#         do
#         python3 mb_experiment_rollout.py $STARTSEED 20 $ADDITIOANL $algo
#         done
#     done
# done