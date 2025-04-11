
# for algo in 'pid'
for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
do
    python3 results_noise.py $algo 'obs_noise'
    python3 results_noise.py $algo 'proc_noise'
done

for algo in 'pid'
for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
do
    python3 results_dw.py $algo
done

for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
# for ADDITIOANL in '_11'
do
    # for algo in 'pid'
    for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
    do
        python3 results_rollout.py $ADDITIOANL $algo 
    done
done
