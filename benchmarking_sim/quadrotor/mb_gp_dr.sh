
GP_BASE_TAG='100_200'
# get the time
START_TIME=$(date +%s)

# param
for RAND_TYPE in '_param' 
do
    GP_TAG=$GP_BASE_TAG$RAND_TYPE
    python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP' $RAND_TYPE
    python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
done

# python3 ../del_acados_files.py
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Elapsed time: $ELAPSED_TIME seconds"