#!/bin/sh
algo=$1
worker=$2
cfg='/starter-kit/cfg/simulator_round3_flow0.cfg'
stop_iters=10

echo "======================================"
echo "algorithm : ${algo}    cfg : ${cfg}    workers : ${worker}"
nohup python3 rllib_train.py --sim_cfg $cfg --algorithm ${algo} --stop-iters ${stop_iters} --foldername train_result --num_workers ${worker} --thread_num 4 >> ./bash_result/${algo}_${worker}worker.log 2>&1 &

