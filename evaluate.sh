#!/bin/bash
agent=$1
out=$2
cfg_array=('/starter-kit/cfg/simulator_round3_flow0.cfg')
vehicle_log=$3
thread_num=$4

for cfg in ${cfg_array[*]}
do
  echo "=========================="
  echo "now test ${cfg}"
  nohup python3 evaluate.py --input_dir $agent --output_dir $out --vehicle_info_path $vehicle_log --sim_cfg ${cfg}  --metric_period 120 --threshold 1.4 --thread_num $thread_num > ./bash_result/evaluate.log 2>&1 &
done



