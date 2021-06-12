from ray import tune
import gym
from agent.CBEngine_round3 import CBEngine_round3 as CBEngine_rllib_class
import citypb
import ray
from ray import tune
import os
import numpy as np
import argparse
import sys
import subprocess
parser = argparse.ArgumentParser()



if __name__ == "__main__":
    # some argument
    parser.add_argument(
        "--num_workers",
        type=int,
        default=30,
        help="rllib num workers"
    )
    parser.add_argument(
        "--multiflow",
        '-m',
        action="store_true",
        default = False,
        help="use multiple flow file in training"
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=10,
        help="Number of iterations to train.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="A3C",
        help="algorithm for rllib"
    )
    parser.add_argument(
        "--sim_cfg",
        type=str,
        default="/starter-kit/cfg/simulator_round3_flow0.cfg",
        help = "simulator file for CBEngine"
    )
    parser.add_argument(
        "--metric_period",
        type=int,
        default=3600,
        help = "simulator file for CBEngine"
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help = "thread num for CBEngine"
    )
    parser.add_argument(
        "--gym_cfg_dir",
        type = str,
        default="agent",
        help = "gym_cfg (observation, reward) for CBEngine"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type = int,
        default = 5,
        help = "frequency of saving checkpoint"
    )

    parser.add_argument(
        "--foldername",
        type = str,
        default = 'train_result',
        help = 'The result of the training will be saved in ./model/$algorithm/$foldername/. Foldername can\'t have any space'
    )

    # find the submission path to import gym_cfg
    args = parser.parse_args()
    for dirpath, dirnames, file_names in os.walk(args.gym_cfg_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "gym_cfg.py":
                cfg_path = dirpath
    sys.path.append(str(cfg_path))
    import gym_cfg as gym_cfg_submission
    gym_cfg_instance = gym_cfg_submission.gym_cfg()
    gym_dict = gym_cfg_instance.cfg
    simulator_cfg_files=[]

    # if set '--multiflow', then the CBEngine will utilize flows in 'simulator_cfg_files'
    if(args.multiflow):
        simulator_cfg_files = [
            '/starter-kit/cfg/simulator_round3_flow0.cfg'
            ]
    else:
        simulator_cfg_files = [args.sim_cfg]
    print('The cfg files of this training   ',format(simulator_cfg_files))
    class MultiFlowCBEngine(CBEngine_rllib_class):
        def __init__(self, env_config):
            env_config["simulator_cfg_file"] = simulator_cfg_files[(env_config.worker_index - 1) % len(simulator_cfg_files)]
            super(MultiFlowCBEngine, self).__init__(config=env_config)


    # some configuration
    env_config = {
        "simulator_cfg_file": args.sim_cfg,
        "thread_num": args.thread_num,
        "gym_dict": gym_dict,
        "metric_period":args.metric_period,
        "vehicle_info_path":"/starter-kit/log/"
    }
    obs_size = gym_dict['observation_dimension']
    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1e10, high=1e10, shape=(obs_size,))
    })
    ACTION_SPACE = gym.spaces.Discrete(9)
    stop = {
        "training_iteration": args.stop_iters
    }
    ################################
    # modify this
    tune_config = {
        # env config
        "env":MultiFlowCBEngine,
        "env_config" : env_config,
        "multiagent": {
            "policies": {
                "default_policy": (None, OBSERVATION_SPACE, ACTION_SPACE, {},)
            }
        },

        "num_cpus_per_worker":args.thread_num,
        "num_workers":args.num_workers,



        # add your training config

    }
    ########################################
    ray.init(address = "auto")
    local_path = './model'
    


    def name_creator(self=None):
        return args.foldername


    # train model
    ray.tune.run(args.algorithm, config=tune_config, local_dir=local_path, stop=stop,
                 checkpoint_freq=args.checkpoint_freq,trial_dirname_creator = name_creator)


