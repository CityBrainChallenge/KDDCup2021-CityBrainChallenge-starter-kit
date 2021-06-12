from agent.CBEngine_round3 import CBEngine_round3 as CBEngine_rllib_class
import gym
import agent.gym_cfg as gym_cfg

# load config
simulator_cfg_file = '/starter-kit/cfg/simulator_round3_flow0.cfg'
mx_step = 360
gym_cfg_instance = gym_cfg.gym_cfg()
gym_configs = gym_cfg_instance.cfg
# gym
env_config = {
    "simulator_cfg_file": simulator_cfg_file,
    "thread_num": 8,
    "gym_dict": gym_configs,
    "metric_period": 200,
    "vehicle_info_path": "/starter-kit/log/"
}
env = CBEngine_rllib_class(env_config)
env.set_info(1)
for i in range(mx_step):
    print("{}/{}".format(i, mx_step))

    # run one step simulation
    # you can use act() in agent.py to get the actions predicted by agent.
    actions = {0: 1}
    obs, rwd, dones, info = env.step(actions)

    # print observations and infos
    # for k, v in obs.items():
    #     print("{}:{}".format(k, v))
    for k, v in info.items():
        print("{}:{}".format(k, v))
