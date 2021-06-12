import gym
from agent.CBEngine_round3 import CBEngine_round3 as CBEngine_rllib_class
from ray.rllib.models import ModelCatalog
import tensorflow as tf
import citypb
import ray
from pathlib import Path
from ray import tune
import os
import re
import numpy as np
import argparse
import sys
import pickle
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
parser = argparse.ArgumentParser()

class FT_agent():
    def __init__(self, interval):
        self.interval = interval
    def act(self,obs):
        actions = {}
        for k,v in obs.items():
            actions[k] = (v//self.interval) % 8 + 1
        return actions

class MP_agent():
    def __init__(self, interval):
        self.now_phase = {}
        self.green_sec = interval
        self.last_change_step = {}
        
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
    def get_phase_pressures(self, lane_vehicle_num):
        pressures = []
        for i in range(8):
            in_lanes = self.phase_lane_map_in[i]
            out_lanes = self.phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += lane_vehicle_num[in_lane] * 3
            for out_lane in out_lanes:
                pressure -= lane_vehicle_num[out_lane]
            pressures.append(pressure)
        # # print("pressures: ", pressures)
        return pressures

    def get_action(self, lane_vehicle_num):
        pressures = self.get_phase_pressures(lane_vehicle_num)
        unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)

        max_pressure_id = np.argmax(pressures) + 1
        while (max_pressure_id in unavailable_phases):
            pressures[max_pressure_id - 1] -= 999999
            max_pressure_id = np.argmax(pressures) + 1
        # # print(max_pressure_id)
        return max_pressure_id



    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases
        # return [5, 6, 7, 8]


    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # preprocess observations
        # a simple fixtime agent
        observations_for_agent = obs
        self.agent_list = list(obs.keys())
        actions = {}
        for agent in self.agent_list:
            # select the now_step
            now_step = observations_for_agent[agent][0]
            lane_vehicle_num = observations_for_agent[agent]
            # print("agent id: ", agent)
            # print("lane vehicle: ", lane_vehicle_num)
            
            action = self.get_action(lane_vehicle_num)
            # print("action: ", action)
            if(agent not in self.last_change_step.keys()):
                self.last_change_step[agent] = 0
                self.now_phase[agent] = 1
            step_diff = now_step - self.last_change_step[agent]
            if (step_diff >= self.green_sec):
                self.now_phase[agent] = action
                self.last_change_step[agent] = now_step


            actions[agent] = self.now_phase[agent]
            # print("phase: ", actions[agent])
            # print("phase available lane: ", self.phase_lane_map_in[actions[agent]-1])
            # print("________")

        return actions

class RLlibTFCheckpointPolicy():
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        self._checkpoint_path = load_path
        self._algorithm = algorithm
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        if self._sess:
            return

        if self._algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif self._algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif self._algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif self._algorithm in ["DQN","APEX"]:
            from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy as LoadPolicy
        else:
            raise TypeError("Unsupport algorithm")

        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        with tf.name_scope(self._policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            self.policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(self._checkpoint_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[self._policy_name]
            list_keys = list(weights.keys())
            for k in list_keys:
                if(k not in self.policy.get_weights().keys()):
                    weights.pop(k)
            self.policy.set_weights(weights)

    def act(self, obs):
        action = {}
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self.policy.compute_actions(obs, explore=False)[0]
        elif isinstance(obs, dict):
            for k,v in obs.items():
                obs = self._prep.transform(v)
                action[k] = self.policy.compute_actions([obs], explore=False)[0][0]
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self.policy.compute_actions([obs], explore=False)[0][0]

        return action
def process_delay_index(lines, roads,step):
    vehicles = {}

    for i in range(len(lines)):
        line = lines[i]
        if(line[0] == 'for'):
            vehicle_id = int(line[2])
            now_dict = {
                'distance': float(lines[i + 1][2]),
                'drivable': int(float(lines[i + 2][2])),
                'road': int(float(lines[i + 3][2])),
                'route': list(map(int, list(map(float, lines[i + 4][2:])))),
                'speed': float(lines[i + 5][2]),
                'start_time': float(lines[i + 6][2]),
                't_ff': float(lines[i+7][2]),
            ##############
                'step': int(lines[i+8][2])
            }
            step = now_dict['step']
            ##################
            vehicles[vehicle_id] = now_dict
            tt = step - now_dict['start_time']
            tt_ff = now_dict['t_ff']
            tt_f_r = 0.0
            current_road_pos = 0
            for pos in range(len(now_dict['route'])):
                if(now_dict['road'] == now_dict['route'][pos]):
                    current_road_pos = pos
            for pos in range(len(now_dict['route'])):
                road_id = now_dict['route'][pos]
                if(pos == current_road_pos):
                    tt_f_r += (roads[road_id]['length'] -
                               now_dict['distance']) / roads[road_id]['speed_limit']
                elif(pos > current_road_pos):
                    tt_f_r += roads[road_id]['length'] / roads[road_id]['speed_limit']
            vehicles[vehicle_id]['tt_f_r'] = tt_f_r
            vehicles[vehicle_id]['delay_index'] = (tt + tt_f_r) / tt_ff

    vehicle_list = list(vehicles.keys())
    delay_index_list = []
    for vehicle_id, dict in vehicles.items():
        # res = max(res, dict['delay_index'])
        if('delay_index' in dict.keys()):
            delay_index_list.append(dict['delay_index'])

    # 'delay_index_list' contains all vehicles' delayindex at this snapshot.
    # 'vehicle_list' contains the vehicle_id at this snapshot.
    # 'vehicles' is a dict contains vehicle infomation at this snapshot
    return delay_index_list, vehicle_list, vehicles

def process_score(log_path,roads,step,scores_dir,travel_time):
    result_write = {
        "data": {
            "total_served_vehicles": -1,
            "delay_index": -1,
            'average_travel_time':travel_time
        }
    }
    with open(log_path / "info_step {}.log".format(step)) as log_file:
        lines = log_file.readlines()
        lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
        # process delay index
        delay_index_list, vehicle_list, vehicles = process_delay_index(lines, roads, step)
        v_len = len(vehicle_list)
        delay_index = np.mean(delay_index_list)

        result_write['data']['total_served_vehicles'] = v_len
        result_write['data']['delay_index'] = delay_index
        with open(scores_dir / 'scores {}.json'.format(step), 'w' ) as f_out:
            json.dump(result_write,f_out,indent= 2)

    return result_write['data']['total_served_vehicles'],result_write['data']['delay_index']

def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if(len(line) == 3 and line[0][0] != '#'):
                configs[line[0]] = line[-1]
    return configs

def process_roadnet(roadnet_file):
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'have_signal': int(line[3]),
                        'end_roads': [],
                        'start_roads': [],
                        'lanes':[]
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents

def get_agent(target_iteration,algorithm,foldername):
    ACTION_SPACE = gym.spaces.Discrete(9)
    for dirpath, dirnames, file_names in os.walk('agent'):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "gym_cfg.py":
                cfg_path = dirpath
    sys.path.append(str(cfg_path))
    import gym_cfg as gym_cfg_submission
    gym_cfg_instance = gym_cfg_submission.gym_cfg()
    gym_dict = gym_cfg_instance.cfg
    obs_size = gym_dict['observation_dimension']
    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1e10, high=1e10, shape=(obs_size,))
    })
    agents = []

    result_path = Path('./model')
    for dirpath, dirnames, file_names in os.walk(result_path / algorithm /foldername,topdown=True):
        dir_list = dirpath.split('/')
        if(dir_list[-1].startswith("checkpoint")):
            iteration = int(dir_list[-1][-6:])
            if(target_iteration!=-1 and iteration != target_iteration):
                continue
            model_path = os.path.join(dirpath,'checkpoint-{}'.format(iteration))
            agents.append(
                (
                    RLlibTFCheckpointPolicy(load_path=model_path, algorithm=algorithm, policy_name="default_policy",
                                            observation_space=OBSERVATION_SPACE, action_space=ACTION_SPACE)
                    , iteration
                )
            )

    return agents

def write_cfg(cfgs,path):
    with open(path,'w') as f:
        for k,v in cfgs.items():
            f.write('{} : {}\n'.format(k,v))
        

if __name__ == "__main__":
    # parser.add_argument(
    #     "--sim_cfg",
    #     type=str,
    #     required=True,
    #     help = "simulator file for CBEngine"
    # )
    parser.add_argument(
        "--use_half",
        action="store_true",
        default = False,
        help="first half Model. second half MP"
    )

    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help = "thread num for CBEngine"
    )
    parser.add_argument(
        "--sim_cfg",
        type=str,
        required=True,
        help = "which simulator cfg to be evaluated"
    )
    parser.add_argument(
        "--gym_cfg_dir",
        type = str,
        default="agent",
        help = "gym_cfg (observation, reward) for CBEngine"
    )
    parser.add_argument(
        "--metric_period",
        type=int,
        default=3600,
        help = "simulator file for CBEngine"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required = True,
        help = "algorithm to be evaluate"
    )
    parser.add_argument(
        '--iteration',
        type = int,
        required = True,
        help = "which iteration to be evaluated"
    )
    parser.add_argument(
        '--foldername',
        type = str,
        required =True,
        help = "which folder in model/${algorithm}/"
    )
    args = parser.parse_args()
    for dirpath, dirnames, file_names in os.walk(args.gym_cfg_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "gym_cfg.py":
                cfg_path = dirpath
    sys.path.append(str(cfg_path))
    import gym_cfg as gym_cfg_submission
    gym_cfg_instance = gym_cfg_submission.gym_cfg()
    gym_dict = gym_cfg_instance.cfg
    metric_period = args.metric_period


    # ray.init()
    
    mp_agent_instance = MP_agent(2)
    # get agents
    if(args.algorithm not in ['FT','MP']):
        agents = get_agent(target_iteration=args.iteration, algorithm=args.algorithm, foldername = args.foldername)
    elif(args.algorithm == 'FT'):
        agents = [(FT_agent(2),None)]
    elif(args.algorithm == 'MP'):
        agents = [(MP_agent(2),None)]
        gym_dict['observation_features'] = ['lane_vehicle_num']
    # get sim_cfg list
    sim_cfgs = []
    if(args.sim_cfg==None):
        sim_cfgs = [
            '/starter-kit/cfg/simulator_round3_flow0.cfg'
            ]
    else:
        sim_cfgs = [args.sim_cfg]

    ################ scoreing
    for agent,iteration in agents:
        for sim_cfg in sim_cfgs:
            simulator_configs = read_config(sim_cfg)
            
            log_path = Path(simulator_configs['report_log_addr'])
            model_log_path = log_path / args.foldername
            score_path = model_log_path / 'iteration_{}/'.format(iteration)
            if(not os.path.exists(log_path)):
                os.makedirs(log_path)
            if(not os.path.exists(model_log_path)):
                os.makedirs(model_log_path)
            if(not os.path.exists(score_path)):
                os.makedirs(score_path)

            logger.info("log_path :{}\nmodel_log_path :{}\nscore_path :{}".format(log_path,model_log_path,score_path))
            # change cfg
            simulator_configs = read_config(sim_cfg)
            simulator_configs['report_log_addr'] =   str(score_path) + '/'

            sim_cfg_modified = sim_cfg[:-4] + "_iter{}_{}.cfg".format(iteration,args.foldername) 
            write_cfg(simulator_configs,sim_cfg_modified)


            env_config = {
                "simulator_cfg_file": sim_cfg_modified,
                "thread_num": args.thread_num,
                "gym_dict": gym_dict,
                "metric_period": args.metric_period,
                "vehicle_info_path":score_path
            }
            
            
            roadnet_path = Path(simulator_configs['road_file_addr'])
            intersections, roads, agents = process_roadnet(roadnet_path)
            env = CBEngine_rllib_class(env_config)
            env.set_log(1)
            obs = env.reset()
            dones = {"__all__":False}
            step = 0
            while not dones['__all__']:
                step+=1
                if(args.use_half==False):
                    if(args.algorithm == 'FT'):
                        for k, v in obs.items():
                            obs[k] = step
                    elif(args.algorithm == 'MP'):
                        for k,v in obs.items():
                            temp = obs[k]['observation']
                            obs[k] = [step]
                            obs[k] += temp
                    for k,v in obs.items():
                        logger.info("cur_phase: {}".format(env.agent_curphase[int(k)]))
                        break
                    action = agent.act(obs)
                else:
                    if(step < 180):
                        action = agent.act(obs)
                    else:
                        for k,v in obs.items():
                            temp = obs[k]['observation']
                            obs[k] = [step]
                            obs[k] += temp
                        action = mp_agent_instance.act(obs)            
                obs, reward, dones, info = env.step(action)
                for k,v in obs.items():
                    logger.info("step : {}\nobs : {}\naction: {}\n=====================".format(step,obs[k],action[k]))
                    break
                if (step * 10 % metric_period == 0):
                    try:
                        tot_v, d_i = process_score(score_path, roads, step * 10 - 1, score_path, travel_time = env.eng.get_average_travel_time())
                    except Exception as e:
                        logger.error(e)
                        logger.error('Error in process_score. Maybe no log')
                        continue

            logger.info("cfgfile : {} , iteration {} evaluating finished".format(sim_cfg,iteration))
            os.remove(sim_cfg_modified)


