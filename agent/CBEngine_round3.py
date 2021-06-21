# -*- coding: utf-8 -*-
import numpy as np
import citypb
from ray import tune
import os
from CBEngine_rllib.CBEngine_rllib import CBEngine_rllib as CBEngine_rllib_class
import argparse


class CBEngine_round3(CBEngine_rllib_class):
    """See CBEngine_rllib_class in /CBEngine_env/env/CBEngine_rllib/CBEngine_rllib.py

    Need to implement reward.

    implementation of observation is optional

    """

    def __init__(self, config):
        super(CBEngine_round3, self).__init__(config)
        self.observation_features = self.gym_dict['observation_features']
        self.custom_observation = self.gym_dict['custom_observation']
        self.observation_dimension = self.gym_dict['observation_dimension']

    def _get_observations(self):

        if (self.custom_observation == False):
            obs = super(CBEngine_round3, self)._get_observations()
            return obs
        else:
            ############
            # implement your own observation
            #

            #############################
            # observation 1 : 120 dimension
            # obs = {}
            # lane_vehicle = self.eng.get_lane_vehicles()
            # for agent_id, roads in self.agent_signals.items():
            #     result_obs = []
            #     # first 12 lanes
            #     for id, lane in enumerate(self.intersections[agent_id]['lanes']):
            #         if (id > 11):
            #             break
            #         if (lane == -1):
            #             if (self.intersections[agent_id]['lanes'][id:id + 3] == [-1, -1, -1]):
            #                 result_obs.append(0)
            #                 result_obs.append(0)
            #                 result_obs.append(0)
            #         else:
            #             if (lane not in lane_vehicle.keys()):
            #                 result_obs.append(0)
            #             else:
            #                 # the vehicle number of this lane
            #                 result_obs.append(len(lane_vehicle[lane]))
            #     # onehot phase
            #     cur_phase = self.agent_curphase[agent_id]
            #     phase_map = [
            #         [-1, -1],
            #         [0, 6],
            #         [1, 7],
            #         [3, 9],
            #         [4, 10],
            #         [0, 1],
            #         [3, 4],
            #         [6, 7],
            #         [9, 10]
            #     ]
            #     one_hot_phase = [0] * 12
            #     one_hot_phase[phase_map[cur_phase][0]] = 1
            #     one_hot_phase[phase_map[cur_phase][1]] = 1
            #     for i in range(4):
            #         one_hot_phase[i * 3 + 2] = 1
            #     result_obs += one_hot_phase

            #     # calc rest 4 intersections
            #     tar_roads = roads[:4]
            #     tar_inters = []
            #     for road in tar_roads:
            #         tar_inter = -1
            #         if (road != -1):
            #             tar_inter = self.roads[road]['start_inter']
            #         tar_inters.append(tar_inter)

            #     for inter in tar_inters:
            #         if (inter == -1):
            #             for kk in range(12):
            #                 result_obs.append(0)
            #             for kk in range(12):
            #                 result_obs.append(1)
            #         else:
            #             if ('lanes' in self.intersections[inter].keys()):
            #                 # first 12 lanes
            #                 for id, lane in enumerate(self.intersections[inter]['lanes']):
            #                     if (id > 11):
            #                         break
            #                     if (lane == -1):
            #                         if (self.intersections[inter]['lanes'][id:id + 3] == [-1, -1, -1]):
            #                             result_obs.append(0)
            #                             result_obs.append(0)
            #                             result_obs.append(0)
            #                     else:
            #                         if (lane not in lane_vehicle.keys()):
            #                             result_obs.append(0)
            #                         else:
            #                             # the vehicle number of this lane
            #                             result_obs.append(len(lane_vehicle[lane]))
            #                 # onehot phase
            #                 cur_phase = self.agent_curphase[inter]
            #                 phase_map = [
            #                     [-1, -1],
            #                     [0, 6],
            #                     [1, 7],
            #                     [3, 9],
            #                     [4, 10],
            #                     [0, 1],
            #                     [3, 4],
            #                     [6, 7],
            #                     [9, 10]
            #                 ]
            #                 one_hot_phase = [0] * 12
            #                 one_hot_phase[phase_map[cur_phase][0]] = 1
            #                 one_hot_phase[phase_map[cur_phase][1]] = 1
            #                 for i in range(4):
            #                     one_hot_phase[i * 3 + 2] = 1
            #                 result_obs += one_hot_phase
            #             else:
            #                 for in_road in self.intersections[inter]['end_roads']:
            #                     for xx in range(3):
            #                         lane = in_road * 100 + xx
            #                         if (lane not in lane_vehicle.keys()):
            #                             result_obs.append(0)
            #                         else:
            #                             result_obs.append(len(lane_vehicle[lane]))
            #                 remain_roads = 4 - len(self.intersections[inter]['end_roads'])
            #                 for kk in range(3 * remain_roads):
            #                     result_obs.append(0)

            #                 for kk in range(12):
            #                     result_obs.append(1)
            #     obs[agent_id] = {"observation": result_obs}
            ##############################

            ##############################
            # observation 2, dim = 36
            obs = {}
            lane_vehicle = self.eng.get_lane_vehicles()
            # red is closer
            lane_vehicle_red = {}
            lane_vehicle_blue = {}

            def get_remain_time(self,vehicle_id):
                '''
                For a vehicle, estimate the lower bound of travel time from current location to the intersection ahead.
                '''
                remain_time = 0
                info = self.eng.get_vehicle_info(vehicle_id)
                speed = info['speed'][0]
                distance = info['distance'][0]
                lane_id = info['drivable'][0]
                road = info['road'][0]
                length = self.roads[road]['length']
                speed_limit = self.roads[road]['speed_limit']

                acc_time = (speed_limit - speed) / 2.0
                acc_dis = (speed_limit + speed)/ 2.0 * acc_time

                if(acc_dis + distance > length):
                    remain_dis = length - distance
                    acc_time_finish = (-speed + np.sqrt(speed * speed + 4 * remain_dis)) / 2.0
                    remain_time = acc_time_finish
                else:
                    remain_time += acc_time
                    distance += acc_dis

                    remain_dis = length - distance
                    remain_time += remain_dis / speed_limit

                return remain_time, lane_id

            v_list = self.eng.get_vehicles()
            threshold = 10.0
            for vehicle in v_list:
                    remain_time, lane_id = get_remain_time(self,vehicle)
                    if(lane_id not in lane_vehicle_red.keys() or lane_id not in lane_vehicle_blue.keys()):
                        lane_vehicle_red[lane_id] = []
                        lane_vehicle_blue[lane_id] = []

                    if(remain_time < threshold):
                        lane_vehicle_red[lane_id].append(vehicle)
                    else:
                        lane_vehicle_blue[lane_id].append(vehicle)
            for agent_id, roads in self.agent_signals.items():
                ##############################
                result_obs = []
                for id, lane in enumerate(self.intersections[agent_id]['lanes']):
                    if(id>11):
                        break
                    if(lane == -1):
                        result_obs.append(0)
                    else:
                        if(lane not in lane_vehicle_red.keys()):
                            result_obs.append(0)
                        else:
                            result_obs.append(len(lane_vehicle_red[lane]))
                for id, lane in enumerate(self.intersections[agent_id]['lanes']):
                    if(id>11):
                        break
                    if(lane == -1):
                        result_obs.append(0)
                    else:
                        if(lane not in lane_vehicle_blue.keys()):
                            result_obs.append(0)
                        else:
                            result_obs.append(len(lane_vehicle_blue[lane]))
                            # if(len(lane_vehicle_blue[lane])>500):
                            #     print("{} : {}".format(lane,lane_vehicle_blue[lane]))
                # onehot phase
                cur_phase = self.agent_curphase[agent_id]
                phase_map = [
                    [-1, -1],
                    [0, 6],
                    [1, 7],
                    [3, 9],
                    [4, 10],
                    [0, 1],
                    [3, 4],
                    [6, 7],
                    [9, 10]
                ]
                one_hot_phase = [0] * 12
                one_hot_phase[phase_map[cur_phase][0]] = 1
                one_hot_phase[phase_map[cur_phase][1]] = 1
                for i in range(4):
                    one_hot_phase[i * 3 + 2] = 1
                result_obs += one_hot_phase



                obs[agent_id] = {"observation": result_obs}
            #########################
            # Here agent_id must be str. So here change int to str
            int_agents = list(obs.keys())
            for k in int_agents:
                obs[str(k)] = obs[k]
                obs.pop(k)

            return obs
            ############

    def _get_reward(self):

        rwds = {}

        ##################
        ## Example : pressure as reward.
        lane_vehicle = self.eng.get_lane_vehicles()
        for agent_id, roads in self.agent_signals.items():
            result_obs = []
            for lane in self.intersections[agent_id]['lanes']:
                # -1 indicates empty roads in 'signal' of roadnet file
                if (lane == -1):
                    result_obs.append(-1)
                else:
                    # -2 indicates there's no vehicle on this lane
                    if (lane not in lane_vehicle.keys()):
                        result_obs.append(0)
                    else:
                        # the vehicle number of this lane
                        result_obs.append(len(lane_vehicle[lane]))
            pressure = (np.sum(result_obs[12: 24]) - np.sum(result_obs[0: 12]))
            rwds[agent_id] = pressure
        ##################

        ##################
        ## Example : queue length as reward.
        # v_list = self.eng.get_vehicles()
        # for agent_id in self.agent_signals.keys():
        #     rwds[agent_id] = 0
        # for vehicle in v_list:
        #     vdict = self.eng.get_vehicle_info(vehicle)
        #     if(float(vdict['speed'][0])<0.5 and float(vdict['distance'][0]) > 1.0):
        #         if(int(vdict['road'][0]) in self.road2signal.keys()):
        #             agent_id = self.road2signal[int(vdict['road'][0])]
        #             rwds[agent_id]-=1
        # normalization for qlength reward
        # for agent_id in self.agent_signals.keys():
        #     rwds[agent_id] /= 10

        ##################

        ##################
        ## Default reward, which can't be used in rllib
        ## self.lane_vehicle_state is dict. keys are agent_id(int), values are sets which maintain the vehicles of each lanes.

        # def get_diff(pre,sub):
        #     in_num = 0
        #     out_num = 0
        #     for vehicle in pre:
        #         if(vehicle not in sub):
        #             out_num +=1
        #     for vehicle in sub:
        #         if(vehicle not in pre):
        #             in_num += 1
        #     return in_num,out_num
        #
        # lane_vehicle = self.eng.get_lane_vehicles()
        # bound of travel time from current location to the intersection ahead
        # for agent_id, roads in self.agents.items():
        #     rwds[agent_id] = []
        #     for lane in self.intersections[agent_id]['lanes']:
        #         # -1 indicates empty roads in 'signal' of roadnet file
        #         if (lane == -1):
        #             rwds[agent_id].append(-1)
        #         else:
        #             if(lane not in lane_vehicle.keys()):
        #                 lane_vehicle[lane] = set()
        #             rwds[agent_id].append(get_diff(self.lane_vehicle_state[lane],lane_vehicle[lane]))
        #             self.lane_vehicle_state[lane] = lane_vehicle[lane]
        ##################
        # Change int keys to str keys because agent_id in actions must be str
        int_agents = list(rwds.keys())
        for k in int_agents:
            rwds[str(k)] = rwds[k]
            rwds.pop(k)
        return rwds
