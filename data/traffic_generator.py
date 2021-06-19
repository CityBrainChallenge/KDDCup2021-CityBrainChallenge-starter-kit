import networkx as nx
import random
from tqdm import tqdm
class Flow:
    def __init__(self, roadnet_path='roadnet_round3.txt', graph=True, roadgraph_path=None):
        '''
        :param roadmap_path: road map file
        :param graph: whether to build road graph again
        :param roadgraph_path: path to load road graph
        '''
        self.roadgraph = None # a networkX Graph instance
        self.read_roadnet(roadnet_path=roadnet_path) 
        if graph:
            self.generate_roadGraph()
        else:
            if roadgraph_path is None:
                exit("The road graph path is not provided!")
            self.roadgraph = nx.read_gpickle()
        
        self.left_lon = 115.7501 # left border longitude of the traffic zone, please do not change this default value
        self.right_lon = 115.9878 # right border longitude of the traffic zone, please do not change this default value
        self.bottom_lat = 28.5951 # bottom border latitude of the traffic zone, please do not change this default value
        self.top_lat = 28.7442 # top border latitude of the traffic zone, please do not change this default value
        self.flow = []
        self.Veh_num_cur = 0 # total number of generated vehicle trips until now
        self.zone_info = None # zone information data
        self.Oprob = None # probabilities of a vehicle departs from Origin traffic zones
        self.ODprob = None # probabilities of a vehicle depart from an Origin zone and arrived into a Destination zone
        self.traffic_duration = 1200 # By default, 1200-second traffic sample data is generated

    def clear(self):
        self.flow = []
        self.Veh_num_cur = 0
        self.zone_info = None

    def divide_roadNet(self, numRows=6, numColumns=8, Oprob_mode=3, prob_corner=0.2, prob_mid=0.15):
    
        '''
        Divide road network into numRows * numColumns rectangle traffic zones
        return: IDs (rowIndex, columnIndex) of created traffic sub-zones
        '''
        if type(numRows) != int or type(numColumns) != int:
            exit("Please enter integer numRows and numColumns.")

        self.numRows = numRows
        self.numColumns = numColumns
        self.unit_lon = (self.right_lon - self.left_lon) / self.numColumns
        self.unit_lat = (self.top_lat - self.bottom_lat) / self.numRows
        self.zone_info = {}
        for row in range(self.numRows):
            for col in range(self.numColumns):
                self.zone_info[(row, col)] = {
                    'inters_id': [], # a list of IDs of intersections
                    'left_lon': self.left_lon + col*self.unit_lon, # leftmost longitude of the zone
                    'right_lon': self.left_lon + (col+1)*self.unit_lon, # rightmost logitude of the zone
                    'top_lat': self.top_lat - row*self.unit_lat, # top latitude of the zone
                    'bottom_lat':self.top_lat - (row+1)*self.unit_lat, # bottom latitude of the zone
                    'num_inters': 0, # number of intersections within the zone
                    'num_signals': 0, # number of signalized intersections within the zone
                    'roadlen': 0.0, # road length within the zone
                    'roadseg':0 # number of road segments within the zone
                }
        for key in self.inter_info.keys():
            zone_id = self.get_inter_zoneID(key)
            self.zone_info[zone_id]['inters_id'].append(key)
            self.zone_info[zone_id]['num_inters'] += 1
            self.zone_info[zone_id]['num_signals'] += self.inter_info[key]['sign']
            self.zone_info[zone_id]['roadlen'] += self.inter_info[key]['roadlen']
            self.zone_info[zone_id]['roadseg'] += self.inter_info[key]['roadseg']
        print('Divide road network into {}*{} traffic zones successfully!'.format(self.numRows, self.numColumns))
        
        # Estimate the OD matrix (in form of probabilities) based on road network information
        self.get_Oprob(Oprob_mode) # Get probabilities of a vehicle departs from zones (Origin zone) of the road network
        self.get_ODprob(prob_corner, prob_mid) #Get probabilities of a vehicle depart from an Origin zone and arrived into a Destination zone
        
        return self.zone_info.keys()

    def generate_traffic(self, numVeh=50000, percentVeh=[.3,.1,.2,.2,.2], weight=0.2):
        '''
        Generate 1 sample traffic flow data given the road network data 
        :param numVeh: total number of vehicles that will enter the network in 1-hour
        :param percentVeh: percentages of vehicles that will enter the network in each period (e.g., in 4-minute)  
        :param weight: the larger the weight, the more diverse of route choice given Origin-Destination of a trip 
        '''
        if self.zone_info is None:
            exit('You need to divide road network into traffic sub-zones!')
        
        if abs(sum(percentVeh) - 1) > 0.001:
            exit('sum of percentages of vehicle entering in network over time should be 1!')

        num_intervals = len(percentVeh) # total number of periods with different traffic demands
        interval_length  = self.traffic_duration / len(percentVeh) # compute the duration of one period (e.g., 600 seconds)
        numVeh_perInterval = [int(percent * numVeh) for percent in percentVeh] # number of vehicles entering the network over periods
        numVeh_generated = 0 # record number of generated vehicle trips

        for interval in tqdm(range(num_intervals)):
            for o_zone in self.zone_info.keys():
                if len(self.zone_info[o_zone]['inters_id']) == 0:
                    continue
                num_per_inter = numVeh_perInterval[interval] * self.Oprob[o_zone] / self.zone_info[o_zone]['num_inters'] # number of vehicles per intersection
                for o_interid in self.zone_info[o_zone]['inters_id']:
                    d_zone = self.random_weight_choose(self.ODprob[o_zone]) # choose destination zone given ODprobs
                    while len(self.zone_info[d_zone]['inters_id']) == 0:
                        d_zone = self.random_weight_choose(self.ODprob[o_zone])
                    d_interid = random.choice(self.zone_info[d_zone]['inters_id'])
                    
                    start_time = random.randint(interval * interval_length, interval * interval_length + 10) # start time is in range of interval begining and +10 seconds
                    end_time = (interval + 1) * interval_length 
                    
                    try:
                        tmp_flow, tmp_num = self.get_route(o_interid, d_interid, num_per_inter, start_time, end_time, weight)
                    except TypeError:
                        continue
                    self.flow.append(tmp_flow)
                    numVeh_generated += tmp_num
        print("Generating {} Vehicles. Current {} Vehicles in total.".format(int(numVeh_generated), int(self.Veh_num_cur)))
        
    def add_tripOD(self, o_zone, d_zone, start_time, end_time, num_Veh, weight=0.2):
        '''
            Add extra trips from o_zone to d_zone in addition to background traffic
            This will help you simulate traffic during special events, for example, football games
        '''
        if end_time > self.traffic_duration or start_time < 0:
            exit("end_time should be in time range of [0, traffic_duration]!")

        if o_zone[0] >= self.numRows or d_zone[0] > self.numRows:
            exit("The zone row index should be smaller than numRows!")

        if o_zone[1] >= self.numColumns or d_zone[1] > self.numColumns:
            exit("The zone column index should be smaller than numColumns!")

        num = 0
        num_per_inter = num_Veh / self.zone_info[o_zone]['num_inters']
        for o_interid in self.zone_info[o_zone]['inters_id']:
            d_interid = random.choice(self.zone_info[d_zone]['inters_id'])
            try:
                tmp_flow, tmp_num = self.get_route(o_interid, d_interid, num_per_inter, start_time, end_time, weight)
            except TypeError:
                continue
            self.flow.append(tmp_flow)
            num += tmp_num
        print('Adding {} vehicles. Current {} Vehicles'.format(int(num), int(self.Veh_num_cur)))

    def get_route(self, O_interid, D_interid, numVeh, start_time, end_time, weight=0.2):
        '''
        get route from o_inter to d_inter
        :param O_interid:
        :param D_interid:
        :param numVeh:
        :param start_time:
        :param end_time:
        :param weight: 
        :return:
        '''
        start_time = int(start_time)
        end_time = int(end_time)
        path_nodes = nx.shortest_path(self.roadgraph, source=O_interid, target=D_interid, weight='length') # return a list of intersection IDs

        if len(path_nodes) <= 3: # we omit the vehicle trips with number of edges <= 3 
            return None

        interval = max(1,int((end_time - start_time) / numVeh))
        tmp_data = []
        tmp_data.append(start_time)
        tmp_data.append(end_time)
        tmp_data.append(interval)
        tmp_data.append(len(path_nodes) - 1) # append number of edges (road segments)

        for idx in range(len(path_nodes) - 1):
            self.roadgraph[path_nodes[idx]][path_nodes[idx + 1]]['length'] += numVeh * weight # update the 'length' of roadgraph considering the number of vehicles passed the edge
            
            edge_id = self.roadgraph[path_nodes[idx]][path_nodes[idx + 1]]['id'] # generate edge (road segment) ID given two adjacent node (intersection) IDs 
            tmp_data.append(edge_id)
        self.Veh_num_cur += (end_time-start_time) / interval
        return tmp_data, (end_time-start_time) / interval


    def output(self, output_path):
        '''
        Write the traffic flow data into a .txt file
        :param output_path:
        :return: self.zone_info.keys()
        '''
        file = open(output_path, 'w')
        file.write("{}\n".format(len(self.flow)))
        for i in self.flow:
            for j in range(len(i)):
                if (j == 2) or (j == 3) or (j == len(i) - 1):
                    file.write("{}\n".format(i[j]))
                else:
                    file.write("{} ".format(i[j]))
        file.close()


    def get_Oprob(self, mode=None):
        '''
        Get probabilities of a vehicle departs from zones (Origin zone) of the road network,
        Default method: use road length within zones as default reference for probabilities estimation

        :param mode: 
            1->use road length as reference for origin zone probabilities estimation 
            2->use number of road segments as reference for origin zone probabilities estimation  
            3->use number of intersections as reference for origin zone probabilities estimation 
            4->use number of signalized intersection as reference for origin zone probabilities estimation
        :return:
        '''
        if mode == 1:
            self.Oprob = {(row, col): self.zone_info[(row,col)]['roadlen']
                          for row in range(self.numRows) for col in range(self.numColumns)}
        elif mode == 2:
            self.Oprob = {(row, col): self.zone_info[(row, col)]['roadseg']
                          for row in range(self.numRows) for col in range(self.numColumns)}
        elif mode == 3:
            self.Oprob = {(row, col): self.zone_info[(row, col)]['num_inters']
                          for row in range(self.numRows) for col in range(self.numColumns)}
        elif mode == 4:
            self.Oprob = {(row, col): self.zone_info[(row, col)]['num_signals']
                          for row in range(self.numRows) for col in range(self.numColumns)}
        else:
            exit("Mode Error")
        total = sum(self.Oprob.values())
        for key in self.Oprob.keys():
            self.Oprob[key] /= total
        return self.Oprob

    def get_ODprob(self, prob_corner=0.3, prob_mid=0.8):
        '''
        Get probabilities of a vehicle depart from an Origin zone and arrived into a Destination zone
        Default method: use zone's row and column indices difference to estimate the probabilities

        :param prob_corner: probability for a vehicle's Origin and Destination within the same zone if it is a corner traffic zone, e.g. zone-(0,0)
        :param prob_mid: probability for a vehicle's Origin and Destination within the same zone if it is a middle zone, e.g. zone-(3,4)
        :return:
        '''
        self.ODprob = {}  # (row, col):{(row2, col2): probability}
        for o_row in range(self.numRows):
            for o_col in range(self.numColumns):
                self.ODprob[(o_row, o_col)] = {}
                for d_row in range(self.numRows):
                    for d_col in range(self.numColumns):
                        dis = abs(o_row - d_row) + abs(o_col - d_col)
                        if dis == 0:
                            if (o_row == 0 or o_row == 5 or o_col == 0 or o_col == 7):
                                self.ODprob[(o_row, o_col)][(d_row, d_col)] = prob_corner
                            else:
                                self.ODprob[(o_row, o_col)][(d_row, d_col)] = prob_mid
                        else:
                            self.ODprob[(o_row, o_col)][(d_row, d_col)] = 1 / dis
        return

    def get_zoneInfo(self, zoneid=None):
        '''Return information of a traffic zone given zoneID'''
        if zoneid is None:
            return self.zone_info
        if zoneid[0] not in range(self.numRows) or zoneid[1] not in range(self.numColumns):
            print('Error: zone id')
            return
        print(self.zone_info[zoneid])
        return self.zone_info[zoneid]

    def get_inter_zoneID(self, inter_id):
        '''Given an intersection ID, return the traffic zone ID - (rowIndex, columnIndex)'''
        lat = self.inter_info[inter_id]['lat']
        lon = self.inter_info[inter_id]['lon']

        row = int((lat - self.bottom_lat) / self.unit_lat)
        col = int((lon - self.left_lon) / self.unit_lon)
        return (row, col)

    def generate_roadGraph(self):
        '''Translate roadnetwork data into networkX format, return a networkX graph'''
        DG = nx.DiGraph()
        for key in self.inter_info.keys():
            DG.add_node(key, **self.inter_info[key])  # node_id
        for key in self.road_info.keys():
            pair = (self.road_info[key]['ininter_id'], self.road_info[key]['outinter_id'])
            DG.add_edge(*pair, **{"id": key, "length": self.road_info[key]['roadlen'], "speed": self.road_info[key]['speed']})

        nx.write_gpickle(DG, "roadGraph.gpickle")
        print('Building road networkX graph successfully!')
        self.roadgraph = DG

    def read_roadnet(self, roadnet_path):
        '''Read road network data'''
        roadnet = open(roadnet_path, 'r')

        # read inters
        self.inter_num = int(roadnet.readline())
        self.inter_info = {}
        print("Total number of intersections:{}".format(self.inter_num))
        for _ in range(self.inter_num):
            line = roadnet.readline()
            lat, lon, id, sign = self.read_inter_line(line)
            self.inter_info[id] = {'lat': lat, 'lon': lon, 'sign': sign, 'roadlen': 0.0, 'roadseg':0}

        # read roads
        self.road_info = {}
        self.road_num = int(roadnet.readline())
        print("Total number of road segments:{}".format(self.road_num))
        for _ in range(self.road_num):
            line = roadnet.readline()
            inter_id1, inter_id2, roadlen, speed, road_id1, road_id2 = self.read_road_line(line)
            #print(road_id1, road_id2)
            self.road_info[road_id1] = {'ininter_id': inter_id1, 'outinter_id': inter_id2,
                                        'roadlen': roadlen, 'speed': speed}
            self.road_info[road_id2] = {'ininter_id': inter_id2, 'outinter_id': inter_id1,
                                        'roadlen': roadlen, 'speed': speed}
            self.inter_info[inter_id1]['roadlen'] += roadlen
            self.inter_info[inter_id2]['roadlen'] += roadlen
            self.inter_info[inter_id1]['roadseg'] += 1
            self.inter_info[inter_id2]['roadseg'] += 1
            roadnet.readline()
            roadnet.readline()
        roadnet.close()

    def read_inter_line(self, line):
        '''Read intersection data line-by-line'''
        line = line.split()
        lat = float(line[0])
        lon = float(line[1])
        id = int(line[2])
        sign = bool(line[3])
        return lat, lon, id, sign

    def read_road_line(self, line):
        '''Read road segment data line-by-line'''
        line = line.split()
        inter_id1 = int(line[0])
        inter_id2 = int(line[1])
        roadlen = float(line[2])
        speed = float(line[3])

        road_id1 = int(line[6])
        road_id2 = int(line[7])
        return inter_id1, inter_id2, roadlen, speed, road_id1, road_id2

    def random_weight_choose(self, weight_data):
        '''
        Helper function for choosing the Destination zone of a vehicle trip, given weight_data = self.ODprob[o_zone]
        return the destination traffic zone, i.e., d_zone - (rowIndex, columnIndex)
        '''
        total = sum(weight_data.values())   
        ran = random.uniform(0, total)  
        curr_sum = 0
        d_zone = None

        for key in weight_data.keys():
            curr_sum += weight_data[key]  
            if ran <= curr_sum: 
                d_zone = key
                break
        return d_zone



fl = Flow(roadnet_path='roadnet_round3.txt')
fl.divide_roadNet(numRows=6, numColumns=8, Oprob_mode=3)
fl.generate_traffic(numVeh=50000, percentVeh=[.3,.1,.2,.2,.2], weight=0.2)

fl.add_tripOD(o_zone=(0,0), d_zone=(2,0), start_time=360, end_time=700, num_Veh=2000)
fl.add_tripOD(o_zone=(1,2), d_zone=(0,0), start_time=500, end_time=700, num_Veh=1500)
fl.add_tripOD(o_zone=(1,3), d_zone=(2,5), start_time=400, end_time=680, num_Veh=3000)
fl.add_tripOD(o_zone=(2,2), d_zone=(2,4), start_time=400, end_time=700, num_Veh=2000)
fl.output(output_path='flow_round3.txt')
