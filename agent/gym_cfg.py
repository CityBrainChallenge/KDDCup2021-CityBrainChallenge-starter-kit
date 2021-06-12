class gym_cfg():
    def __init__(self):
        '''
        'custom_observation': If 'True', use costom observation feature in CBEngine_round3.py of agent.zip. If 'False', use 'observation_features'

        'observation_features' : Same as round2. Add 'classic' observation feature, which has dimension of 16.

        'observation_dimension' : The dimension of observation. Need to be correct both custom observation and default observation.

        '''

        self.cfg = {
            'observation_features':['lane_vehicle_num','classic'],
            'observation_dimension':40,
            'custom_observation' : False
        }