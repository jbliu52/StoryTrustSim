import random

'''
Experiment definitions for various scenarios. Each of these returns a dictionary specifying
events that will be used in a Storyboard.
'''

def mislead_experiment(mislead, length):
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "same"}
    event_dict[11] = {"name":"move", "actors":['1'], "location": ['1']}
    event_dict[12] = {"name": "exclusive_random", "actors": ['0','1'], "stop": 12 + mislead}
    event_dict[12 + mislead] = {"name":"move", "actors":['1'],"location":['2']}
    event_dict[12 + mislead+1] = {"name": "exclusive_random", "actors": ['0',"1"], "stop": length}
    #experiment_info = {'cross path location': loc[0], 'poi':poi, 'last':third_loc}
    return event_dict, "1"



def second_order_tom_experiment(mislead, length):
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": ['0','1','2'], "location": ['0'], "path_type":"same"}
    event_dict[16] = {"name": "cross_paths","actors": ['1','2'], "location": ['1'], "path_type":"same", "prev": ['0']}
    event_dict[17] = {"name": "exclusive_random", "actors": ['0','1','2'], "stop": 17 + mislead}
    event_dict[17 + mislead] = {"name":"move", "actors":['2'],"location":['2']}
    event_dict[17 + mislead + 1] = {"name": "exclusive_random", "actors": ['0','1','2'], "stop": length}
    # experiment_info = {'cross path locs': [loc_1, loc_2], 'poi': poi, 'last': loc_3}
    return event_dict, "1"

    
# Deprecated
def cross_path_overlap(actors, locs, g, mislead, length, n):
    poi = random.sample(actors, n)
    loc = random.sample(locs, 1)
    second_loc = random.choice(g[loc[0]])
    event_dict = {}
    event_dict[15] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
    event_dict[16] = {"name":"move", "actor":poi[-1], "location": second_loc}
    event_dict[17] = {"name": "exclusive_random", "actors": poi, "stop": 17 + mislead}
    event_dict[17 + mislead] = {"name": "mislead", "actors": poi}
    event_dict[17 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi}
    return event_dict, second_loc, experiment_info

def sally_anne(n):
    event_dict = {}
    event_dict[4] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "same"}
    event_dict[5] = {"name":"move", "actors":['1'], "location": ['1']}
    event_dict[6] = {"name": "exclusive_random", "actors": ['0','1'], "stop": n}
    manual_actions_dict = {}
    obj = random.sample(["marble", "figurine", "doll"], 2)
    manual_actions_dict[4] = {'action': f'1 places a {obj[0]} in the basket'}
    manual_actions_dict[5] = {'action':f'0 empties the basket and places a {obj[1]} inside'}
    # experiment_info = {'cross path location': ['0'], 'poi':['0','1'], "obj": obj[0]}
    return event_dict, manual_actions_dict, "2", obj[0]
# TODO: manual_actions?