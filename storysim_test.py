from storyactor import Actor
from storysim import StorySimulator
from storyboard import Storyboard
import random

possible_people = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia", "Hank", "Isaac", "Jake", "Kevin"]
num_people = 3

possible_actors = []

for person in possible_people: possible_actors.append(Actor(person, []))
for actor in possible_actors: actor.set_actors(possible_actors)
# graph = {
#         "hole_1": ["hole_2", "field"],
#         "hole_2": ["hole_1", "hole_3"],
#         "hole_3": ["hole_2", "hole_4"],
#         "hole_4": ["hole_3", "field"],
#         "field": ["hole_1", "hole_4"]
#     }

# graph = { 
#     "hole_1": ["hole_2", "the_field","hole_5"],
#     "hole_2": ["hole_1", "hole_3","the_field"],
#     "hole_3": ["hole_2", "hole_4","the_field"],
#     "hole_4": ["hole_3", "hole_5","hole_1"],
#     "hole_5": ["hole_4", "hole_1","hole_2"],
#     "the_field": ["hole_1", "hole_4","hole_2"]
# }

graph = {
    "the_park": ["the_cafe", "the_field", 'the_store'],
    "the_cafe": ["the_park", "the_field", "the_cafe"],
    'the_field': ["the_park", "the_cafe", "the_store"],
    'the_store': ["the_park", "the_cafe", "the_field"]
}

# graph = { 
#     "hole_1": ["hole_2", "field", "hole_5", "hole_6", "hole_4"],
#     "hole_2": ["hole_1", "hole_3", "field", "hole_5", "hole_7"],
#     "hole_3": ["hole_2", "hole_4", "field", "hole_8", "hole_1"],
#     "hole_4": ["hole_3", "hole_5", "hole_1", "hole_9", "field"],
#     "hole_5": ["hole_4", "hole_1", "hole_2", "hole_10", "field"],
#     "hole_6": ["hole_1", "hole_7", "hole_9", "field"],
#     "hole_7": ["hole_6", "hole_2", "hole_8", "field"],
#     "hole_8": ["hole_7", "hole_3", "hole_9", "field"],
#     "hole_9": ["hole_8", "hole_4", "hole_6", "hole_10", "field"],
#     "hole_10": ["hole_9", "hole_5", "field", "hole_7"],
#     "field": ["hole_1", "hole_2", "hole_3", "hole_4", "hole_9", "hole_10"]
# }

locations = list(graph.keys())
story_length = 16
num_trials = 3
mislead = 3

random.seed(25)

# for _ in range(num_trials):
#     event_dict = {}
    
#     event_dict = {}
#     event_dict[10] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "same"}
#     event_dict[11] = {"name":"move", "actors":['1'], "location": ['1']}
#     event_dict[12] = {"name": "exclusive_random", "actors": ['0','1'], "stop": 12 + mislead}
#     event_dict[12 + mislead] = {"name":"move", "actors":['1'],"location":['2']}
#     event_dict[12 + mislead+1] = {"name": "exclusive_random", "actors": ['0',"1"], "stop": story_length}
    
#     storyboard = Storyboard('enters', graph, possible_people[:num_people], story_length, event_dict)

#     sim = StorySimulator(
#         people=possible_people[:num_people],
#         locations=locations,
#         action=["enters", "widens", "illuminates", "deepens"],
#         params={'prompt': '3', 'type': 'cot'},
#         storyboard=storyboard,
#         graph=graph
#     )
    
#     res = sim.run_simulation(story_length)
#     story = sim.formal_to_story(res)
#     print("\n".join(story.split(".")))
#     print('-----')

    
for _ in range(num_trials):

    event_dict = dict()
    manual_action_dict = dict()
    
    event_dict[1] = {"name":"move", "actors":['0'],"location":['0']}
    event_dict[5] = {"name": "cross_paths","actors": ['0','1'], "location": ['1'], "path_type": "same"}
    manual_action_dict[5] = {'action': '0 and 1 exchange phone numbers'}
    manual_action_dict[7] = {'action': '0 asks 1 for aid'}
    event_dict[9] = {"name":"cross_paths","actors": ['0','1'], "location": ['2'], "path_type": "same"}
    manual_action_dict[9] = {'action': '0 calls their friends on the phone to tell them what happened'}
    storyboard = Storyboard('goes_to', graph, possible_people[:num_people], story_length, event_dict, manual_actions=manual_action_dict)
    
    sim = StorySimulator(
        people=possible_actors[:num_people],
        locations=locations,
        action=['goes_to', 'arrives_at', 'heads_to'],
        # params={},
        storyboard=storyboard,
        graph=graph
    )
    
    res = sim.run_simulation(story_length)
    story = sim.formal_to_story(res)
    print("\n".join(story.split(".")))
    print('-----')
