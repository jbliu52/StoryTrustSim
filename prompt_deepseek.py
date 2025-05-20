from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storysim import StorySimulator
import pandas as pd
from together import Together

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_KEY')


# This example is the new way to use the OpenAI lib for python


def prompt_gpt(prompt, model):
    if 'gpt' in model:
        client = OpenAI()
    else:
        client = Together()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content


def mislead_experiment(actors, locs, g, mislead, length):
    poi = random.sample(actors, 2)
    loc = random.sample(locs, 1)
    second_loc = random.choice(g[loc[0]])
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
    event_dict[11] = {"name":"move", "actor":poi[-1], "location": second_loc}
    event_dict[12] = {"name": "exclusive_random", "actors": poi, "stop": 12 + mislead}
    event_dict[12 + mislead] = {"name": "mislead", "actors": poi}
    event_dict[12 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi}
    return event_dict, second_loc, experiment_info

def spaced_mislead_experiment(actors, locs, g, mislead, length):
    event_dict = {}
    poi = random.sample(actors, 2)
    loc = random.sample(locs, 1)
    label = random.choice([l for l in g[loc[-1]] if l != loc[-1]])
    event_dict[15] = {"name": "cross_paths","actors": poi, "location": loc}
    event_dict[16] = {"name": "exclusive_random", "actors": poi, "stop": 17}
    event_dict[17] = {"name":"move", "actor":poi[-1],"location":label}
    event_dict[18] = {"name": "exclusive_random", "actors": poi, "stop": 18+mislead}
    event_dict[18+mislead] = {"name": "mislead", "actors": poi}
    event_dict[18 + mislead +1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi}
    return event_dict, label, experiment_info

def number_of_moves_experiment(actors, locs, g, length):
    poi = random.sample(actors, 2)
    loc = random.sample(locs,1)
    num_moves = 0
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc, "path_type":"same"}
    prev = loc[0]
    movement = []
    for i in range(1,num_moves+1):
        new_loc = random.choice([l for l in g[prev] if l not in loc])
        movement.append(new_loc)
        event_dict[10+i] = {"name":"move", "actor":poi[-1], "location": new_loc}
        prev = new_loc
    #event_dict[10+num_moves+1] = {"name": "mislead", "actors": poi}
    label =  movement[0]
    event_dict[10+num_moves+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi}
    return event_dict, label, experiment_info

def second_order_tom_experiment(actors, locs, g, length):
    poi = random.sample(actors, 3)
    loc_1 = random.sample(locs,1)
    loc_2 = random.sample(g[loc_1[0]], 1)
    loc_3 = random.sample([l for l in g[loc_2[0]] if l != loc_1[0]], 1)
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc_1, "path_type":"same"}
    event_dict[20] = {"name": "cross_paths","actors": poi[1:], "location": loc_2, "path_type":"same"}
    event_dict[21] = {"name":"move", "actor":poi[-1],"location":loc_3[0]}
    event_dict[22] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc_1[0], 'poi':poi}
    return event_dict, loc_2, experiment_info
    

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

def write_on_wall_then_erase(actors, locs, g, mislead, length):
    poi = random.sample(actors, 3)
    loc_1 = random.sample(locs,1)
    loc_2 = random.sample(g[loc_1[0]], 1)
    loc_3 = random.sample([l for l in g[loc_2[0]] if l != loc_1[0]], 1)
    event_dict = {}
    actions_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc_1, "path_type":"same"}
    actions_dict[10] = {'actor':poi[-1], 'action': 'draws a circle on the wall'}
    event_dict[20] = {"name": "cross_paths","actors": poi[1:], "location": loc_2, "path_type":"same", "prev": loc_1[0]}
    actions_dict[20] = {'actor':poi[-1], 'action': 'draws a circle on the wall'}
    event_dict[21] = {"name":"move", "actor":poi[-1],"location":loc_3[0]}
    event_dict[22] = {"name": "exclusive_random", "actors": poi, "stop": length}
    actions_dict[length-2] = {'actor':poi[-1], 'action': 'draws a circle on the wall'}
    experiment_info = {'cross path location': loc_1[0], 'poi':poi, "draw":[11, 22, length-2]}
    return event_dict, actions_dict, loc_2, experiment_info

def placed_objects(actors, locs, g, mislead, length):
    poi = random.sample(actors, 3)
    loc_1 = random.sample(locs,1)
    loc_2 = random.sample(g[loc_1[0]], 1)
    loc_3 = random.sample([l for l in g[loc_2[0]] if l != loc_1[0]], 1)
    event_dict = {}
    actions_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc_1, "path_type":"same"}
    actions_dict[10] = {'actor':poi[-1], 'action': 'draws a circle on the wall'}
    event_dict[20] = {"name": "cross_paths","actors": poi[1:], "location": loc_2, "path_type":"same", "prev": loc_1[0]}
    actions_dict[20] = {'actor':poi[-1], 'action': 'puts their phone on the ground'}
    event_dict[21] = {"name":"move", "actor":poi[-1],"location":loc_3[0]}
    event_dict[22] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc_1[0], 'poi':poi, "draw":[11, 22, length-2]}
    return event_dict, actions_dict, "No", experiment_info
    
def second_order_tom_extension(actors, locs, g, length, actions_done=False):
    poi = random.sample(actors, 3)
    loc_1 = random.sample(locs,1)
    loc_2 = random.sample(g[loc_1[0]], 1)
    loc_3 = random.sample([l for l in g[loc_2[0]] if l != loc_1[0]], 1)
    event_dict = {}
    actions_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc_1, "path_type":"same"}
    event_dict[20] = {"name": "cross_paths","actors": poi[1:], "location": loc_2, "path_type":"same", "prev": loc_1[0]}
    event_dict[21] = {"name":"move", "actor":poi[-1],"location":loc_3[0]}
    event_dict[22] = {"name": "exclusive_random", "actors": poi, "stop": length}
    
    shape = None
    if actions_done:
        shape = random.choice(['circle', 'square', 'triangle', 'oval', 'star'])
        actions_dict[10] = {'actor':poi[-1], 'action': f'draws a {shape} on the wall'}
        actions_dict[20] = {'actor':poi[-1], 'action': 'puts their phone on the ground'}

    experiment_info = {'cross path location': loc_1[0], 'poi':poi, "shape":shape}
    
    return event_dict, actions_dict, (loc_2[0], loc_3[0]), experiment_info




df = pd.DataFrame({'Story':[], 'Tom_Label':[], 'P1':[], 'P2':[], 'WM_Label':[]})
possible_people = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia", "Hank", "Isaac", "Jake", "Kevin"]
num_people = 6
# graph = {
#         "hole_1": ["hole_2", "field"],
#         "hole_2": ["hole_1", "hole_3"],
#         "hole_3": ["hole_2", "hole_4"],
#         "hole_4": ["hole_3", "field"],
#         "field": ["hole_1", "hole_4"]
#     }


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

graph = { 
    "room_1": ["room_2", "the_hallway","room_5"],
    "room_2": ["room_1", "room_3","the_hallway"],
    "room_3": ["room_2", "room_4","the_hallway"],
    "room_4": ["room_3", "room_5","room_1"],
    "room_5": ["room_4", "room_1","room_2"],
    "the_hallway": ["room_1", "room_4","room_2"]
}

locations = list(graph.keys())
story_length = 30
num_trials = 100
mislead_distance = 3

random.seed(25)

for _ in range(num_trials):
    event_dict, action_dict, labels, experiment_dict = second_order_tom_extension(possible_people[:num_people],
                                                                                  locations[:-1], graph, story_length, actions_done=True)
   

    sim = StorySimulator(
        people=possible_people[:num_people],
        locations=locations,
        relation="enters",
        params={'prompt': '3', 'type': 'cot'},
        graph=graph,
        events=event_dict,
        actions=action_dict
    )
    res = sim.run_simulation(story_length)
    
    story = sim.formal_to_story(res)
    
    #split_story.insert(story_length-1, f" {experiment_dict['poi'][1]} forgets what they last saw on the wall")
    # print(".\n".join(story))

    d = {'Story':[], 'Tom_Label':[], 'P1':[], 'P2':[], 'WM_Label':[]}
    d['P1'] = ",".join(experiment_dict['poi'][:-1]) if len(experiment_dict['poi'][:-1]) > 1 else experiment_dict['poi'][:-1]
    #print(d["P1"])
    d['P2'] = experiment_dict['poi'][-1]
    d['Story'].append(story)
    # d['Label'].append(movement[0])
    d['Tom_Label'].append(labels[0])
    d['WM_Label'].append(labels[0])
    df = pd.concat([df, pd.DataFrame(d)])
    
    print(df.head())
    
    
intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother."
tom_responses, wm_responses = [], []
fewshot = False
model_choice = "deepseek-ai/DeepSeek-R1"
tom_total, wm_total = 0, 0
print('Beginning prompts')
for _ ,row in df.iterrows():
    p = row['P1'].split(',')
    formatted = []
    tom_answer = ''
    wm_answer = ''
    if fewshot:
        #formatted = [f'{ex[0]}\nQ: Where does {ex[1][0]} think {ex[1][1]} thinks {ex[1][2]} is?\nA: {ex[2]}' for ex in fewshot_examples]
        formatted = [f'{ex[0]}\nQ: Where does {ex[1][0]} think {ex[1][1]} is?\nA: {ex[2]}' for ex in []]
        formatted = '\n'.join(formatted)
        prompt = f"{intial_prompt}\n{formatted}\n{row['Story']}\nQ: Where does {p[0]} think {row['P2']} is?\nA:"
        #prompt = f"{intial_prompt}\n{formatted}\n{row['Story']}\nQ: Who has seen more drawings, {p[0]} or {row['P2']}?\nA:"
        print(prompt)
        print('*****')
        #answer = prompt_gpt(prompt, model_choice)
    else:
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        # Where does person 0 think the target is
        tom_prompt = f"{prompt_prefix}Q: Where does {p[0]} think {p[1]} thinks {row['P2']} is?\nA:"
        wm_prompt = f"{prompt_prefix}Q: When {p[0]} and {p[1]} were in the same room as {row['P2']}, where did {row['P2']} go?\nA:"
        tom_total += len(tom_prompt)
        wm_total += len(wm_prompt)
        tom_answer = prompt_gpt(tom_prompt, model_choice)
        wm_answer = prompt_gpt(wm_prompt, model_choice)
        #print(f'{tom_prompt}\n\n{wm_prompt}')
        #print("=====")
        #answer = prompt_gpt(prompt, model_choice)
    tom_responses.append(tom_answer)
    wm_responses.append(wm_answer)        
df['TOM Responses'] = tom_responses
df['WM Responses'] = wm_responses

input_cost = (0.3 * (tom_total + wm_total) / 1000000) * 3
output_cost = (0.3 * (sum(df["TOM Responses"].apply(lambda x: len(x))) + sum(df["WM Responses"].apply(lambda x: len(x)))) / 1000000) * 7
print(f'Input cost: {input_cost}')

print(f'Output cost: {output_cost}')
print(f'Total cost: {input_cost+output_cost}')

df.to_csv("saved_results/deepseek_r1_second_order_tom_q1q3_100.csv", index=False) 