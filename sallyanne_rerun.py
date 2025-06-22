from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storysim import StorySimulator
import pandas as pd
from together import Together
from experiment_defs import sally_anne
from storyboard import Storyboard
load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_KEY')


def prompt_model(prompt, model):
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

def compute_score_unsure(label, response):
    #label = label.split(',')[0][2:-1]
    base = f'{label.split("_")[0]}_' if not label.startswith('hallway') else label
    response = response.split("\n")[-1]
    response = response.lower().replace("_",' ')
    if response.count(base) <= 1:
        return str(label in response or label.replace('_'," ") in response or 
                   label.replace('_',"") in response or
                   ("hallway" in label and "hallway" in response) or
                   label.replace(" ",'') in response)
    elif 'Therefore,' in response:
        response = response.split('Therefore,')[-1]
        return str(label in response or label.replace('_'," ") in response or label.replace('_',"") in response or ("hallway" in label and "hallway" in response))
    return f'{response}, {label}'


'''

Experiment defitions

'''

# def mislead_experiment(actors, locs, g, mislead, length):
#     poi = random.sample(actors, 2)
#     loc = random.sample(locs, 1)
#     second_loc = random.choice(g[loc[0]])
#     third_loc = random.choice([l for l in g[loc[0]] if l not in loc])
#     event_dict = {}
#     event_dict[3] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
#     event_dict[4] = {"name":"move", "actor":poi[-1], "location": second_loc}
#     event_dict[5] = {"name": "exclusive_random", "actors": poi, "stop": 5 + mislead}
#     event_dict[5 + mislead] = {"name":"move", "actor":poi[-1],"location":third_loc}
#     event_dict[5 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
#     experiment_info = {'cross path location': loc[0], 'poi':poi, 'last':third_loc}
#     return event_dict, second_loc, experiment_info

# def spaced_mislead_experiment(actors, locs, g, mislead, length):
#     event_dict = {}
#     poi = random.sample(actors, 2)
#     loc = random.sample(locs, 1)
#     label = random.choice([l for l in g[loc[-1]] if l != loc[-1]])
#     event_dict[15] = {"name": "cross_paths","actors": poi, "location": loc}
#     event_dict[16] = {"name": "exclusive_random", "actors": poi, "stop": 17}
#     event_dict[17] = {"name":"move", "actor":poi[-1],"location":label}
#     event_dict[18] = {"name": "exclusive_random", "actors": poi, "stop": 18+mislead}
#     event_dict[18+mislead] = {"name": "mislead", "actors": poi}
#     event_dict[18 + mislead +1] = {"name": "exclusive_random", "actors": poi, "stop": length}
#     experiment_info = {'cross path location': loc[0], 'poi':poi}
#     return event_dict, label, experiment_info

# def number_of_moves_experiment(actors, locs, g, length):
#     poi = random.sample(actors, 2)
#     loc = random.sample(locs,1)
#     num_moves = 0
#     event_dict = {}
#     event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc, "path_type":"same"}
#     prev = loc[0]
#     movement = []
#     for i in range(1,num_moves+1):
#         new_loc = random.choice([l for l in g[prev] if l not in loc])
#         movement.append(new_loc)
#         event_dict[10+i] = {"name":"move", "actor":poi[-1], "location": new_loc}
#         prev = new_loc
#     #event_dict[10+num_moves+1] = {"name": "mislead", "actors": poi}
#     label =  movement[0]
#     event_dict[10+num_moves+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
#     experiment_info = {'cross path location': loc[0], 'poi':poi}
#     return event_dict, label, experiment_info

# def second_order_tom_experiment(actors, locs, g, length):
#     poi = random.sample(actors, 3)
#     loc_1 = random.sample(locs,1)
#     loc_2 = random.sample(g[loc_1[0]], 1)
#     loc_3 = random.sample([l for l in g[loc_2[0]] if l != loc_1[0]],1)
#     event_dict = {}
#     event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc_1, "path_type":"same"}
#     event_dict[16] = {"name": "cross_paths","actors": poi[1:], "location": loc_1, "path_type":"same"}
#     event_dict[17] = {"name":"move", "actor":poi[-1],"location":loc_3}
#     event_dict[18] = {"name": "exclusive_random", "actors": poi, "stop": length}
    

# def cross_path_overlap(actors, locs, g, mislead, length, n):
#     poi = random.sample(actors, n)
#     loc = random.sample(locs, 1)
#     second_loc = random.choice(g[loc[0]])
#     event_dict = {}
#     event_dict[15] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
#     event_dict[16] = {"name":"move", "actor":poi[-1], "location": second_loc}
#     event_dict[17] = {"name": "exclusive_random", "actors": poi, "stop": 17 + mislead}
#     event_dict[17 + mislead] = {"name": "mislead", "actors": poi}
#     event_dict[17 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
#     experiment_info = {'cross path location': loc[0], 'poi':poi}
#     return event_dict, second_loc, experiment_info

# def sally_anne(actors, locs, g, n):
#     poi = random.sample(actors, 2)
#     loc = random.sample(locs, 1)
#     second_loc = random.choice(g[loc[0]])
#     event_dict = {}
#     event_dict[4] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
#     event_dict[5] = {"name":"move", "actor":poi[-1], "location": second_loc}
#     event_dict[6] = {"name": "exclusive_random", "actors": poi, "stop": n}
#     manual_actions_dict = {}
#     obj = random.sample(["marble", "figurine", "doll"], 2)
#     manual_actions_dict[4] = {'actor':poi[-1], 'action': f'places a {obj[0]} in the basket'}
#     manual_actions_dict[5] = {'actor':poi[0], 'action':f'empties the basket and places a {obj[1]} inside'}
#     experiment_info = {'cross path location': loc[0], 'poi':poi, "obj": obj[0]}
#     return event_dict, second_loc, experiment_info, manual_actions_dict



'''
Actual Experiements
'''
# Num of people
# Mislead
# values = [3,5,10,50]

knowns, unkowns = [], []
possible_people = ["Sally", "Anne", "Charlie"]


print('beginning')
print('Sally Anne')

df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[]})
num_people = 3

graph = { 
    "room_1": ["room_2", "the_hallway"],
    "room_2": ["room_1", "the_hallway"],
    "the_hallway": ["room_1", "room_2"]
}


locations = list(graph.keys())
story_length = 7
num_trials = 2

random.seed(25)

for _ in range(num_trials):
    
    #event_dict, label, experiment_dict, manual_actions_dict = sally_anne(possible_people[:num_people], locations[:-1], graph, story_length)        
    event_dict, manual_action_dict, max_actor, label = sally_anne(story_length)
    storyboard = Storyboard('enters', graph, possible_people[:num_people], story_length, event_dict, manual_actions=manual_action_dict)
    
    sim = StorySimulator(
        people=possible_people[:num_people],
        locations=locations,
        action="enters",
        storyboard=storyboard,
        graph=graph
    )

    res = sim.run_simulation(story_length)



    story = sim.formal_to_story(res)
    d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
    # TODO: Actor mapping? How do you know who P1 and P2 are?
    d['P1'] = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])
    #print(d["P1"])
    d['P2'] = storyboard.actor_mapping[str(int(max_actor))]
    d['Story'].append(story)
    d['Label'].append(label)
    #d['Label'].append(max(storyboard.loc_mapping.keys()))
    #d['Last'] = experiment_dict['obj']
    df = pd.concat([df, pd.DataFrame(d)])   
    
# Prompt model
    
intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother."
tom_responses, wm_responses = [], []
model_choice =  "gpt-4"
tom_total, wm_total = 0, 0
for _ ,row in df.iterrows():
    p = row['P1'].split(',')
    prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
    prompt = f'{prompt_prefix}Q:What does {row["P1"]} think {row["P2"]} think is in the basket?\nA:'
    answer = prompt_model(prompt, model_choice)
    tom_responses.append(answer)
df['TOM Responses'] = tom_responses

outs= df.apply(lambda x: compute_score_unsure(x['Label'], x['TOM Responses']), axis=1)
known = [k for k in outs if k == 'True' or k == 'False']
print(sum([1 for k in known if k == 'True']))
knowns.append(known)
unknown = [k for k in outs if k != 'True' and k != 'False']
unkowns.append(unknown)
print('UKNOWNS')
for i in range(len(outs)):
    if outs.iloc[i] != 'True' and outs.iloc[i] != 'False':
        clean_out = outs.iloc[i].split('<think>')[-1]
        print(f'{clean_out}')
        print(f'Index {i}')
        print("\n-////==============////-\n")
print(len(unknown))
#df.to_csv(f'sally_anne/{model_choice.replace("/","_")}.csv')
    
    
    
    