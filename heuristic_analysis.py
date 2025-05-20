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

def compute_score_unsure(label, response, starting_loc='hallway'):
    #label = label.split(',')[0][2:-1]
    base = f'{label.split("_")[0]}_' if not label.startswith(starting_loc) else label
    response = response.split("\n")[-1]
    response = response.lower().replace("_",' ')
    if response.count(base) <= 1:
        return str(label in response or label.replace('_'," ") in response or 
                   label.replace('_',"") in response or
                   (starting_loc
                    in label and starting_loc in response) or
                   label.replace(" ",'') in response)
    elif 'Therefore,' in response:
        response = response.split('Therefore,')[-1]
        return str(label in response or label.replace('_'," ") in response or label.replace('_',"") in response or (starting_loc in label and starting_loc in response))
    return f'{response}, {label}'

'''

Experiment defitions

'''

def mislead_experiment(actors, locs, g, mislead, length):
    poi = random.sample(actors, 2)
    loc = random.sample(locs, 1)
    second_loc = random.choice(g[loc[0]])
    third_loc = random.choice([l for l in g[second_loc] if l not in loc])
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
    event_dict[11] = {"name":"move", "actor":poi[-1], "location": second_loc}
    event_dict[12] = {"name": "exclusive_random", "actors": poi, "stop": 12 + mislead}
    event_dict[12 + mislead] = {"name":"move", "actor":poi[-1],"location":third_loc}
    event_dict[12 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi, 'last':third_loc}
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

def second_order_tom_experiment(actors, locs, g, mislead, length):
    poi = random.sample(actors, 3)
    loc_1 = random.sample(locs,1)
    loc_2 = random.choice(g[loc_1[0]])
    loc_3 = random.choice([l for l in g[loc_2] if l != loc_1[0]], )
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc_1, "path_type":"same"}
    event_dict[16] = {"name": "cross_paths","actors": poi[1:], "location": [loc_2], "path_type":"same", "prev":loc_1[0]}
    event_dict[17] = {"name": "exclusive_random", "actors": poi, "stop": 17 + mislead}
    event_dict[17 + mislead] = {"name":"move", "actor":poi[-1],"location":loc_3}
    event_dict[17 + mislead + 1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path locs': [loc_1, loc_2], 'poi': poi, 'last': loc_3}
    return event_dict, loc_2, experiment_info
    

def cross_path_overlap(actors, locs, g, mislead, length, n):
    poi = random.sample(actors, n)
    loc = random.sample(locs, 1)
    second_loc = random.choice(g[loc[0]])
    event_dict = {}
    event_dict[15] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
    event_dict[16] = {"name":"move", "actor":poi[-1], "location": second_loc}
    event_dict[17] = {"name": "exclusive_random", "actors": poi, "stop": 17 + mislead}
    event_dict[17 + mislead] = {"name": "mislead", "actor": poi}
    event_dict[17 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi}
    return event_dict, second_loc, experiment_info


'''
Actual Experiements
'''
# Num of people
#values = [3,10,20,40]

# Mislead
values = [5,10, 20, 30, 40, 50, 60, 70, 80]
#values = [30,40]
#values = values[-2:]

knowns, unkowns = [], []
possible_people = [
    "Alice", "Bob", "Charlie", "Danny", "Edward",
    "Frank", "Georgia", "Hank", "Isaac", "Jake",
    "Kevin", "Liam", "Mia", "Nina", "Oliver",
    "Paula", "Quinn", "Rachel", "Steve", "Tina",
    "Uma", "Victor", "Wendy", "Xander", "Yara",
    "Zane", "Amber", "Brandon", "Carmen", "Derek",
    "Elena", "Felix", "Grace", "Harvey", "Ivy",
    "Jasmine", "Kyle", "Leah", "Miles", "Naomi",
    "Henry", "Wyatt", "Jose", "Neil","Seth"
]


print('beginning')
print('Heurisitc: Num of people')

for v in range(len(values)):
    print(f'Mislead = {values[v]}')
    df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[], 'Last':[], 'CP_Loc':[]})
    num_people = 7
    
    # graph = { 
    #     "room_1": ["room_2", "the_hallway","room_5"],
    #     "room_2": ["room_1", "room_3","the_hallway"],
    #     "room_3": ["room_2", "room_4","the_hallway"],
    #     "room_4": ["room_3", "room_5","room_1"],
    #     "room_5": ["room_4", "room_1","room_2"],
    #     "the_hallway": ["room_1", "room_4","room_2"]
    # }

    graph = { 
        "hole_1": ["hole_2", "the_field","hole_5"],
        "hole_2": ["hole_1", "hole_3","the_field"],
        "hole_3": ["hole_2", "hole_4","the_field"],
        "hole_4": ["hole_3", "hole_5","hole_1"],
        "hole_5": ["hole_4", "hole_1","hole_2"],
        "the_field": ["hole_1", "hole_4","hole_2"]
    }


    locations = list(graph.keys())
    story_length = 100
    num_trials = 100
    mislead_distance = values[v]

    random.seed(25)

    for _ in range(num_trials):
        
        event_dict, label, experiment_dict = mislead_experiment(possible_people[:num_people], locations[:-1], graph, mislead_distance, story_length)

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            relation="enters",
            params={'prompt': '3', 'type': 'cot'}, 
            graph=graph,
            events=event_dict,
            actions={}
        )

        res = sim.run_simulation(story_length)

        story = sim.formal_to_story(res)
        d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
        d['P1'] = ",".join(experiment_dict['poi'][:-1]) if len(experiment_dict['poi'][:-1]) > 1 else experiment_dict['poi'][:-1]
        #print(d["P1"])
        d['P2'] = experiment_dict['poi'][-1]
        d['Story'].append(story)
        # d['Label'].append(movement[0])
        d['Label'].append(label)
        d['Last'] = experiment_dict['last']
        d['CP_Loc'] = experiment_dict.get('cross paths locs', [None, None])[-1]
        df = pd.concat([df, pd.DataFrame(d)])   
        
    # Prompt model
    
        
    intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother."
    tom_responses, wm_responses = [], []
    #model_choice =  "deepseek-ai/DeepSeek-R1"
    model_choice =  "gpt-4"
    tom_total, wm_total = 0, 0
    for _ ,row in df.iterrows():
        p = row['P1'].split(',')[-1]
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        prompt = f'{prompt_prefix}Q: Where does {p} think {row["P2"]} is?\nA:'
        answer = prompt_model(prompt, model_choice)
        tom_responses.append(answer)
    df['TOM Responses'] = tom_responses

    outs= df.apply(lambda x: compute_score_unsure(x['Label'], x['TOM Responses']), axis=1)
    known = [k for k in outs if k == 'True' or k == 'False']
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
    df.to_csv(f'mislead_second_order/holes_{model_choice.replace("/","_")}_{values[v]}.csv')
        
for i in range(len(knowns)):
    score = sum([1 for k in knowns[i] if k == 'True'])
    print(f'Mislead = {values[i]}: {score}/{num_trials}, {len(unkowns[i])} unkown')
        
        
    