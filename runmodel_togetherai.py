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




model_list = ["gpt-4",
              "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
              "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
              "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
              "meta-llama/Llama-3.2-3B-Instruct-Turbo",
              "meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "Qwen/QwQ-32B-Preview",
              "deepseek-ai/DeepSeek-R1"]

'''
Few shot set up
'''

num_fewshot = 3
possible_people = ["The shoe", "The ball", "The hat", "The apple", "The chess piece", "The bottle", "The flashlight", "The cup"]
human_names = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia"]
num_people = 6

object_to_human = {possible_people[i]:human_names[i] for i in range(num_people)}
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
mislead_distance = 3
fewshot_examples = []
random.seed(47)

for _ in range(num_fewshot):
    event_dict, action_dict, labels, experiment_dict = second_order_tom_extension(possible_people[:num_people], locations[:-1], graph, story_length, actions_done=False)
   

    sim = StorySimulator(
        people=possible_people[:num_people],
        locations=locations,
        relation="is_moved_to",
        params={'prompt': '3', 'type': 'cot'},
        graph=graph,
        events=event_dict,
        actions=action_dict
    )
    res = sim.run_simulation(story_length)
    
    story = sim.formal_to_story(res)
    

    p1 = ",".join(experiment_dict['poi'][:-1]) if len(experiment_dict['poi'][:-1]) > 1 else experiment_dict['poi'][:-1]
    p1 = p1.lower()
    p2 = experiment_dict['poi'][-1]
    p2 = p2.lower()
    example = (story.replace("_", ' '), p1, p2, labels[0].replace("_", ' '))
    fewshot_examples.append(example)
print(fewshot_examples)


'''
Running the experiment
'''
for m in model_list[1:4]:
    df = pd.DataFrame({'Story':[], 'Tom_Label':[], 'P1':[], 'P2':[], 'WM_Label':[]})
    possible_people = ["The shoe", "The ball", "The hat", "The apple", "The chess piece", "The bottle", "The flashlight", "The cup"]
    human_names = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia"]
    num_people = 6

    object_to_human = {possible_people[i]:human_names[i] for i in range(num_people)}
    object_to_human_lower = {possible_people[i].lower():human_names[i] for i in range(num_people)}

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
        event_dict, action_dict, labels, experiment_dict = second_order_tom_extension(possible_people[:num_people], locations[:-1], graph, story_length, actions_done=False)
    

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            relation="is_moved_to",
            params={'prompt': '3', 'type': 'cot'},
            graph=graph,
            events=event_dict,
            actions=action_dict
        )
        
        res = sim.run_simulation(story_length)
        story = sim.formal_to_story(res)
        
        d = {'Story':[], 'Tom_Label':[], 'P1':[], 'P2':[], 'WM_Label':[]}
        d['P1'] = ",".join(experiment_dict['poi'][:-1]) if len(experiment_dict['poi'][:-1]) > 1 else experiment_dict['poi'][:-1]
        d["P1"] = d["P1"].lower()
        d['P2'] = experiment_dict['poi'][-1]
        d["P2"] = d["P2"].lower()
        d['Story'].append(story)
        d['Tom_Label'].append(labels[0].replace('_', ' '))
        d['WM_Label'].append(labels[0].replace('_', ' '))
        df = pd.concat([df, pd.DataFrame(d)])
    print(df.head())

    print(f"Beginning Prompts with {m}...")
    
    intial_prompt = f"Read the following story and answer the question at the end. Note that all subjects start in {sim.locations[-1].replace('_',' ')}. Subjects of the story that are in the same location can see where eachother go when someone leaves. If subjects are in different locations, they cannot see eachother. There is only one of each subject in the story. Output your final answer as one word at the end of your response."

    tom_responses, wm_responses_1, wm_responses_2 = [], [], []
    fewshot = True
    model_choice = m
    tom_total, wm_total = 0, 0
    for _ ,row in df.iterrows():
        p = row['P1'].split(',')
        formatted = []
        tom_answer, wm_answer_h, wm_answer_o = '', '', ''
        formatted_tom, formatted_wm_humans, formatted_wm_objects = [], [], []

        if fewshot:
            # Extract the story, subjects, and label for fewshot examples
            for ex in fewshot_examples:
                p_ex = ex[1].split(',')
                human_story = ex[0]
                for x in object_to_human.items():
                    human_story = human_story.replace(x[0], x[1]).replace('is moved to', 'moves to')
                formatted_tom.append(f"{human_story}\nQ: Where does {object_to_human_lower[p_ex[0]]} think {object_to_human_lower[p_ex[1]]} thinks {object_to_human_lower[ex[2]]} is?\nA: {ex[3]}")
                formatted_wm_humans.append(f"{human_story}\nQ: When {object_to_human_lower[p_ex[0]]} and {object_to_human_lower[p_ex[1]]} were in the same room as {object_to_human_lower[ex[2]]}, where did {object_to_human_lower[ex[2]]} go?\nA: {ex[3]}")
                formatted_wm_objects.append(f"{ex[0]}\nQ: When {p_ex[0]} and {p_ex[1]} were in the same room as {ex[2]}, where did {ex[2]} go?\nA: {ex[3]}")
                
            # Join fewshot examples together for prompting
            fewshot_string_tom, fewshot_string_wmh, fewshot_string_wmo = "\n".join(formatted_tom), "\n".join(formatted_wm_humans), "\n".join(formatted_wm_objects)
            
            mapped_story = row['Story'].replace('is moved to', 'moves to')
            for x in object_to_human.items():
                mapped_story = mapped_story.replace(x[0], x[1])
            
            # Prompt the model

            tom_prompt = f"{intial_prompt}\n{fewshot_string_tom}\n{mapped_story.replace('_',' ')}.\nQ: Where does {object_to_human_lower[p[0]]} think {object_to_human_lower[p[1]]} thinks {object_to_human_lower[row['P2']]} is?\nA:"
            wm_prompt_1 = f"{intial_prompt}\n{fewshot_string_wmh}\n{mapped_story.replace('_',' ')}.\nQ: When {object_to_human_lower[p[0]]} and {object_to_human_lower[p[1]]} were in the same room as {object_to_human_lower[row['P2']]}, where did {object_to_human_lower[row['P2']]} go?\nA:"
            wm_prompt_2 = f"{intial_prompt}\n{fewshot_string_wmo}\n{row['Story'].replace('_',' ')}.\nQ: When {p[0]} and {p[1]} were in the same room as {row['P2']}, where was {row['P2']} moved to?\nA:"

            # For tracking token amounts
            tom_total += len(tom_prompt)
            wm_total += len(wm_prompt_1) + len(wm_prompt_2)
            
            # Prompt model
            tom_answer = prompt_model(tom_prompt, model_choice)
            wm_answer_h = prompt_model(wm_prompt_1, model_choice)
            wm_answer_o = prompt_model(wm_prompt_2, model_choice)
            
            #print(f'{tom_prompt}\n\n{wm_prompt_1}\n\n{wm_prompt_2}')
            #print("=====")            
        else:
            prompt_prefix = f"{intial_prompt}\n{row['Story'].replace('_',' ')}.\n"
            # Where does person 0 think the target is
            mapped_prefix = prompt_prefix
            for x in object_to_human.items():
                mapped_prefix = mapped_prefix.replace(x[0], x[1])
            
            # Construct prompts
            tom_prompt = f"{mapped_prefix.replace('is moved to', 'moves to')}Q: Where does {object_to_human_lower[p[0]]} think {object_to_human_lower[p[1]]} thinks {object_to_human_lower[row['P2']]} is?\nA:"
            wm_prompt_1 = f"{mapped_prefix.replace('is moved to', 'moves to')}Q: When {object_to_human_lower[p[0]]} and {object_to_human_lower[p[1]]} were in the same room as {object_to_human_lower[row['P2']]}, where did {object_to_human_lower[row['P2']]} go?\nA:"
            wm_prompt_2 = f"{prompt_prefix}Q: When {p[0]} and {p[1]} were in the same room as {row['P2']}, where was {row['P2']} moved to?\nA:"
            # Keep track of tokens
            tom_total += len(tom_prompt)
            wm_total += len(wm_prompt_1) + len(wm_prompt_2)
            
            # Prompt model
            tom_answer = prompt_model(tom_prompt, model_choice)
            wm_answer_h = prompt_model(wm_prompt_1, model_choice)
            wm_answer_o = prompt_model(wm_prompt_2, model_choice)
            #print(f'{tom_prompt}\n\n{wm_prompt_1}\n\n{wm_prompt_2}')
            #print("=====")
            #answer = prompt_model(prompt, model_choice)
        tom_responses.append(tom_answer)
        wm_responses_1.append(wm_answer_h)
        wm_responses_2.append(wm_answer_o)        
    df['TOM Responses'] = tom_responses
    df['WM Responses Humans'] = wm_responses_1
    df['WM Responses Objects'] = wm_responses_2

    if 'deepseek' in m:
        input_cost = (0.3 * (tom_total + wm_total) / 1000000) * 3
        output_cost = (0.3 * (sum(df["TOM Responses"].apply(lambda x: len(x))) + sum(df["WM Responses Humans"].apply(lambda x: len(x))) + sum(df["WM Responses Objects"].apply(lambda x: len(x)))) / 1000000) * 7
        print(f'Input cost: {input_cost}')
        print(f'Output cost: {output_cost}')
        print(f'Total cost: {input_cost+output_cost}')
    else:
        print(f"Total input words(approximate tokens): {tom_total + wm_total}")

    print(f"triplets_results_fewshot/{m.split('/')[-1].replace('.', '_')}_triplets_100.csv")
    df.to_csv(f"triplets_results_fewshot/{m.split('/')[-1].replace('.', '_')}_triplets_100.csv", index=False) 