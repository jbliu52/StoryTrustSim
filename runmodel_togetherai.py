from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storysim import StorySimulator
from storyboard import Storyboard
from experiment_defs import *
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
    event_dict, max_actor = second_order_tom_experiment(3, story_length)
    storyboard = Storyboard('enters', graph, possible_people[:num_people], story_length, event_dict)

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

    p1 = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor)+1)])
    p1 = p1.lower()
    p2 = storyboard.actor_mapping[max_actor]
    p2 = p2.lower()
    label = storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]]
    example = (story.replace("_", ' '), p1, p2, label)
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

    random.seed(25)

    for _ in range(num_trials):
        event_dict, max_actor = second_order_tom_experiment(3, story_length)
        storyboard = Storyboard('enters', graph, possible_people[:num_people], story_length, event_dict)

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            action="enters",
            storyboard=storyboard,
            graph=graph
        )

        res = sim.run_simulation(story_length)

        story = sim.formal_to_story(res)
        d = {'Story':[], 'Tom_Label':[], 'WM_Label':[], 'P1':[], 'P2':[]}
        d['P1'] = [','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor)+1)])]
        d['P2'] = [storyboard.actor_mapping[max_actor]]
        d['Story'].append(story)
        # Add Tom and WM labels
        d['Tom_Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
        d['WM_Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
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
                p_ex = ex[1].lower().split(','), ex[2].lower()
                human_story = ex[0]
                for x in object_to_human.items():
                    human_story = human_story.replace(x[0], x[1]).replace('is moved to', 'moves to')
                formatted_tom.append(f"{human_story}\nQ: Where does {object_to_human_lower[p_ex[0][0]]} think {object_to_human_lower[p_ex[0][1]]} thinks {object_to_human_lower[p_ex[1]]} is?\nA: {ex[3]}")
                formatted_wm_humans.append(f"{human_story}\nQ: When {object_to_human_lower[p_ex[0][0]]} and {object_to_human_lower[p_ex[0][1]]} were in the same room as {object_to_human_lower[p_ex[1]]}, where did {object_to_human_lower[ex[2]]} go?\nA: {ex[3]}")
                formatted_wm_objects.append(f"{ex[0]}\nQ: When {p_ex[0][0]} and {p_ex[0][0]} were in the same room as {p_ex[1]}, where did {p_ex[1]} go?\nA: {ex[3]}")
                
            # Join fewshot examples together for prompting
            fewshot_string_tom, fewshot_string_wmh, fewshot_string_wmo = "\n".join(formatted_tom), "\n".join(formatted_wm_humans), "\n".join(formatted_wm_objects)
            
            mapped_story = row['Story'].replace('is moved to', 'moves to')
            for x in object_to_human.items():
                mapped_story = mapped_story.replace(x[0], x[1])
            
            # Prompt the model

            tom_prompt = f"{intial_prompt}\n{fewshot_string_tom}\n{mapped_story.replace('_',' ')}.\nQ: Where does {object_to_human_lower[p[0].lower()]} think {object_to_human_lower[p[1].lower()]} thinks {object_to_human_lower[row['P2'].lower()]} is?\nA:"
            wm_prompt_1 = f"{intial_prompt}\n{fewshot_string_wmh}\n{mapped_story.replace('_',' ')}.\nQ: When {object_to_human_lower[p[0].lower()]} and {object_to_human_lower[p[1]]} were in the same room as {object_to_human_lower[row['P2'].lower()]}, where did {object_to_human_lower[row['P2']]} go?\nA:"
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
            tom_prompt = f"{mapped_prefix.replace('is moved to', 'moves to')}Q: Where does {object_to_human_lower[p[0].lower()]} think {object_to_human_lower[p[1]]} thinks {object_to_human_lower[row['P2']]} is?\nA:"
            wm_prompt_1 = f"{mapped_prefix.replace('is moved to', 'moves to')}Q: When {object_to_human_lower[p[0].lower()]} and {object_to_human_lower[p[1]]} were in the same room as {object_to_human_lower[row['P2']]}, where did {object_to_human_lower[row['P2']]} go?\nA:"
            wm_prompt_2 = f"{prompt_prefix}Q: When {p[0]} and {p[1]} were in the same room as {row['P2']}, where was {row['P2']} moved to?\nA:"
            # Keep track of tokens
            tom_total += len(tom_prompt)
            wm_total += len(wm_prompt_1) + len(wm_prompt_2)
            
            # Prompt model
            tom_answer = prompt_model(tom_prompt, model_choice)
            wm_answer_h = prompt_model(wm_prompt_1, model_choice)
            wm_answer_o = prompt_model(wm_prompt_2, model_choice)

        tom_responses.append(tom_answer)
        wm_responses_1.append(wm_answer_h)
        wm_responses_2.append(wm_answer_o)        
    df['TOM Responses'] = tom_responses
    df['WM Responses Humans'] = wm_responses_1
    df['WM Responses Objects'] = wm_responses_2


    # Cost analysis using provided deepseek pricing
    if 'deepseek' in m:
        input_cost = (0.3 * (tom_total + wm_total) / 1000000) * 3
        output_cost = (0.3 * (sum(df["TOM Responses"].apply(lambda x: len(x))) + sum(df["WM Responses Humans"].apply(lambda x: len(x))) + sum(df["WM Responses Objects"].apply(lambda x: len(x)))) / 1000000) * 7
        print(f'Input cost: {input_cost}')
        print(f'Output cost: {output_cost}')
        print(f'Total cost: {input_cost+output_cost}')
    else:
        print(f"Total input words(approximate tokens): {tom_total + wm_total}")

    #print(f"triplets_results_fewshot/{m.split('/')[-1].replace('.', '_')}_triplets_100.csv")
    #df.to_csv(f"triplets_results_fewshot/{m.split('/')[-1].replace('.', '_')}_triplets_100.csv", index=False) 