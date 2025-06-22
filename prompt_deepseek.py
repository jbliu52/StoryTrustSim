from dotenv import load_dotenv
import os
import random
from openai import OpenAI
from storysim import StorySimulator
from storyboard import Storyboard
from experiment_defs import second_order_tom_experiment
import pandas as pd
from together import Together

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_KEY')

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


df = pd.DataFrame({'Story':[], 'Tom_Label':[], 'P1':[], 'P2':[], 'WM_Label':[]})
possible_people = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia", "Hank", "Isaac", "Jake", "Kevin"]
num_people = 6

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
    storyboard = Storyboard("enters", graph, possible_people[:num_people], story_length, event_dict)
   

    sim = StorySimulator(
        people=possible_people[:num_people],
        locations=locations,
        action="enters",
        storyboard=storyboard,
        graph=graph
    )
    res = sim.run_simulation(story_length)
    
    story = sim.formal_to_story(res)
    
    #split_story.insert(story_length-1, f" {experiment_dict['poi'][1]} forgets what they last saw on the wall")
    # print(".\n".join(story))

    d = {'Story':[], 'Tom_Label':[], 'P1':[], 'P2':[], 'WM_Label':[]}
    d['P1'].append(",".join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor)+1)]))
    d['P2'].append(storyboard.actor_mapping[max_actor])
    d['Story'].append(story)
    d['Tom_Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
    d['WM_Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
    df = pd.concat([df, pd.DataFrame(d)])    
    
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