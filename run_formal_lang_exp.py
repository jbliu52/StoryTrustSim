from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import numpy as np
import regex as re
import pandas as pd
import random
from scipy import stats

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

people = ['Alice', 'Bob', 'Joe', 'Dylan', 'Josh', 'Abu', 'Nate', 'Dom'] # mind reading
people = people[:5]
# Last location is initial one
locations = ['hole_1', 'hole_2', 'hole_3', 'hole_4','field'] # complicate the environment?
relation = 'jumps_in' # more actions/relations?
# goal directedness

      
#current_locations = {p:[locations[-1]] for p in people}

# Update the state dict so that current_locations[person][t] is the person's location at time step t
def update_state(subject, new_loc, current_locations):
    for p in people:
        if p == subject:
            current_locations[p].append(new_loc)
        else:
            current_locations[p].append(current_locations[p][-1])

print('All start at ' + locations[-1])

def run_sim(steps):
    
    person = ''
    loc = ''
    sequences = ''
    #global current_locations
    current_locations = {p:[locations[-1]] for p in people}
    for t in range(steps):
        person = random.choice(people)
        cur_loc = current_locations[person]
        if locations[0][:-2] in cur_loc[-1]:
            loc = locations[-1]
        else:
            loc = random.choice(locations[:-1])
        update_state(person, loc, current_locations)
        sequences += f'{relation}({person}, {loc}, {t})\n'    
        #print(f'{relation}({person}, {loc}, {t})') 
    return sequences, current_locations


# At timestep t where did subject think target was?
# Answer is wherever the subject saw target at during t or prior
# If they were in the same place at exactly t, then thats where subject knows target is
# If < t they were in the same place and target moved, then its where the target moved
# If < t they were in the same place and subject moved or both moved, its where target remains
def last_observed_change(subject, target, t, current_locations):
    target_history = current_locations[target]
    subject_history = current_locations[subject]
    #check first case
    if target_history[t] == subject_history[t]:
        return target_history[t], t
    
    # Check the second two cases
    prev = 0
    for t_i in range(t,-1,-1):
        if target_history[t_i] == subject_history[t_i]:
            prev = t_i
    if subject_history[prev+1] == subject_history[prev+1] and target_history[prev+1] != target_history[prev+1]:
        return target_history[prev+1], prev+1
    else:
        return target_history[prev], prev
   
    
def print_sequences(sequences, relation):
    # Translate to text
    s = sequences.split('\n')[:-1]
    strings = []
    for event in s:
        event = event.replace(f'{relation}(','').replace(')','')
        subject, loc, _ = event.split(',')
        # Temporal?
        #print(f'{subject} enters{loc} at time{t}')
        res = f'{subject} {relation.replace("_", " ") if "hole" in loc else relation.replace("_", " ").replace("in","out to the") }{loc}'
        #print(res)
        strings.append(res)
    return '. '.join(strings)            
            
# Question event, who/multiple ppl who observed an event

 
def find_question_events(t, current_locations):
    observe_matrix = []
    for p1 in people:
        observe_matrix.append([])
        for p2 in people:
            if p1 != p2:
                loc, time = last_observed_change(p1,p2,t, current_locations)
                observe_matrix[-1].append((loc, time))
            else:
                observe_matrix[-1].append(None)
    return observe_matrix


def make_temporal_tracking_map(steps, current_locations):
    tracking_map = [{'field':set(people)}]
    for t in range(1,steps):
        location_to_ppl = dict()
        for p in people:
            curr_loc = current_locations[p][t]
            if curr_loc in location_to_ppl:
                location_to_ppl[curr_loc].add(p)
            else:
                location_to_ppl[curr_loc] = {p}
        tracking_map.append(location_to_ppl)
    return tracking_map


def find_observance_event(temporal_map, target, current_locations):
    target_locations = current_locations[target]
    observers = set()
    start = 0
    res = []
    for t in range(len(temporal_map)-1):
        if target_locations[t] != target_locations[t+1]:
            if len(observers) == 0:
                observers = temporal_map[t][target_locations[t]].difference({target})
                start = t
            # Find next location change
            else:
                new_loc = target_locations[t+1]
                if temporal_map[t+1][new_loc].intersection(observers) != len(observers):
                    # People who see target coming to the new location
                    observe_incoming = temporal_map[t+1][new_loc]
                    # People who see target leaving the old location
                    observe_outgoing = temporal_map[t][target_locations[t]]
                    unaware_subjects = observers.difference(observe_incoming).difference(observe_outgoing)
                    res.append((unaware_subjects,start, t+1))
                    start = 0
                observers = set()
    return res


client = OpenAI()
initial_prompt = f'Read the following story and answer the question at the end. Note that all characters start in the {locations[-1]}.'
def prompt_gpt(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content

''' 
Few shot generation code
'''
def split__into_parts(ex):
    parts = []
    try:
        # Split by "Q:"
        pre_question, question_and_answer = ex.split("Q:", 1)
        parts.append(pre_question.strip())

        # Split the remaining part by "A:"
        question, answer = question_and_answer.split("A:", 1)
        parts.append("Q: " + question.strip())
        parts.append("A: " + answer.strip())
    except ValueError:
        print("The story doesn't have the expected format with 'Q:' and 'A:'.")
    
    return parts

def make_fewshot_examples(num_examples, cot, length_thresh):
    example_string = ''
    #random.seed(47)
    n_steps_fewshot = 40
    # TODO: Synthetically generate code
    if cot:
        with open('few_shot_cot.txt','r') as f:
            content = f.read()
            examples = content.split('---')
            examples =  [e.strip() for e in examples]
            
            for e in examples:
                parts = split__into_parts(e)
                example_string += f'{"".join(parts[0])}\n {parts[1]}\n {parts[2]}\n\n;'
            return example_string
    else:    
        # --------
        for _ in range(num_examples):
            print(f'Generating {num_examples} training examples')
            # Find places to ask questions.
            while(True):
                # Generate experiment story
                test_sims_test, current_locations_test = run_sim(n_steps_fewshot)
                story_test = print_sequences(test_sims_test, relation)
                s_test = story_test.split('.')
                tm_test_fewshot = make_temporal_tracking_map(n_steps_fewshot, current_locations_test)
                # Find places to ask questions.
                new_order = random.sample(people, len(people))
                length_found = False
                for p in new_order:
                    obs_events_fs = find_observance_event(tm_test_fewshot, p, current_locations_test)
                    # Filter out the ones that are 
                    obs_events_fs = [e for e in obs_events_fs if len(e[0]) != 0 and e[2]-1 >= length_thresh]
                    if len(obs_events_fs) == 0:
                        continue
                    length_found = True
                    oe_fs = random.choice(obs_events_fs)
                    # for oe in obs_events:
                    #     if len(oe[0]) != 0:
                    #         prompt_story = s[:oe[2]]
                    #         for ob in oe[0]:
                    #             #print(oe[2]-1)
                    #             res.append((prompt_story, f'Where does {ob} think {p} is?', current_locations_main[p][oe[2]-1]))
                    prompt_story_fs = s_test[:oe_fs[2]]
                    ob_fs = oe_fs[0].pop()
                    example_string += f'{".".join(prompt_story_fs)}\nQ: Where does {ob_fs} think {p} is?\nA: {current_locations_test[p][oe_fs[2]-1]}\n\n;'
                    break
                if length_found:
                    break
            # print(example_string)
        return example_string
'''
Start of test code
'''
# will add not gpt models in later
not_gpt = ['llama3.1-405b','llama3.1-70b', 'llama3.1-8b', 'llama3-70b', 'llama3-8b','mixtral-8x22b-instruct', 'mixtral-8x7b-instruct', 'mistral-7b-instruct']
model_list = ['gpt-3.5-turbo','gpt-4']
# Number of trials for experiments
num_trials = 50
# Threshold for story length
length_threshold = 10
print('starting experiments...')
accuracies = {m:[] for m in model_list}
# TODO: Save results with prompt for debugging purposes
res_dict = {'Prompt':[], 'GPT3.5':[], 'GPT4':[], 'Label':[]}
df = pd.DataFrame(res_dict)


# Seed for experiments is 50
random.seed(50)
for i in range(num_trials):
    # print progress
    if i % 10 == 0:
        print(f'Trial {i}')
    n_steps = 100
    res = ()
    # Find places in the story to ask questions. Reject random samples less than length threshold
    while(True):
        # Generate experiment story
        test_sims, current_locations_main = run_sim(n_steps)
        story = print_sequences(test_sims, relation)
        s = story.split('.')
        tm_test = make_temporal_tracking_map(n_steps, current_locations_main)
        # Find places to ask questions.
        
        
        new_order = random.sample(people, len(people))
        length_found = False
        for p in new_order:
            obs_events = find_observance_event(tm_test, p, current_locations_main)
            obs_events = [e for e in obs_events if len(e[0]) != 0 and e[2]-1 >= length_threshold]
            if len(obs_events) == 0:
                continue
            length_found = True
            oe = random.choice(obs_events)
            # for oe in obs_events:
            #     if len(oe[0]) != 0:
            #         prompt_story = s[:oe[2]]
            #         for ob in oe[0]:
            #             #print(oe[2]-1)
            #             res.append((prompt_story, f'Where does {ob} think {p} is?', current_locations_main[p][oe[2]-1]))
            prompt_story = s[:oe[2]]
            ob = oe[0].pop()
            res = (prompt_story, f'Where does {ob} think {p} is?', current_locations_main[p][oe[2]-1])
            break
        if length_found:
            break
    
    
    model_res = {m:[] for m in model_list}
    for m in model_list:
        # ---- Few shot prompt(CHANGE BOOLEAN UP TOP TO MAKE COT) ----
        # uncommon when running few shot with either chain of thought or not
        examples = make_fewshot_examples(8, False, length_threshold).replace(";",'')
        gpt_res = prompt_gpt(f'{initial_prompt}\n{examples}{res[0]}\nQ: {res[1]}\nA:', m)
        # ---- Zero shot prompt ----
        # uncomment when running normal zero shot
        # gpt_res = prompt_gpt(f'{initial_prompt}\n{res[0]}\n{res[1]}', m)
        # ---- CoT Zero shot Prompt ----
        # uncomment when running with chain of thought zero shot
        #gpt_res = prompt_gpt(f'{initial_prompt}\n{res[0]}\n{res[1]} Think step by step before providing your answer.', m)
        model_res[m].append((gpt_res, res[2]))       
        score = [1 if r[1] in r[0] else 0 for r in model_res[m]]
        accuracies[m].append(sum(score)/len(model_res[m]))
    #print([len(r[0]) for r in res])
    add_to_df = {'Prompt':f'{" ".join(res[0])} {res[1]}', 'GPT3.5':[model_res['gpt-3.5-turbo'][-1][0]],
                 'GPT4':[model_res['gpt-4'][-1][0]], 'Label':res[2]}
    df = pd.concat([df, pd.DataFrame(add_to_df)], ignore_index=True)
    

print(f'Model Accuracies: --{accuracies}--')

for m in model_list:
    avg = sum(accuracies[m])/len(accuracies[m])
    sem = stats.sem(accuracies[m])
    confidence_level = 0.95
    degrees_freedom = len(accuracies[m]) - 1
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                        loc=avg, scale=sem)
    print(f'{m} Avg: {avg}')
    print(f'{m} 95 percent Confidence Interval: {confidence_interval}')
name = 'redo_fewshot_50'
df.to_csv(f'results_{name}.csv', index=False)
