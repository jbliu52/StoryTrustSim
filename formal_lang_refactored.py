# File path: /mnt/data/refactored_experiment_runner.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import numpy as np
import regex as re
import pandas as pd
from scipy import stats

class ExperimentRunner:
    def __init__(self, people, locations, relation, models, num_trials, length_threshold, trial_seed, params):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_KEY')
        os.environ['OPENAI_API_KEY'] = self.api_key
        
        self.distractor_actions = [
        "draws a triangle on the wall",
        "writes their name on the floor",
        "picks up a stone",
        "sits down to rest",
        "places 3 twigs on the ground",
        "takes off their shoes",
        "hopes they don't have to come back here"
        ]

        # Experiment constants
        self.people = people
        self.locations = locations
        self.relation = relation
        self.models = models
        self.num_trials = num_trials
        self.length_threshold = length_threshold
        self.seed = trial_seed
        random.seed(self.seed)
        self.params = params
        self.initial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in the {self.locations[-1]}."
        # OpenAI client setup
        self.client = OpenAI()
        res_dict = {'Prompt':[], 'GPT3.5':[], 'GPT4':[], 'Label':[]}
        self.res_df = pd.DataFrame(res_dict)
        # Data storage
        self.accuracies = {m: [] for m in models}
        self.results_df = pd.DataFrame({'Prompt': [], 'GPT3.5': [], 'GPT4': [], 'Label': []})

    def initialize_locations(self):
        return {person: [self.locations[-1]] for person in self.people}

    def update_state(self, subject, new_loc, current_locations):
        for person in self.people:
            if person == subject:
                current_locations[person].append(new_loc)
            else:
                current_locations[person].append(current_locations[person][-1])

    def run_simulation(self, steps):

        current_locations = self.initialize_locations()
        sequences = ""
        for t in range(steps):
            person = random.choice(self.people)
            cur_loc = current_locations[person]
            loc = self.locations[-1] if self.locations[0][:-2] in cur_loc[-1] else random.choice(self.locations[:-1])
            self.update_state(person, loc, current_locations)
            sequences += f"{self.relation}({person}, {loc}, {t})\n"
            if random.random() > 0.3:
                sequences += f"{person} {random.choice(distractor_actions)}\n"
        return sequences, current_locations

    def print_sequences(self, sequences):
        events = sequences.strip().split("\n")
        strings = []
        for event in events:
            if self.relation not in event:
                res = event
            else:
                event = event.replace(f"{self.relation}(", "").replace(")", "")
                subject, loc, _ = event.split(",")
                res = f'{subject} {self.relation.replace("_", " ") if "hole" in loc else self.relation.replace("_", " ").replace("in","out to the") }{loc}'
            strings.append(res)
        return ". ".join(strings)
    

    def make_temporal_tracking_map(self, steps, current_locations):
        steps = steps
        tracking_map = [{self.locations[-1]: set(self.people)}]
        for t in range(1, steps):
            location_to_people = {}
            for person in self.people:
                curr_loc = current_locations[person][t]
                if curr_loc not in location_to_people:
                    location_to_people[curr_loc] = set()
                location_to_people[curr_loc].add(person)
            tracking_map.append(location_to_people)
        return tracking_map

    def find_observance_event(self, temporal_map, target, current_locations):
        target_locations = current_locations[target]
        observers = set()
        events = []
        start = 0

        for t in range(len(temporal_map) - 1):
            if target_locations[t] != target_locations[t + 1]:
                if not observers:
                    observers = temporal_map[t][target_locations[t]].difference({target})
                    start = t
                else:
                    new_loc = target_locations[t + 1]
                    incoming = temporal_map[t + 1][new_loc]
                    outgoing = temporal_map[t][target_locations[t]]
                    unaware_subjects = observers.difference(incoming).difference(outgoing)
                    events.append((unaware_subjects, start, t + 1))
                    observers = set()
                    start = 0
        return events
    
    def generate_examples(self, num: int, prompt_type: str):
        example_string = ''
        # Make a little story
        example_trials = self.run_trials(num)
        # Make reasoning traces from the example trials
        for trial in example_trials:
            question = trial[0]
            answer = trial[1]
            temporal_map = trial[2]
            if prompt_type == 'standard':
                example_string += f'{question} {answer}\n'
                continue
            # Alice jumps out to the field.\nQ: Where does Dylan think Alice is?\nA:'
            q = question.split('\n')[1].split(' ')
            # This is hard-coded, need to fix this later
            subject = q[q.index('think')-1]
            target = q[q.index('think')+1]
            story = question.split('Q:')[0]
            story = story.split(".")
            # Iterate through the story and create a reasoning trace for the CoT prompt
            # trace_string = f'First, both {target} and {subject} are in the {self.locations[-1]}. Then'
            trace_string = f'First, both {target} and {subject} are in the {self.locations[-1]}.'
            # for t, part in enumerate(story):
            #     if part.strip().startswith(target):
            #         if len(part) == 0:
            #             continue
            #         trace_string += part
            #         current_loc = part.split(' ')[-1]
            #         if subject in temporal_map[t+1][current_loc]:
            #             trace_string += f'. Since {subject} is also in {current_loc} at that point, {subject} sees {target}'
            #         trace_string += '. Then,'
            # trace_string = "".join(trace_string.rsplit(' Then,', 1))
                
            subject_string = f' The last place {subject} saw {target} was when they were both in '
            # last_loc = ''
            # Go backwards through the temporal map and find when they were both together
            found = False
            for tm in reversed(temporal_map):
                for loc_t in tm:
                    if subject in tm[loc_t] and target in tm[loc_t]:
                        example_string += question + " " +trace_string + subject_string + loc_t + f". Then, {subject} saw {target} go to {answer}"+ f". Therefore, {subject} thinks {target} is in {answer}.\n" 
                        found = True
                if found:
                    break   
        return example_string

    def prompt_gpt(self, prompt, model):
        full_prompt = ''
        examples = ''
        if self.params['prompt'] == '0':
            full_prompt = f'{prompt}{" Think step by step before providing your answer" if self.params["type"] == "cot" else ""}'
        else:
            # Make examples
            examples = self.generate_examples(int(self.params['prompt']), self.params['type'])
            full_prompt = self.initial_prompt + " Think step by step before providing your answer." +'\n' +examples + prompt 
        # response = self.client.chat.completions.create(
        #     model=model,
        #     messages=[{"role": "user", "content": examples+full_prompt}],
        # )
        
        # return response.choices[0].message.content
        return examples+full_prompt

    def run_trials(self, n, seed=None):
        #print(f"Generating {self.num_trials} stories...")
        
        if seed is not None:
            random.seed(seed)
        trial_res = []
        for trial in range(n):
            prompt_story = ''
            current_trial = ()
            while True:
                n_steps = 100
                test_simulations, current_locations = self.run_simulation(n_steps)
                story = self.print_sequences(test_simulations)
                sentences = story.split(". ")
                temporal_map = self.make_temporal_tracking_map(n_steps, current_locations)
                found = False
                # Find observance events
                new_order = random.sample(self.people, len(self.people))
                for person in new_order:
                    events = self.find_observance_event(temporal_map, person, current_locations)
                    events = [e for e in events if len(e[0]) > 0 and e[2] - 1 >= self.length_threshold]
                    if events:
                        event = random.choice(events)
                        prompt_story = sentences[:event[2]]
                        observer = random.choice(list(event[0]))
                        question = f"Where does {observer} think {person} is?"
                        answer = current_locations[person][event[2] - 1]  
                        current_trial = (f"{'. '.join(prompt_story) + '.'}\nQ: {question}\nA:", answer, temporal_map)
                        found = True
                        break
                    else:
                        continue
                if found:    
                    break
            trial_res.append(current_trial)
        random.seed(self.seed)
        #print(trial_res)
        return trial_res
        
            

    def model_evals(self):
        # Model evaluations
        print('Beginning...')
        trial_results = self.run_trials(self.num_trials, seed=50)
        
        for idx,tr in enumerate(trial_results):
            if idx % 5 == 0:
                print('Trial ', idx)
            prompt = tr[0]
            model_res = {mod:'' for mod in self.models}
            for model in self.models:
                #print(prompt)   
                response = self.prompt_gpt(prompt, model)
                model_res[model] = response 
                #is_correct = tr[1] in response.split("A:")[0][:-1].split('.')[-1]
                # This should work with CoT
                response = response.split('.')
                response = response[:-1] if response[-1] == '' else response
                is_correct = tr[1] in response[-1] 
                self.accuracies[model].append(int(is_correct))
            add_to_df = {'Prompt':prompt, 'GPT3.5':[""],
                'GPT4':[model_res['gpt-4']], 'Label':tr[1]}
            self.results_df = pd.concat([self.results_df, pd.DataFrame(add_to_df)], ignore_index=True)
        self.results_df.to_csv('results_tests.csv', index=False)
               
        
        for m in self.models:
            avg = sum(self.accuracies[m]) / self.num_trials
            se_m = stats.sem(avg)
            confidence_level = 0.95
            degrees_of_freedom = len(self.accuracies[m]) - 1
            confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom,
                                        loc=avg, scale=se_m)
            print(f'{m} Avg: {avg}')
            print(f'{m} confidence interval: {confidence_interval}')
            # print(f"Trial {trial} completed.")

        print("Experiments finished!")
        return self.accuracies

# Usage example
if __name__ == "__main__":
    # Example experiment
    # ['cot', 4, 20, 5, 0]
    runner = ExperimentRunner(
        people=["Alice", "Bob", "Joe", "Dylan", "Josh"],
        locations=["hole_1", "hole_2", "hole_3", "hole_4", "field"],
        relation="jumps_in",
        models=["gpt-4"],
        num_trials=20,
        length_threshold=50,
        trial_seed=50,
        params = {'prompt':'3','type':'cot'}
    )
    print(runner.model_evals())
