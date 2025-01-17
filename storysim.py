from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import numpy as np
import regex as re
import pandas as pd
from scipy import stats


class StorySimulator:
    def __init__(self, people, locations, relation, models, num_trials, length_threshold, trial_seed, params):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_KEY')
        os.environ['OPENAI_API_KEY'] = self.api_key
        
        # Experiment constants
        self.people = people
        self.locations = locations
        self.relation = relation
        self.models = models
        self.num_trials = num_trials
        self.length_threshold = length_threshold
        self.seed = trial_seed
        # Simulation seed
        random.seed(self.seed)
        self.params = params
        self.initial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in the {self.locations[-1]}."
        # OpenAI client setup
        self.client = OpenAI()
        # Data storage
        self.accuracies = {m: [] for m in models}
        d = {m:[] for m in self.models}
        d['Prompt'] = []
        d['Label' ] = []
        self.results_df = pd.DataFrame(d)
        
        self.state = dict()
        self.possible_moves = {p:locations[:-1] for p in people}
    

    def update_state(self, subject, new_loc, current_locations):
        for person in self.people:
            if person == subject:
                current_locations[person].append(new_loc)
                # This is where we decide next possible moves
                self.possible_moves[person] = self.locations[:-1] if new_loc == self.locations[-1] else [self.locations[-1]]
            else:
                current_locations[person].append(current_locations[person][-1])
        return current_locations

    def observation_event(self, subject, target, location, current_locations):
        # Insert an observation event
        new = self.locations[-1] if location != self.locations[1] else random.choice(self.locations[:-1])
        event_string = f"""*{self.relation}({subject}, {location})
        *{self.relation}({target}, {location})
        *{self.relation}({subject}, {new})
        ---
        """
        current_locations = self.update_state(subject, location, current_locations)
        current_locations = self.update_state(target, location, current_locations)
        current_locations = self.update_state(subject, new, current_locations)
        
        return event_string, current_locations
        
    
    def align_for_event(self, actors: list, location: str, current_locations):
        events = ''
        for actor in actors:
            done = False
            while(not done):
                # cur = current_locations[actor][-1]
                possible = self.possible_moves[actor]
                if location not in possible:
                    new_loc = random.choice(possible)
                    current_locations = self.update_state(actor, new_loc, current_locations)
                    events += f"#{self.relation}({actor}, {new_loc})\n"
                else:
                    done = True
        return events, current_locations
        
    # TODO: Is it smart to make the string as you generate the sequences? Probably not.
    def run_simulation(self, steps):
        current_locations = {person: [self.locations[-1]] for person in self.people}
        sequences = ""
        
        for t in range(steps):
            obs_event = random.choice([True, False, False, False, False, False])
            # This is where the observation event needs to happen. For now I'm giving it a 1/6 chance
            if obs_event:
                subject, target = random.sample(self.people, 2)
                # Use these to align them
                new_loc = random.choice(self.locations)
                #print(f'--{new_loc}--')
                alignment_events, current_locations = self.align_for_event([subject, target], new_loc, current_locations)
                new_event, current_locations = self.observation_event(subject, target, new_loc, current_locations)
                sequences += alignment_events + new_event
            else:
                person = random.choice(self.people)
                # cur_loc = current_locations[person]
                loc = random.choice(self.possible_moves[person])
                current_locations = self.update_state(person, loc, current_locations)
                sequences += f"{self.relation}({person}, {loc}, {t})\n"
        return sequences, current_locations
        


if __name__ == '__main__':
    sim = StorySimulator(
        people=["Alice", "Bob"],
        locations=["hole_1", "hole_2", "hole_3", "hole_4", "field"],
        relation="jumps_in",
        models=["gpt-4"],
        num_trials=20,
        length_threshold=50,
        trial_seed=50,
        params = {'prompt':'3','type':'cot'}
    )
    
    print(sim.run_simulation(10)[0])