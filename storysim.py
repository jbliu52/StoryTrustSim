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
    def __init__(self, people, locations, relation, models, num_trials, length_threshold, trial_seed, params, graph=None, obs_steps=None):
        from collections import defaultdict
        # load_dotenv()
        # self.api_key = os.getenv('OPENAI_KEY')
        # os.environ['OPENAI_API_KEY'] = self.api_key
        
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
        #self.client = OpenAI()
        
        # Data storage
        self.accuracies = {m: [] for m in models}
        d = {m: [] for m in self.models}
        d['Prompt'] = []
        d['Label'] = []
        self.results_df = pd.DataFrame(d)
        
        self.state = dict()
        # Adjacency graph for locations
        self.graph = graph if graph else self._create_fully_connected_graph()
        self.possible_moves = {p: self.graph[locations[-1]] for p in people}

        # Simulation time step
        self.time_step = 0
        # Observation steps
        self.obs_steps = obs_steps if obs_steps else []

    def _create_fully_connected_graph(self):
        """Creates a fully connected graph from locations."""
        return {loc: [l for l in self.locations if l != loc] for loc in self.locations}

    def update_state(self, subject, new_loc, current_locations):
        for person in self.people:
            if person == subject:
                current_locations[person].append(new_loc)
                self.possible_moves[person] = self.graph[new_loc]
            else:
                current_locations[person].append(current_locations[person][-1])
        return current_locations

    def observation_event(self, subject, target, location, current_locations):
        new = random.choice(self.graph[location])
        event_string = f"""*{self.relation}({subject}, {location}, {self.time_step})
        *{self.relation}({target}, {location}, {self.time_step+1})
        *{self.relation}({subject}, {new}, {self.time_step+2})
        ---
        """
        self.time_step += 3
        current_locations = self.update_state(subject, location, current_locations)
        current_locations = self.update_state(target, location, current_locations)
        current_locations = self.update_state(subject, new, current_locations)
        return event_string, current_locations

    def align_for_event(self, actors, location, current_locations):
        events = ''
        for actor in actors:
            done = False
            while not done:
                possible = self.possible_moves[actor]
                if location not in possible:
                    new_loc = random.choice(possible)
                    current_locations = self.update_state(actor, new_loc, current_locations)
                    events += f"#{self.relation}({actor}, {new_loc}, {self.time_step})\n"
                    self.time_step += 1
                else:
                    done = True
        return events, current_locations

    def can_align_for_event(self, actors, location, current_locations, max_steps):
        """Checks if alignment is possible for the given actors and location within a time constraint using BFS."""
        def bfs(start, target, max_depth):
            visited = set()
            queue = [(start, 0)]  # (current_location, current_depth)

            while queue:
                current, depth = queue.pop(0)

                if depth > max_depth:
                    continue

                if current == target:
                    return True

                if current not in visited:
                    visited.add(current)
                    queue.extend((neighbor, depth + 1) for neighbor in self.graph[current])

            return False

        for actor in actors:
            start_location = current_locations[actor][-1]
            if not bfs(start_location, location, max_steps):
                return False
        return True

    def run_simulation(self, steps):
        current_locations = {person: [self.locations[-1]] for person in self.people}
        sequences = ""
        
        for t in range(steps):
            if t in self.obs_steps:
                # User-specified observation event details
                obs_event_details = self.obs_steps[t]  # Dict with keys: "actors" and "location"
                subject, target = obs_event_details["actors"]
                new_loc = obs_event_details["location"]
                max_steps = steps - self.time_step

                if self.can_align_for_event([subject, target], new_loc, current_locations, max_steps):
                    alignment_events, current_locations = self.align_for_event([subject, target], new_loc, current_locations)
                    new_event, current_locations = self.observation_event(subject, target, new_loc, current_locations)
                    sequences += alignment_events + new_event
                else:
                    sequences += f"# Observation event at time {t} could not occur due to alignment issues.\n"
            else:
                person = random.choice(self.people)
                loc = random.choice(self.possible_moves[person])
                current_locations = self.update_state(person, loc, current_locations)
                sequences += f"{self.relation}({person}, {loc}, {self.time_step})\n"
                self.time_step += 1
        return sequences, current_locations


if __name__ == '__main__':
    # TODO: Add support for JSON
    graph = {
        "hole_1": ["hole_2", "field"],
        "hole_2": ["hole_1", "hole_3"],
        "hole_3": ["hole_2", "hole_4"],
        "hole_4": ["hole_3", "field"],
        "field": ["hole_1", "hole_4"]
    }

    obs_steps = {
        3: {"actors": ["Alice", "Bob"], "location": "hole_2"},
        7: {"actors": ["Alice", "Bob"], "location": "field"}
    }

    sim = StorySimulator(
        people=["Alice", "Bob"],
        locations=["hole_1", "hole_2", "hole_3", "hole_4", "field"],
        relation="jumps_in",
        models=["gpt-4"],
        num_trials=20,
        length_threshold=50,
        trial_seed=50,
        params={'prompt': '3', 'type': 'cot'},
        graph=graph,
        obs_steps=obs_steps
    )

    print(sim.run_simulation(10)[0])
