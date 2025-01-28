from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import numpy as np
import regex as re
import pandas as pd
from scipy import stats


'''
Storyboard - a set of events that are required to happen. An outline of the story.

constraints = [cross(Alice, Bob, 5, hole_4)]
cross(X,Y) -> Path for X to hole_4 steps interleaved with Y to hole_4
-> Figure out number of steps it takes, then do shuffle on that


'''

class StorySimulator:
    def __init__(self, people, locations, relation, models, num_trials, length_threshold, trial_seed, params, graph=None, obs_steps=None, storyboard=None):
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
        self.sequences = []
        self.storyboard = storyboard
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

    def update_state(self, subject, new_loc, current_locations, t=None):
        if t is not None and t != self.time_step:
            current_locations[subject][t] = new_loc
            return current_locations
        for person in self.people:
            if person == subject:
                current_locations[person].append(new_loc)
                self.possible_moves[person] = self.graph[new_loc]
            else:
                current_locations[person].append(current_locations[person][-1])
        return current_locations

    def find_shortest_path(self, start, target):
            """Uses BFS to find the shortest path and its length from start to target."""
            visited = set()
            queue = [(start, 0, [])]  # (current_location, depth, path)

            while queue:
                current, depth, path = queue.pop(0)

                if current == target:
                    return depth, path + [current]

                if current not in visited:
                    visited.add(current)
                    queue.extend((neighbor, depth + 1, path + [current]) for neighbor in self.graph[current])

            return float('inf'), []  # No path found


    def validate_observation_events(self):
        """Checks if all observation events are feasible given the graph."""
        for t, details in self.obs_steps.items():
            location = details["location"]
            max_steps = t

            for person in details["actors"]:
                path_length, _ = self.find_shortest_path(self.locations[-1], location)
                if 2* path_length >= max_steps:
                    raise ValueError(f"Observation event at time {t} with location {location} is not possible. Path length is {path_length}, but only {max_steps} steps are available.")
                else:
                    return path_length

    def observation_event(self, subject, target, location, current_locations):
        new = random.choice(self.graph[location])
        current_locations = self.update_state(subject, location, current_locations)
        self.sequences.append(f"*{self.relation}({subject}, {location}, {self.time_step})\n")
        current_locations = self.update_state(target, location, current_locations)
        self.sequences.append(f"*{self.relation}({target}, {location}, {self.time_step+1})\n")
        current_locations = self.update_state(subject, new, current_locations)
        self.sequences.append(f"*{self.relation}({subject}, {new}, {self.time_step+2})\n")
        self.time_step += 3
        return current_locations

    def align_actors_to_location(self, actors, location, current_locations):
        """Aligns actors to the target location based on shortest paths. Assume we're starting from 0"""
        events = ""
        aligned_actions = {a:[] for a in actors}
        for actor in actors:
            steps, path = self.find_shortest_path(current_locations[actor][0], location)
            if steps == float('inf'):
                raise ValueError(f"No path exists to align actor {actor} to location {location}.")
            aligned_actions[actor] = path
        # Randomly pick between two actors to build the aligned path
        count = 0
        actors_list = actors
        while(sum([len(aligned_actions[a]) for a in aligned_actions]) != 0):
            actor_choice = random.choice(actors_list)
            actor_step = aligned_actions[actor_choice].pop(0)
            if len(aligned_actions[actor_choice]) == 0:
                actors_list.remove(actor_choice)
            self.sequences[count] = (f"@{self.relation}({actor_choice}, {actor_step}, {count})\n")
            current_locations = self.update_state(actor_choice, actor_step, current_locations, t=count)
            count += 1
        return events, current_locations

    '''
    TODO: Try with multiple events
    '''
    def run_simulation(self, steps):
        current_locations = {person: [self.locations[-1]] for person in self.people}
        # Validate observation events - check if it's possible to reach the locations from the start

        knuth = [0] * steps
        
        left = 0
        required_events = {}
        for t in self.obs_steps:
            path_length, path = self.find_shortest_path(self.locations[-1], self.obs_steps[t]['location'])
            right = left + 2*path_length
            path = path[1:]
            knuth[left:left+path_length] = [1] * path_length
            knuth[left+path_length:right] = [2] * path_length
            required_events[t] = (self.obs_steps[t]['actors'], path)
            x = [a for a in knuth[left:t]]
            random.shuffle(x)
            knuth[left:t] = x  
            
            knuth[t], knuth[t+1] = 100, 101
            # Generation step
            p1 = 0
            p2 = 0
            sequences = []
            print(knuth)
            for i in knuth[left:t]:
                if i == 1:
                    sequences.append(f"{self.relation}({self.obs_steps[t]['actors'][0]}, {path[p1]}, {self.time_step})\n")
                    current_locations = self.update_state(self.obs_steps[t]['actors'][0], path[p1], current_locations)
                    self.time_step += 1
                    p1 += 1
                elif i == 2:
                    sequences.append(f"{self.relation}({self.obs_steps[t]['actors'][1]}, {path[p2]}, {self.time_step})\n")
                    current_locations = self.update_state(self.obs_steps[t]['actors'][1], path[p2], current_locations)
                    self.time_step += 1
                    p2 += 1
                elif i == 100:
                    new_loc = random.choice([l for l in self.possible_moves[self.obs_steps[t]['actors'][1]] if l !=self.obs_steps[t]['location']])
                    sequences.append(f"{self.relation}({self.obs_steps[t]['actors'][1]}, {new_loc}, {self.time_step})\n")
                    current_locations = self.update_state(self.obs_steps[t]['actors'][1], new_loc, current_locations)
                    self.time_step += 1
                else:
                    choices = [p for p in self.people if p != self.obs_steps[t]['actors'][0] and p != self.obs_steps[t]['actors'][1]]
                    person = random.choice(choices)
                    loc = random.choice(self.possible_moves[person])
                    sequences.append(f"{self.relation}({person}, {loc}, {self.time_step})\n")
                    current_locations = self.update_state(person, loc, current_locations)
                    self.time_step += 1
            # After
            left = t
        if left != steps:
            for _ in knuth[left:]:
                person = random.choice(self.people)
                loc = random.choice(self.possible_moves[person])
                sequences.append(f"{self.relation}({person}, {loc}, {self.time_step})\n")
                current_locations = self.update_state(person, loc, current_locations)
                self.time_step += 1
        print('\n'.join(sequences))  

        return sequences

if __name__ == '__main__':
    # Define the graph and observation events
    graph = {
        "hole_1": ["hole_2", "field"],
        "hole_2": ["hole_1", "hole_3"],
        "hole_3": ["hole_2", "hole_4"],
        "hole_4": ["hole_3", "field"],
        "field": ["hole_1", "hole_4"]
    }

    obs_steps = {
        7: {"actors": ["Alice", "Bob"], "location": "hole_2"},
        14: {"actors": ["Charlie", "Danny"], "location": "hole_4"},
    }

    sim = StorySimulator(
        people=["Alice", "Bob", "Charlie", "Danny"],
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

    sim.run_simulation(30)
