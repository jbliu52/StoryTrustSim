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
    def __init__(self, people, locations, relation, params, trial_seed=None, graph=None, events=None, storyboard=None):
        from collections import defaultdict
        # load_dotenv()
        # self.api_key = os.getenv('OPENAI_KEY')
        # os.environ['OPENAI_API_KEY'] = self.api_key


        # Experiment constants
        self.people = people
        self.locations = locations
        self.relation = relation
        self.seed = trial_seed
        if self.seed is not None:
            random.seed(self.seed)
        self.params = params
        # self.initial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in the {self.locations[-1]}."
        self.sequences = []
        self.storyboard = storyboard
        #self.client = OpenAI()
        
        self.state = dict()
        # Adjacency graph for locations
        self.graph = graph if graph else self._create_fully_connected_graph()
        self.possible_moves = {p: self.graph[locations[-1]] for p in people}

        # Simulation time step
        self.time_step = 0
        # Observation steps
        self.events = events if events else dict()
        
        self.current_locations = {person: [self.locations[-1]] for person in self.people}

    def _create_fully_connected_graph(self):
        """Creates a fully connected graph from locations."""
        return {loc: [l for l in self.locations if l != loc] for loc in self.locations}

    def update_state(self, subject, new_loc, t=None):
        if t is not None and t != self.time_step:
            self.current_locations[subject][t] = new_loc
        for person in self.people:
            if person == subject:
                self.current_locations[person].append(new_loc)
                self.possible_moves[person] = self.graph[new_loc]
            else:
                self.current_locations[person].append(self.current_locations[person][-1])
        

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

  
    # TODO: Track question answers
    def run_simulation(self, steps):
        # Initialize
        
        knuth = [0] * steps
        left, start = 0, 0
        # Storing the path for all the required events
        required_events = {}
        
        # Planning phase
        for t in self.events:
            ev = self.events[t]         
            if ev['name'] == 'cross_paths':
                group = ev['actors']
                locs = ev['location']
                # TODO: Fix starting location
                path_info = [self.find_shortest_path(self.current_locations[p][-1], ev['location']) for p in group]
                left = start
                for p_i in range(len(group)):
                    right = left + path_info[p_i][0]
                    knuth[left:right] = [p_i+1] * (right - left)
                    left = right
                knuth[right-1] = 0
                required_events[t] = [p[1][1:] for p in path_info]
                # Shuffle
                x = [a for a in knuth[start:t]]
                random.shuffle(x)
                knuth[start:t] = x
                # Last step is always -100
                knuth[t] = -100
                start = t+1
            elif ev["name"] == "exclusive_random":
                exclude = ev['actors']
                required_events[t] = (exclude, ev['stop'])
                knuth[t:ev['stop']] = [-2] * (ev['stop'] - t)
                start = ev['stop']
            elif ev['name'] == 'mislead':
                required_events[t] = ev['actors']
                knuth[t] = -101
                start = t + 1
            elif ev['name'] == 'move':
                required_events[t] = ev['actor']
                knuth[t] = -102
        # Generation step
        # print(knuth)
        # Generation phase
        sequences = []
        event_list = iter(sorted(self.events.keys()))
        next_event = next(event_list)        
        paths, indices = None, None
        for i in knuth:
            if i > 0 : # Cross paths
                if paths == None: 
                    # Index of a person to make cross paths
                    paths = required_events[next_event]
                    indices = [0] * len(paths)
                actor = self.events[next_event]['actors'][i-1]
                sequences.append(f"{self.relation}({actor}, {paths[i-1][indices[i-1]]}, {self.time_step})\n")
                self.update_state(actor, paths[i-1][indices[i-1]])
                indices[i-1] += 1
                self.time_step += 1    
            elif i == -100:
                # Move the last person in the actor list to the appropriate spot
                new_loc = self.events[next_event]['location']
                actor = self.events[next_event]['actors'][-1]
                sequences.append(f"{self.relation}({actor}, {new_loc}, {self.time_step})\n")
                self.update_state(actor, new_loc)
                self.time_step += 1
                # Reset
                paths = None
                indices = None
                try:
                    next_event = next(event_list)
                except:
                    # This means we're done
                    continue
            elif i == -101:
                choices = []
                mislead_person = required_events[next_event][0]
                poi = required_events[next_event][1]
                # Mislead person 1
                choices = [l for l in self.possible_moves[poi] if l != self.current_locations[mislead_person][-1]]
                new_loc = random.choice(choices)
                sequences.append(f"{self.relation}({poi}, {new_loc}, {self.time_step})\n")
                self.update_state(poi, new_loc)
                self.time_step += 1
                try:
                    next_event = next(event_list)
                except:
                    # This means we're done
                    continue
            elif i == -102:
                # Move person 2, but person 1 knows
                new_loc = self.events[next_event]['location']
                if self.events[next_event]['name'] != 'move':
                    print(self.events[next_event])
                actor = self.events[next_event]['actor']
                sequences.append(f"{self.relation}({actor}, {new_loc}, {self.time_step})\n")
                self.update_state(actor, new_loc)
                self.time_step += 1
                try:
                    next_event = next(event_list)
                except:
                    # This means we're done
                    continue
            elif i == -2:
                excluded_actors = set(required_events[next_event][0])
                person = random.choice([p for p in self.people if p not in excluded_actors])
                loc = random.choice(self.possible_moves[person])
                sequences.append(f"{self.relation}({person}, {loc}, {self.time_step})\n")
                self.update_state(person, loc)
                self.time_step += 1
                if self.time_step == required_events[next_event][1]:
                    try:
                        next_event = next(event_list)
                    except:
                        # This means we're done
                        continue
            else:
                if paths == None:
                    choices = self.people
                else:
                    choices = [p for p in self.people if p not in self.events[next_event]['actors']]
                person = random.choice(choices)
                loc = random.choice(self.possible_moves[person])
                sequences.append(f"{self.relation}({person}, {loc}, {self.time_step})\n")
                self.update_state(person, loc)
                self.time_step += 1
        return sequences
    
    def formal_to_story(self, sequence_list):
        strings = []
        for e in sequence_list:
            e.replace('\n','')
            e = e.replace(f'{self.relation}(','').replace(')','')
            subject, loc, time = e.split(',')
            res = f'{subject} {self.relation.replace("_", " ") if "hole" in loc else self.relation.replace("_", " ").replace("in","out to the") }{loc}'
            #print(res)
            strings.append(res)
        return '. '.join(strings).strip()   

if __name__ == '__main__':
    # Define the graph and observation events
    possible_people = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia"]
    num_people = 5
    locations = ["hole_1", "hole_2", "hole_3", "hole_4", "field"]
    graph = {
            "hole_1": ["hole_2", "field"],
            "hole_2": ["hole_1", "hole_3"],
            "hole_3": ["hole_2", "hole_4"],
            "hole_4": ["hole_3", "field"],
            "field": ["hole_1", "hole_4"]
        }
    story_length = 21
    num_trials = 5
    mislead_distance = 3

    random.seed(25)

    for _ in range(num_trials):
        event_dict = {}
        poi = random.sample(possible_people[:num_people], 2)
        loc = random.sample(locations,3)
        #second_loc = random.choice(graph[loc])
        event_dict[15] = {"name": "cross_paths","actors": poi, "location": [loc[0], loc[2]]}
        event_dict[16] = {"name":"move", "actor":poi[-1], "location": loc[2]}
        event_dict[17] = {"name": "exclusive_random", "actors": poi, "stop": 17 + mislead_distance}
        event_dict[17 + mislead_distance] = {"name": "mislead", "actors": poi}
        #event_dict[17 + mislead_distance+1] = {"name": "exclusive_random", "actors": poi, "stop": story_length}
        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=["hole_1", "hole_2", "hole_3", "hole_4", "field"],
            relation="jumps_in",
            params={'prompt': '3', 'type': 'cot'},
            graph=graph,
            events=event_dict
        )
        res = sim.run_simulation(story_length)
        cp_answer = sim.current_locations[poi[1]][10]
        story = sim.formal_to_story(res)
        
        
