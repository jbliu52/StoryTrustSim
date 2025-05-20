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
    def __init__(self, people, locations, relation, params, trial_seed=None, graph=None, events=None, storyboard=None, actions=None):
        from collections import defaultdict
        # load_dotenv()
        # self.api_key = os.getenv('OPENAI_KEY')
        # os.environ['OPENAI_API_KEY'] = self.api_key


        # Experiment constants
        self.people = people
        self.locations = locations
        self.relation = relation
        self.seed = trial_seed
        self.actions = actions
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
                    return depth, path[1:] + [current]

                if current not in visited:
                    visited.add(current)
                    queue.extend((neighbor, depth + 1, path + [current]) for neighbor in self.graph[current])

            return float('inf'), []  # No path found
        
    def find_k_unique_paths(self,g, start, end, k):
        def dfs(node, path, visited, paths):
            if len(paths) >= k:  # Stop early if we found k paths
                return
            
            if node == end:  # If reached destination, store the path
                paths.append(list(path))
                return

            for neighbor in g.get(node, []):  # Explore neighbors
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)

                    dfs(neighbor, path, visited, paths)

                    # Backtrack
                    visited.remove(neighbor)
                    path.pop()

        paths = []
        dfs(start, [start], {start}, paths)  # Start DFS
        return [len(p)-1 for p in paths[:k]],[p[1:] for p in paths[:k]]  # Return up to k paths

  
    def run_simulation(self, steps):
        # Initialize
        
        knuth = [0] * steps
        left, start = 0, 0
        # Storing the path for all the required events
        required_events = {}
        if self.actions:
            actions = {t: f"{self.actions[t]['actor']} {self.actions[t]['action']}" for t in self.actions}
        # Planning phase
        for t in self.events:
            ev = self.events[t]
            if ev['name'] == 'cross_paths':
                group = ev['actors']
                locs = ev['location']
                path_info = []
                for person in group:
                    path_step = []
                    # Start at hallway? No, start at the place you were before
                    prev = self.locations[-1] if 'prev' not in ev else ev['prev']
                    for l in locs:
                        if ev['path_type'] == 'unique':
                            pi = self.find_k_unique_paths(self.graph, prev, l, len(group))
                            i = group.index(person)
                            path_step.append((pi[0][i], pi[1][i]))
                        else:
                            pi = self.find_shortest_path(prev, l)
                            path_step.append(pi)
                        prev = path_step[-1][1][-1]
                    path_step = (sum([p[0] for p in path_step]), sum([p[1] for p in path_step], []))
                    path_info.append(path_step)
                # Combine paths
                # path_info = [self.find_shortest_path(self.current_locations[p][-1], ev['location']) for p in group]
                left = start
                for p_i in range(len(group)):
                    right = left + path_info[p_i][0]
                    knuth[left:right] = [p_i+1] * (right - left)
                    left = right
                knuth[right-1] = 0
                required_events[t] = [p[1] for p in path_info]
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
                start = t + 1
        # Generation step
        # print(knuth)
        # Generation phase
        sequences = []
        if self.events:
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
                sequences.append(f"{self.relation}({actor}, {new_loc[-1]}, {self.time_step})\n")
                self.update_state(actor, new_loc[-1])
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
                mislead_person = required_events[next_event][0:-1]
                poi = required_events[next_event][-1]
                # Mislead person 1
                location_set = {self.current_locations[p][-1] for p in mislead_person}
                choices = [l for l in self.possible_moves[poi] if l not in location_set]
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
                if self.events:
                    choices = [p for p in self.people if p not in self.events[next_event]['actors']]
                else:
                    choices = self.people
                person = random.choice(choices)
                loc = random.choice(self.possible_moves[person])
                sequences.append(f"{self.relation}({person}, {loc}, {self.time_step})\n")
                self.update_state(person, loc)
                self.time_step += 1
            if self.actions and self.time_step - 1 in actions:
                sequences.append(f'*{actions[self.time_step-1]}') 
        return sequences
    
    def formal_to_story(self, sequence_list):
        strings = []
        for e in sequence_list:
            if e[0] == '*':
                strings.append(e[1:])
            else:
                e.replace('\n','')
                e = e.replace(f'{self.relation}(','').replace(')','')
                subject, loc, time = e.split(',')
                # TODO: Fix for both objects being placed and holes set up  
                res = f'{subject} {self.relation.replace("_", " ") if "hole" in loc else self.relation.replace("_", " ").replace("in","in") }{loc}'
                #print(res)
                strings.append(res)
        return '. '.join(strings).strip() 
    
    
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

if __name__ == '__main__':
    # possible_people = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia", "Hank", "Isaac", "Jake", "Kevin"]
    # num_people = 8
    # # graph = {
    # #         "hole_1": ["hole_2", "field"],
    # #         "hole_2": ["hole_1", "hole_3"],
    # #         "hole_3": ["hole_2", "hole_4"],
    # #         "hole_4": ["hole_3", "field"],
    # #         "field": ["hole_1", "hole_4"]
    # #     }
    
    # graph = { 
    #     "hole_1": ["hole_2", "field","hole_5"],
    #     "hole_2": ["hole_1", "hole_3","field"],
    #     "hole_3": ["hole_2", "hole_4","field"],
    #     "hole_4": ["hole_3", "hole_5","hole_1"],
    #     "hole_5": ["hole_4", "hole_1","hole_2"],
    #     "field": ["hole_1", "hole_4","hole_2"]
    # }
    
    # # graph = { 
    # #     "hole_1": ["hole_2", "field", "hole_5", "hole_6", "hole_4"],
    # #     "hole_2": ["hole_1", "hole_3", "field", "hole_5", "hole_7"],
    # #     "hole_3": ["hole_2", "hole_4", "field", "hole_8", "hole_1"],
    # #     "hole_4": ["hole_3", "hole_5", "hole_1", "hole_9", "field"],
    # #     "hole_5": ["hole_4", "hole_1", "hole_2", "hole_10", "field"],
    # #     "hole_6": ["hole_1", "hole_7", "hole_9", "field"],
    # #     "hole_7": ["hole_6", "hole_2", "hole_8", "field"],
    # #     "hole_8": ["hole_7", "hole_3", "hole_9", "field"],
    # #     "hole_9": ["hole_8", "hole_4", "hole_6", "hole_10", "field"],
    # #     "hole_10": ["hole_9", "hole_5", "field", "hole_7"],
    # #     "field": ["hole_1", "hole_2", "hole_3", "hole_4", "hole_9", "hole_10"]
    # # }
    
    # locations = list(graph.keys())
    # story_length = 25
    # num_trials = 10
    # mislead_distance = 3

    # random.seed(25)

    # for _ in range(num_trials):
    #     event_dict = {}
        
    #     mislead_1 = 3
    #     poi = random.sample(possible_people[:num_people], 3)
    #     loc = random.sample(locations[:-1],1)
    #     second_loc = random.sample([l for l in locations[:-1] if l != loc[0]], 1)
    #     third_loc = random.sample([l for l in locations[:-1] if l != loc[0] and l != second_loc[0]], 1)
    #     print(poi)
        
    #     event_dict[10] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
    #     event_dict[11] = {"name":"move", "actor":poi[-1], "location": second_loc[0]}
    #     event_dict[17] = {"name": "cross_paths","actors": poi[1:], "location": third_loc, "path_type": "same"}
    #     event_dict[18] = {"name": "exclusive_random", "actors": poi, "stop": story_length}   
            
    #     sim = StorySimulator(
    #         people=possible_people[:num_people],
    #         locations=list(graph.keys()),
    #         relation="jumps_in",
    #         params={'prompt': '3', 'type': 'cot'},
    #         graph=graph,
    #         events=event_dict
    #     )
    #     res = sim.run_simulation(story_length)
    #     story = sim.formal_to_story(res)
    #     print("\n".join(story.split(".")))
    #     print('-----')
    possible_people = ["Alice", "Bob", "Charlie", "Danny", "Edward", "Frank", "Georgia", "Hank", "Isaac", "Jake", "Kevin"]
    num_people = 5

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
    num_trials = 3
    mislead_distance = 3

    random.seed(25)

    for _ in range(num_trials):
        event_dict, actions_dict, label, experiment_dict = write_on_wall_then_erase(possible_people[:num_people], locations[:-1], graph, mislead_distance, story_length)
    

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            relation="enters",
            params={'prompt': '3', 'type': 'cot'},
            graph=graph,
            events=event_dict,
            actions=actions_dict
        )
        res = sim.run_simulation(story_length)
        
        story = sim.formal_to_story(res)
        
        split_story = story.split('.')
        #split_story.insert(story_length-1, f" {experiment_dict['poi'][1]} forgets what they last saw on the wall")
        print(".\n".join(split_story))
        print('-----')
        pass
            
            
