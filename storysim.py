import random
from storyboard import Storyboard

'''
StorySimulator: A class to generate a story based on a Storyboard. The events of the story will be randomly generated aside from what's specified by the storyboard. 
'''

class StorySimulator:
    def __init__(self, people, locations, action: str | list[str], storyboard: Storyboard, trial_seed=None, graph=None, manual_actions=None):
        
        # Experiment constants
        self.people = people
        self.locations = locations
        self.action = action
        self.seed = trial_seed
        self.manual_actions = manual_actions
        if self.seed is not None:
            random.seed(self.seed)
        self.sequences = []
        self.storyboard = storyboard
        
        self.state = dict()
        # Adjacency graph for locations
        self.graph = graph if graph else self._create_fully_connected_graph()
        self.possible_moves = {p: self.graph[locations[-1]] for p in people}

        # Simulation time step
        self.time_step = 0
        # Observation steps
        self.events = storyboard.events if storyboard is not None else {}
        self.manual_actions = storyboard.manual_actions if storyboard is not None else {}
        
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
        if self.manual_actions:
            manual_actions = {t: f"{self.manual_actions[t]['action']}" for t in self.manual_actions}
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
                required_events[t] = ev['actors']
                knuth[t] = -102
                start = t + 1
        # Generation phase
        sequences = []
        if self.events:
            event_list = iter(sorted(self.events.keys()))
            next_event = next(event_list)        
            paths, indices = None, None
        for i in knuth:
            # Cross paths
            if i > 0 :
                if paths == None: 
                    # Index of a person to make cross paths
                    paths = required_events[next_event]
                    indices = [0] * len(paths)
                actor = self.events[next_event]['actors'][i-1]
                sequences.append(self.event_statement(actor, paths[i-1][indices[i-1]]))
                self.update_state(actor, paths[i-1][indices[i-1]])
                indices[i-1] += 1
                self.time_step += 1    
            elif i == -100:
                # Last step of cross paths
                new_loc = self.events[next_event]['location']
                actor = self.events[next_event]['actors'][-1]
                sequences.append(self.event_statement(actor, new_loc[-1]))
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
                # Mislead
                choices = []
                mislead_person = required_events[next_event][0:-1]
                poi = required_events[next_event][-1]
                location_set = {self.current_locations[p][-1] for p in mislead_person}
                choices = [l for l in self.possible_moves[poi] if l not in location_set]
                new_loc = random.choice(choices)
                sequences.append(self.event_statement(poi, new_loc))
                self.update_state(poi, new_loc)
                self.time_step += 1
                try:
                    next_event = next(event_list)
                except:
                    # This means we're done
                    continue
            elif i == -102:
                # Move
                new_loc = self.events[next_event]['location'][0]
                if self.events[next_event]['name'] != 'move':
                    print(self.events[next_event])
                actor = self.events[next_event]['actors'][0]
                sequences.append(self.event_statement(actor, new_loc))
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
                sequences.append(self.event_statement(person, loc))
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
                sequences.append(self.event_statement(person, loc))
                self.update_state(person, loc)
                self.time_step += 1
            if self.manual_actions and self.time_step - 1 in manual_actions:
                sequences.append(f'*{manual_actions[self.time_step-1]}') 
        return sequences
    
    
    '''
    Converts the events in the form of r(c, L) to a more natural sentence. Redefine this function as needed for custom parsing.
    '''
    def formal_to_story(self, sequence_list: list[str], custom_function: callable = None):
        strings = []
        for e in sequence_list:
            if custom_function is not None:
                strings.append(custom_function(e))
                continue
            # Basic parser used for experiments
            if e[0] == '*':
                strings.append(e[1:])
            else:
                e = e.replace('\n','')
                r = e.split('(')[0]
                e = e.replace(f'{r}(','').replace(')','')
                subject, loc, time = e.split(',')
                res = f'{subject} {r.replace("_", " ") if "hole" in loc else r.replace("_", " ").replace("in","in") }{loc}'
                strings.append(res)
        return '. '.join(strings).strip()
    
    def event_statement(self, subject, object):
        if isinstance(self.action, str):
            return f"{self.action}({subject}, {object}, {self.time_step})\n"
        elif  isinstance(self.action, list):
            action = random.choice(self.action)
            return f"{action}({subject}, {object}, {self.time_step})\n"
        else:
            raise ValueError("action must be a string or a list of strings.")
