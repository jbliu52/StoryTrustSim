import random
import copy
'''
A class used by a StorySim object to generate stories that satisfy the specified events in `self.events`.

Events are specified in a dictionary of dictionaries, where the keys are the time steps where the events occur,
and the values are dictionaries with keys that correspond to metadata about the event, such as the type of event,
the characters involved, and the location of the event. Each event will need to specifiy the actors that are involved using numbers that correspond to actors, and the storyboard will assign characters to those numbers randomly from the actor pool.

'''

class Storyboard:    
    def __init__(self, action: str, graph: dict, actor_pool: list, story_length: int, events: dict, manual_actions: dict = None, locations: list = []):
        self.action = action
        self.graph = graph
        # TODO: manual_actions
        self.manual_actions = manual_actions if manual_actions is not None else {}
        if graph is not None:
            self.locations = list(graph.keys())
        elif len(locations) > 0:
            self.locations = locations
        else:
            raise ValueError("Either graph or locations must be provided.")
        
        
        self.events = copy.deepcopy(events)
        self.manual_actions = copy.deepcopy(manual_actions) if manual_actions is not None else {}
        self.actor_pool = actor_pool.copy()

        self.actor_mapping = dict()
        self.loc_mapping = dict()
        self.story_length = story_length
        self.events = events
        
        self.fill_storyboard()


    # Fills in the actor names in the events dictionary based on the actor pool. Does the same for locations        
    def fill_storyboard(self):    
        random.shuffle(self.actor_pool)
        graph = None
        if self.graph is None:
            random.shuffle(self.locations)
            # Create a mapping of actor indices to actor names
            self.loc_mapping = {f"{i}": self.locations[i] for i in range(len(self.locations))}
        else:
            # Iterate through the graph to create a valid set of location assignments that is constrained by the graph.
            start_loc = random.choice(self.locations)
            graph = traverse_graph(self.graph, start_loc)
            self.loc_mapping = dict()
            
        self.actor_mapping = {f"{i}": self.actor_pool[i] for i in range(len(self.actor_pool))}
        
        for time_step in self.events.keys():
            # Turn into list of indices as strings
            # self.events[time_step]['actors'] = self.events[time_step]['actors'].split(',')
            self.events[time_step]['actors'] = [self.actor_mapping[i] for i in self.events[time_step]['actors']]
            
            if 'location' in self.events[time_step]:
                # self.events[time_step]['location'] = self.events[time_step]['location'].split(',')
                if self.graph is None:
                    self.events[time_step]['location'] = [self.loc_mapping[label] for label in self.events[time_step]['location']]
                else:
                    for i, label in enumerate(self.events[time_step]['location']):
                        if label not in self.loc_mapping:
                            self.loc_mapping[label] = next(graph)                         # type: ignore
                        self.events[time_step]['location'][i] = self.loc_mapping[label]
            if 'prev' in self.events[time_step]:
                self.events[time_step]['prev'] = self.loc_mapping[self.events[time_step]['prev'][0]]
        # Use the same mappings to construct manual_actions
        for time_step in self.manual_actions.keys():
            self.manual_actions[time_step]['action'] = self.manual_actions[time_step]['action'].split(' ')
            mapped_str = [self.actor_mapping[a] if a in self.actor_mapping else a for a in self.manual_actions[time_step]['action']]
            self.manual_actions[time_step]['action'] = ' '.join(mapped_str)
            
            
            
            
    def __str__(self):
        event_str = '\n'.join([str(item) for item in self.events.items()])
        action_str = '\n'.join([str(item) for item in self.manual_actions.items()])
        return f"Events: {event_str} \n manual_actions: {action_str}"

        
    def __len__(self):
        return len(self.events)
        
'''
Graph functions
'''
def traverse_graph(g: dict, start):
    """Lightweight DFS traversal generator"""
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            yield vertex
            # Add neighbors to stack
            stack.extend(g.get(vertex, []))

def traverse_graph_bfs(g: dict, start):
    """Lightweight BFS traversal generator"""
    visited = set()
    queue = [start]
    
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            yield vertex
            # Add neighbors to queue
            queue.extend(g.get(vertex, []))        
    


# # Testing

# graph = { 
#     "room_1": ["room_2", "the_hallway"],
#     "room_2": ["room_1", "the_hallway"],
#     "the_hallway": ["room_1", "room_2"]
# }

# actor_pool = ["Alice", "Bob", "Charlie", "Diana"]
# story_length = 10
# events = {
#     1: {"name": "cross_paths", "actors": "0,1", "location": "1", "path_type": "same"},
#     2: {"name": "move", "actors": "1", "location": "2"},
#     3: {"name": "exclusive_random", "actors": "0,1", "stop": 5},
#     5: {"name": "exclusive_random", "actors": "0,1", "stop": story_length-1},
#     story_length-1: {"name":"move", "actors": "3", "location": "3"}
# }
# manual_actions = {2: {'action': "1 places a marble in the basket, 0 empties the basket and places a figurine inside"}}
# storyboard = Storyboard("enter", graph, actor_pool, story_length, events, manual_actions=manual_actions)
# print(storyboard)