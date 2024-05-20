import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

# Parameters for holding time range
min_ht = 10
max_ht = 20


class NetworkEnv(gym.Env):
    def __init__(self) -> None:
        super(NetworkEnv, self).__init__()
        
        self.graph = nx.read_gml('nsfnet.gml')
        
        # Reset the graph capacities to 10 for all edges
        for u, v in self.graph.edges():
            self.graph[u][v]['capacity'] = 10
        
        # Create a mapping from node names to numerical IDs
        name_to_id = {name: idx for idx, name in enumerate(self.graph.nodes())}
        self.id_to_name = {idx: name for name, idx in name_to_id.items()}  # Reverse mapping for lookup by ID

        # Relabel the graph nodes with numerical IDs
        self.graph = nx.relabel_nodes(self.graph, name_to_id)


        
        
        self.observation_space = spaces.Dict({
            "links": spaces.Box(low=-100, high=100, shape=(len(self.graph.edges()), 10), dtype=np.float32),
            "source": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
            "destination": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
            "holding_time": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        })

        self.action_space = spaces.Discrete(3)  # 0: block, 1: use path1, 2: use path2

        # Initialize the state
        self.state = {
            "links":  np.zeros((len(self.graph.edges()), 10), dtype=np.float32),
            "source": 7,
            "destination": 1,
            "holding_time": np.random.randint(min_ht, max_ht)

        }
        
        self.current_round = 0
    
    def _get_obs(self):
        
        obs = {
            'links': self.state['links'],
            'source': np.array([self.state["source"]]),
            'destination': np.array([self.state["destination"]]),
            'holding_time': np.array([self.state['holding_time']])
        }
        
        return obs


        
    def find_edge_index(self, u, v):
        edge_list = list(self.graph.edges())
        try:
            return edge_list.index((u, v))
        except ValueError:
            return edge_list.index((v, u))

    def _generate_req(self):
        source =  np.random.randint(len(self.graph.nodes))
        destination = np.random.randint(len(self.graph.nodes))
        holding_time = np.random.randint(min_ht, max_ht)
        return source, destination, holding_time
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the state
        
        self.state["source"] = 7
        self.state["destination"] = 1
        '''
        self.state["source"] = 7  
        self.state["destination"] = 1
        '''
        
        while self.state["destination"] == self.state["source"]:
            self.state["destination"] = np.random.randint(len(self.graph.nodes))
        
        # Sample a random holding time from 10 to 20
        #self.state["holding_time"] = np.random.randint(10, 21)
        self.state["holding_time"] = np.random.randint(min_ht, max_ht)

        
        # Reset the graph capacities to 10 for all edges
        for u, v in self.graph.edges():
            self.graph[u][v]['capacity'] = 10
        
        # Update the links array to reflect the new capacities
        self.state["links"] = np.zeros((len(self.graph.edges()), 10), dtype=np.float32)

        self.current_round = 0
        obs = self._get_obs()
        info = {}
    
        return obs, info  # Return the observation as a dictionary
    


    def step(self, action):
        source = self.state["source"]
        destination = self.state["destination"]
        holding_time = self.state["holding_time"]

        self.state["links"] -= np.where(self.state["links"] != 0, 1, 0)
        

        for i, edge in enumerate(self.graph.edges()):
            num_zeros = np.count_nonzero(self.state["links"][i] == 0)
            self.graph.edges[edge]["capacity"] = num_zeros
    
        
        self.current_round += 1
        print(self.current_round)
        terminated = (self.current_round == 100)
        
        paths = list(nx.shortest_simple_paths(self.graph, source, destination))
        if len(paths) < 2:
            path1, path2 = paths[0], None
        else:
            path1, path2 = paths[:2]
        
        chosen_path = path1 if action == 1 and path1 is not None else path2
        
        
        reward = -1
        
        
        if action == 0:
            reward = -1
        elif action == 1:
            chosen_path = path1
            #print(path1)
            path_capacities = [self.graph[u][v]['capacity'] for u, v in zip(chosen_path[:-1], chosen_path[1:])]
            if all(cap > 0 for cap in path_capacities):
                            
                #reward = sum(path_capacities) / (holding_time * len(chosen_path) * len(chosen_path))
                for u, v in zip(chosen_path[:-1], chosen_path[1:]):
                    self.graph[u][v]['capacity'] -= 1
                    
                    edge_index = self.find_edge_index(u, v)

                    
                    # Update the corresponding element in self.state["links"]
                    zero_indices = np.where(self.state["links"][edge_index] == 0)[0]
    
                    if len(zero_indices) > 0:
                        # Update the first zero index with self.state["holding_time"]
                        self.state["links"][edge_index][zero_indices[0]] = holding_time
                        
                utilizations = []
                for u, v in zip(chosen_path[:-1], chosen_path[1:]):
                    edge_index = self.find_edge_index(u, v)
                    link_states = self.state["links"][edge_index]
                    utilization = np.count_nonzero(link_states) / len(link_states)
                    utilizations.append(utilization)
                
                if utilizations:
                    avg_utilization = np.mean(utilizations)
                else:
                    avg_utilization = 0

                if avg_utilization > 0:
                    reward = 2 / avg_utilization
                else:
                    reward = -1
                #print('Average Utilization', avg_utilization)    
            else:
                reward = -1
                   
                        
                        
            
        elif action == 2:
            chosen_path = path2
            if path2 == None:
                reward = -2
            else:
                #print(path2)
                path_capacities = [self.graph[u][v]['capacity'] for u, v in zip(chosen_path[:-1], chosen_path[1:])]
                if all(cap > 0 for cap in path_capacities):
                    #reward = sum(path_capacities) / (holding_time * len(chosen_path) * len(chosen_path))
                
                    for u, v in zip(chosen_path[:-1], chosen_path[1:]):
                        self.graph[u][v]['capacity'] -= 1
            
                        edge_index = self.find_edge_index(u, v)
                        
                        # Update the corresponding element in self.state["links"]
                        zero_indices = np.where(self.state["links"][edge_index] == 0)[0]
        
                        if len(zero_indices) > 0:
                            # Update the first zero index with self.state["holding_time"]
                            self.state["links"][edge_index][zero_indices[0]] = holding_time
                    
                    utilizations = []
                    for u, v in zip(chosen_path[:-1], chosen_path[1:]):
                        edge_index = self.find_edge_index(u, v)
                        link_states = self.state["links"][edge_index]
                        utilization = np.count_nonzero(link_states) / len(link_states)
                        utilizations.append(utilization)
                    
                    if utilizations:
                        avg_utilization = np.mean(utilizations)
                    else:
                        avg_utilization = 0

                    if avg_utilization > 0:
                        reward = 2 / avg_utilization
                    else:
                        reward = -1
                    #print('Average Utilization', avg_utilization)
                
                else:
                    reward = -1

                


        # Assuming self.state["links"] contains the matrix
        matrix = np.array(self.state["links"])  # Create a deep copy of the original matrix

        # Replace all non-zero values with 1 in the matrix
        matrix[matrix != 0] = 1

        # Count the number of non-zero values (i.e., count of 1's)
        num_non_zero_values = np.sum(matrix)

        # Total number of elements in the matrix
        total_elements = matrix.size

        # Calculate the average
        average = num_non_zero_values / total_elements

            
        
        # Write averages to a separate file
        with open("averages.txt", "a") as f:
            f.write(",".join(map(str, [average])) + "\n")

        
        observation = self._get_obs()
        info = {}
        
        self.state["source"] , self.state["destination"] , self.state["holding_time"] = self._generate_req()
            
        
        return observation, reward, terminated, terminated, info



def main():
    # Initialize the environment
    env = NetworkEnv()
    
    # Seed the environment for reproducibility
    seed = 42
    obs, _ = env.reset(seed=seed)
    
    print("Initial observation after reset:")
    print(obs)

    # Convert single-element numpy arrays to scalars
    obs_source = obs['source'].item() if obs['source'].size == 1 else tuple(obs['source'])
    obs_destination = obs['destination'].item() if obs['destination'].size == 1 else tuple(obs['destination'])
    obs_holding_time = obs['holding_time'].item() if obs['holding_time'].size == 1 else tuple(obs['holding_time'])

    print(f"Source: {env.id_to_name[obs_source]}")
    print(f"Destination: {env.id_to_name[obs_destination]}")
    print(f"Holding Time: {obs_holding_time}")

    
    # Define the number of steps you want to run for debugging
    num_steps = 200
    
    for step in range(num_steps):
        print(f"\nStep {step + 1}/{num_steps}")
        
        # Sample a random action
        action = env.action_space.sample()
        print(f"Action taken: {action}")
        
        # Take the action and get the result
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("Observation:")
        print(obs)
        # Convert single-element numpy arrays to scalars
        obs_source = obs['source'].item() if obs['source'].size == 1 else tuple(obs['source'])
        obs_destination = obs['destination'].item() if obs['destination'].size == 1 else tuple(obs['destination'])

        print(f"Source: {env.id_to_name[obs_source]}")
        print(f"Destination: {env.id_to_name[obs_destination]}")

        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        
        if terminated or truncated:
            print("Episode finished.")
            break

if __name__ == "__main__":
    main()

