import numpy as np
import env
from MACROS import AGENT, TEAMMATE
import copy 

class Node():
    def __init__(self, p:int, state:np.ndarray, available_actions:list[int]) -> None:
        self.player = p
        self.state = state
        self.available_actions = available_actions
        self.parent: Node = None
        self.children: dict[int:Node] = {a:None for a in self.available_actions}
        self.sims = 0
        self.returns = 0
        self.is_leaf = False


class MCTS():
    def __init__(self, sim_env:env.Mendikot, c:int = 0.1, n:int = 2) -> None:
        self.c = c
        self.n = n
        self.env = copy.deepcopy(sim_env)
        self.temp_env = copy.deepcopy(sim_env)
        
        
    @staticmethod
    def is_fully_explored(node:Node):
        return False if None in node.children.values() else True
    
    def get_next_player(self, player:int):
        return ((player + 1) % self.env.num_players)
    
    @staticmethod
    def get_parent_sims(node:Node):
        return 0 if node.parent == None else node.parent.sims
    
    def get_best_action(self, node:Node):
        best_returns = -np.inf
        for action, n in node.children.items():
            # print(self.temp_env.get_render_str(card_idx=action), n.returns)
            if n.returns > best_returns:
                best_returns = n.returns
                best_action = action
        
        return best_action
    
    def reset_env(self):
        # Need to reset the environment after one pass of MCTS
        del self.temp_env
        self.temp_env = copy.deepcopy(self.env)


    def calculate_ucb(self, node:Node, player_type:int):
        return ((node.returns/node.sims) + (self.c*(np.sqrt(np.log(self.get_parent_sims(node))/node.sims)))*player_type)
    
    def select(self, root:Node, player_type:int) -> Node:
        
        curr_node : Node = root 
        terminated = False

        while self.is_fully_explored(curr_node):
            # print(curr_node.children.items())

            ucb_best = -np.inf * player_type
            for action, node in curr_node.children.items():
                ucb = self.calculate_ucb(node, player_type)
                # print(action, ucb)
                
                if ucb > ucb_best:
                    selected_action = action
                    selected_node = node
                    ucb_best = ucb
            
            # print("Selection:\t", curr_node.player)
            
            self.temp_env.get_available_cards(curr_node.player)
            _, _, terminated, _  = self.temp_env.step(selected_action, curr_node.player)

            if terminated:
                break

            curr_node = selected_node
            # print("Explored:\t", self.is_fully_explored(curr_node))

        curr_node.sims += 1
        return curr_node, terminated

    def expand(self, node:Node, player:int):
        # Create a temp env for taking step
        # temp_env = copy.deepcopy(self.env)

        # Random choice for the action
        self.temp_env.get_available_cards(node.player)
        action = np.random.choice([a for a in node.children.keys() if node.children[a] is None])
        # action = np.random.choice(node.available_actions)
        
        # Take the action and observe next state
        # next_player = self.get_next_player(player)
        # next_player = node.player
        # print("Expansion:\t", node.player)
        next_state, reward, _, _, = self.temp_env.step(action, node.player)
        # print(self.temp_env.get_render_str(card_idx=action))
        # print(self.temp_env.get_render_str(cards_list=self.temp_env.curr_trick))

        next_player = self.get_next_player(node.player)
        next_actions = self.temp_env.get_available_cards(next_player)

        # Create a new node for this action adn update parent and children
        new_node = Node(next_player, next_state, next_actions)
        new_node.parent = node
        new_node.sims += 1
        node.children[action] = new_node

        if reward is not None:
            new_node.is_leaf = True
    
        return new_node, reward

    def simulate(self, node:Node):
        reward = None
        player = node.player

        while reward is None:
            # print("Simulation:\t", player)
            
            actions = self.temp_env.get_available_cards(player)
            _, reward, _, _ = self.temp_env.step(np.random.choice(actions), player)
            player = self.get_next_player(player)

        # print("--------------------------------------")
        return reward

    def backpropagate(self, reward, sim_node:Node):
        node = sim_node
        while node.parent is not None:
            node.returns += reward
            node = node.parent

    def run_search(self, state:np.ndarray, player:int) -> Node:
        player_type = 1 if ((player == AGENT) or (player == TEAMMATE)) else -1
        root_node = Node(player, state, self.temp_env.get_available_cards(player))

        for i in range(self.n):
            
            selected_node, terminated = self.select(root_node, player_type)

            if not terminated:
                sim_node, reward = self.expand(selected_node, player)
                
                if not sim_node.is_leaf:
                    reward = self.simulate(sim_node)
                
                self.backpropagate(reward, sim_node)

            self.reset_env()


        return self.get_best_action(root_node)
        # if not sim_node.is_leaf:
        #     self.simulate(sim_node, next_player)

        