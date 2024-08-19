import numpy as np
from env import Mendikot
from MACROS import *
from tqdm import *
import matplotlib.pyplot as plt

class LinearFunctionApproximator:
    def __init__(self, state_dim, action_dim, learning_rate=0.3):
        self.weights = np.zeros((action_dim, state_dim))
        self.learning_rate = learning_rate
    
    def predict(self, state):
        vect = np.dot(self.weights, state.flatten())
        return vect
    
    def update(self, state, action, target, init_agent_cards):
        # print(action)
        action_index = np.where(action == init_agent_cards)[0][0]
        # print(action_index)
        self.weights[action_index,:] += self.learning_rate * (target - self.predict(state)[action_index]) * state.flatten()

def epsilon_greedy_policy(m: Mendikot, state, function_approximator, epsilon, init_agent_cards):
    '''
    Returns action according to the epsilon_greedy policy
    '''
    if np.random.rand() < epsilon:
        action = np.random.choice(m.get_available_cards(AGENT))
        return action
    else:
        available_cards = m.get_available_cards(AGENT)
        mask = np.isin(init_agent_cards ,available_cards) 
        pred_array = function_approximator.predict(state)
        action = int(np.argmax(pred_array[mask]))
        return init_agent_cards[action]


def play_random():
    pass

def train_agent(m: Mendikot, function_approximator: LinearFunctionApproximator, num_episodes, discount_factor=0.99, epsilon=0.01):

    players = [AGENT, OPPONENT_1, TEAMMATE, OPPONENT_2]
    episode_rewards = 0
    rewards = []
    for episode in trange(num_episodes):
        
        # Starting new game 
        # Select random player to start new game
        # TO-DO : Can change this to winner of previous game in future versions

        next_player = np.random.choice(len(players))
        state = m.reset()
        done = False
        

        init_agent_cards = m.get_cards_in_hand(AGENT)
        while not done:
            # New Trick starts here

            for iter in range(GAME_PLAYERS):
                player = (next_player+iter) % 4

                if player == AGENT:
                    # Epsilon-greedy policy
                    action = epsilon_greedy_policy(m=m, state=state, function_approximator=function_approximator, epsilon=epsilon, init_agent_cards=init_agent_cards)
                    agent_action = action
                    agent_state = state
                else:
                    action = np.random.choice(m.get_available_cards(player))

                next_state, reward, done, winner_info = m.step(action,player)
                
            
                if reward != None:
                    
                    episode_rewards += reward
                    # Compute TD target
                    td_target = reward + discount_factor * np.max(function_approximator.predict(next_state))
                
                    # Update function approximator
                    function_approximator.update(agent_state, agent_action, td_target, init_agent_cards)
            
            next_player = int(winner_info[0])
            state = next_state

        rewards.append(episode_rewards)
        # print(f"Episode {episode} Rewards : {episode_rewards}")
        episode_rewards = 0

    return rewards, function_approximator.weights


if __name__ == "__main__":
    
    m = Mendikot(cards_per_player=4)
    _ = m.reset()
    
    state_dim = m.get_state().flatten().shape[0]
    # print(state_dim)
    action_dim = m.cards_per_player
    # print(action_dim)
    
    # Initialize function approximator
    function_approximator = LinearFunctionApproximator(state_dim, action_dim)
    
    # # Train the agent
    rewards, weights = train_agent(m, function_approximator, num_episodes=100000)

    rewards = np.array(rewards)
    rewards = rewards * -1
    # print(weights[0].reshape(9,16))
    plt.hist(rewards)
    plt.title("Reward Distribution")
    plt.xlabel("Rewards")
    plt.ylabel("Frequency")
    plt.show()