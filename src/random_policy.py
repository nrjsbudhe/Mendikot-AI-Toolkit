import numpy as np
import env
from MACROS import *
from tqdm import trange
import matplotlib.pyplot as plt

num_trials = 10000

m = env.Mendikot(cards_per_player = 5)

_ = m.reset()
terminated = False
next_player = int(np.random.choice(3))

agent_wins = 0
agent_trick_wins = 0
agent_trick_wins_10 = 0
oppo_trick_wins_10 = 0

for _ in trange(num_trials):
    while not terminated:
        for iter in range(4):
            player = (next_player+iter) % 4

            # All players employ random policy
            choice = np.random.choice(m.get_available_cards(player))

            # Play 1 card - 1 step in the environment
            next_state, reward, terminated, winner_info = m.step(choice,player)

        next_player = int(winner_info[0])
        if winner_info[0] == 0 or winner_info[0] == 2:
            agent_trick_wins += 1

        # Tricks won with 10
        if reward == 5:
            agent_trick_wins_10 += 1

        if reward == -5:
            oppo_trick_wins_10 += 1
        

    if m.get_game_winner() == AGENT:
        agent_wins = agent_wins + 1

    _ = m.reset()
    terminated = False
    next_player = int(np.random.choice(3))

print(f"Percentage Game Win: {agent_wins/num_trials*100.0} %")
print(f"Percentage Trick Wins: {(agent_trick_wins)*100/(num_trials*m.cards_per_player)} %")
print(f"Tricks won with 10: {(agent_trick_wins_10)}")
# print(f"Opponent won with 10: {(oppo_trick_wins_10)}")

categories = ['Games Won', 'Tricks Won', 'Tricks won with 10']

values1 = [agent_wins, agent_trick_wins , agent_trick_wins_10] 
values2 = [num_trials-agent_wins, (num_trials*m.cards_per_player) - agent_trick_wins, oppo_trick_wins_10]  

bar_width = 0.35

x = range(len(categories))

plt.bar(x, values1, width=bar_width, label='Agent Team')
plt.bar([i + bar_width for i in x], values2, width=bar_width, label='Opponent Team')

plt.ylabel('Values')
plt.title('Comparative Bar Graph : Random Play')
plt.xticks([i + bar_width / 2 for i in x], categories)
plt.legend()
plt.show()


