import env
from MACROS import *
import os

def print_cards(m:env.Mendikot):
    os.system("cls")
    for p in range(AGENT, OPPONENT_2+1):
        cards = m.get_available_cards(p)
        render_str = m.get_render_str(cards_list=cards)
        print(f"{PLAYER_RENDER[p]:}\t{render_str}")
    print(f"TRUMP\t{SUITS_RENDER[m.trump_suit]}")
    print("\n ----------------------- \n")

def main():
    m = env.Mendikot(cards_per_player = 13)
    players = [AGENT, OPPONENT_1, TEAMMATE, OPPONENT_2]
    _ = m.reset()

    print("Starting Game")
    print_cards(m)
    
    terminated = False
    next_player = np.random.choice(len(players))

    while not terminated:
        input("Press Enter to continue..")
        os.system("cls")
        print(f"New Trick (TRUMP: {SUITS_RENDER[m.trump_suit]})")
        for iter in range(4):
            player = (next_player+iter) % 4            
            if player == AGENT:
                cards = m.get_available_cards(player)
                
                try:
                    inp = abs(int(input(f'Enter card choice [{m.get_render_str(cards_list=cards)}]: ')))
                    choice = cards[inp]
                except IndexError:
                    print("Invalid input! playing the last card")
                    choice = cards[-1]

            else:
                choice = np.random.choice(m.get_available_cards(player))
                print(choice)


            render_str = m.get_render_str(card_idx=choice)
            print(f"{PLAYER_RENDER[player]:}\t{render_str}")

            _, reward, terminated, winner_info = m.step(choice,player)
            

        print(f"Reward: {reward} Trick Winner: {PLAYER_RENDER[winner_info[0]]}")
        next_player = int(winner_info[0])

    print(f"Game Winner: {PLAYER_RENDER[m.get_game_winner()]}")

if __name__ == "__main__":
    main()