import numpy as np
from mcts import MCTS


class SelfPlay:
    def __init__(self, model, game, num_simulations, temperature=1.0):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.temperature = temperature

    def play(self):
        mcts1 = MCTS(self.model, self.num_simulations, self.temperature)
        mcts2 = MCTS(self.model, self.num_simulations, self.temperature)
        states, policies, rewards = [], [], []
        curr_p = 1

        while not self.game.is_game_over():
            state = self.game.get_board_state()
            mcts = mcts1 if curr_p == 1 else mcts2
            action, policy = mcts.search(self.game)
            states.append(state)
            policies.append(policy)
            self.game = self.game.apply_move(action)
            curr_p = -curr_p
            print(self.game)
        winner = self.game.get_result(1)
        rewards = [winner if i % 2 == 0 else -winner for i in range(len(states))]
        print(rewards)
        return states, policies, rewards
