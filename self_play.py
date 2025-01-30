import numpy as np
from mcts import MCTS


class SelfPlay:
    def __init__(self, model, game, num_simulations):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations

    def play(self):
        mcts = MCTS(self.model, self.num_simulations)
        cur_p = 1
        states, policies, rewards = [], [], []

        while not self.game.is_game_over():
            state = self.game.get_board_state()
            action, policy = mcts.search(self.game, cur_p)
            states.append(state)
            policies.append(policy)
            self.game = self.game.apply_move(action)
            cur_p = 0 if cur_p == 1 else 1
            print(self.game)
        winner = self.game.get_result(1)
        rewards = [winner for i in range(len(states))]
        return states, policies, rewards
