import numpy as np
from mcts import MCTS


class SelfPlay:
    def __init__(self, model, game, num_simulations, temperature=1.0):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.temperature = temperature

    def play(self):
        mcts = MCTS(self.model, self.num_simulations, self.temperature)
        states, policies, rewards = [], [], []

        while not self.game.is_game_over():
            state = self.game.get_board_state()
            action, policy = mcts.search(self.game)
            states.append(state)
            policies.append(policy)
            self.game = self.game.apply_move(action)
            print(self.game)
        winner = self.game.get_result(1)
        rewards = [winner for i in range(len(states))]
        return states, policies, rewards
