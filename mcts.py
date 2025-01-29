import math
import random
from game import ChessGame
import torch
import numpy as np


class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = prior

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def select_child(self):
        best_child = None
        best_score = -float("inf")

        for child in self.children:
            prior_score = child.prior * math.sqrt(self.visits) / (child.visits + 1)
            if child.visits > 0:
                value_score = child.value / child.visits
            else:
                value_score = 0

            ucb_score = value_score + prior_score

            if ucb_score > best_score:
                if isinstance(ucb_score, torch.Tensor):
                    ucb_score = ucb_score.item()
                best_score = ucb_score
                best_child = child

        return best_child

    def expand(self, policy, temperature):
        legal_moves = self.state.get_legal_moves()
        move_probs = {move: prob for move, prob in zip(legal_moves, policy)}
        for move in legal_moves:
            prob = move_probs.get(move, 0)
            try:
                action = self.select_action(move_probs, temperature)
            except:
                action = move
            new_state = self.state.apply_move(action)
            child_node = Node(new_state, parent=self, prior=prob)
            self.children.append(child_node)

    def select_action(self, temperature):
        visit_counts = np.array([child.visits for child in self.children])
        actions = [child.state.last_move for child in self.children]

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution /= visit_count_distribution.sum()
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, model, num_simulations=800, temperature=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature

    def search(self, initial_state, cur_p):
        root = Node(initial_state)
        for _ in range(self.num_simulations):
            node = root

            while not node.state.is_game_over() and node.is_fully_expanded():
                node = node.select_child()

            if not node.state.is_game_over():
                board_state = node.state.get_board_state()
                board_state = board_state[np.newaxis, ...]
                policy, _ = self.model(board_state)
                policy = policy.squeeze().detach().numpy()
                node.expand(policy, self.temperature)

            value = self.simulate(node.state, cur_p)
            node.backpropagate(value)

        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.state.board.peek(), policy

    def simulate(self, state, cur_p):
        while not state.is_game_over():
            board_state = state.get_board_state()
            board_state = board_state[np.newaxis, ...]
            policy, value = self.model(board_state)
            legal_moves = state.get_legal_moves()
            move_probs = {
                move: prob
                for move, prob in zip(legal_moves, policy.squeeze().detach().numpy())
            }
            move = max(move_probs, key=move_probs.get)
            state = state.apply_move(move)
        return state.get_result(cur_p)


if __name__ == "__main__":
    from model import AlphaZeroNet

    game = ChessGame()
    print("Initial board:")
    print(game)

    model = AlphaZeroNet(8, 4672)
    mcts = MCTS(model, num_simulations=100)
    best_move = mcts.search(game)

    print(f"\nBest move: {best_move}")
