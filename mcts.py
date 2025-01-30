import math
import random
from game import ChessGame
import torch
import numpy as np
import itertools

FILES = "abcdefgh"
RANKS = "12345678"

# 64개 위치를 모두 생성
SQUARES = [f + r for f, r in itertools.product(FILES, RANKS)]

# 모든 가능한 움직임 조합 생성
ALL_MOVES = [
    from_sq + to_sq for from_sq in SQUARES for to_sq in SQUARES if from_sq != to_sq
]

PROMOTIONS = ["q", "r", "b", "n"]
ALL_MOVES += [move + promo for move in ALL_MOVES for promo in PROMOTIONS]

uci_to_index = {move: i for i, move in enumerate(ALL_MOVES)}


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

    def expand(self, policy):
        legal_moves = self.state.get_legal_moves()
        uci_idx = [
            (
                uci_to_index.get(move.uci()[:-1], -1)
                if len(move.uci()) > 4
                else uci_to_index.get(move.uci(), -1)
            )
            for move in legal_moves
        ]

        move_probs = {move: policy[idx] for move, idx in zip(legal_moves, uci_idx)}
        for move in legal_moves:
            prob = move_probs.get(move, 0)
            new_state = self.state.apply_move(move)
            child_node = Node(new_state, parent=self, prior=prob)
            self.children.append(child_node)

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, model, num_simulations=5):
        self.model = model
        self.num_simulations = num_simulations

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
                node.expand(policy)

            value = self.simulate(node.state, cur_p)
            node.backpropagate(value)

        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.state.board.peek(), policy

    def simulate(self, state, cur_p):
        while not state.is_game_over():
            board_state = state.get_board_state()
            board_state = board_state[np.newaxis, ...]
            policy, value = self.model(board_state)
            policy = policy.squeeze().detach().numpy()
            legal_moves = state.get_legal_moves()

            uci_idx = [
                (
                    uci_to_index.get(move.uci()[:-1], -1)
                    if len(move.uci()) > 4
                    else uci_to_index.get(move.uci(), -1)
                )
                for move in legal_moves
            ]

            move_probs = {move: policy[idx] for move, idx in zip(legal_moves, uci_idx)}
            move = max(move_probs, key=move_probs.get)
            state = state.apply_move(move)
        return state.get_result(cur_p)


if __name__ == "__main__":
    from model import AlphaZeroNet

    game = ChessGame()
    print("Initial board:")
    print(game)

    model = AlphaZeroNet(8, 4032, 5, 64, 16)
    mcts = MCTS(model, num_simulations=10)
    best_move = mcts.search(game, 1)

    print(f"\nBest move: {best_move}")
