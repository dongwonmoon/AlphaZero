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
    def __init__(self, state, parent=None, prior=0, cur_p=1):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.cur_p = cur_p

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
                best_score = ucb_score
                best_child = child
        return best_child

    def select_action(self, temperature):
        visit_counts = np.array([child.visits for child in self.children])
        actions = [children.state.board.peek() for children in self.children]
        if temperature == 0:
            action_idx = np.argmax(visit_counts)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution /= visit_count_distribution.sum()
            action_idx = np.random.choice(len(actions), p=visit_count_distribution)
        return action_idx

    def expand(self, policy, cur_p):
        self.cur_p = cur_p
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
            child_node = Node(new_state, parent=self, prior=prob, cur_p=cur_p * -1)
            self.children.append(child_node)

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, model, num_simulations=5, temperature=1):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature

    def search(self, initial_state, cur_p):
        root = Node(initial_state)
        root.expand([1] * 4032, cur_p)  # 수정 필요

        for _ in range(self.num_simulations):
            node = root
            while not node.state.is_game_over() and node.is_fully_expanded():
                node = node.select_child()

            if not node.state.is_game_over():
                board_state = node.state.get_board_state()
                board_state = board_state[np.newaxis, ...]
                policy, _ = self.model(board_state)
                policy = policy.squeeze().detach().numpy()
                node.expand(policy, node.parent.cur_p * -1)

            value = self.simulate(node.state, cur_p)
            node.backpropagate(value)

        action_idx = root.select_action(self.temperature)
        best_child = root.children[action_idx]
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
    mcts = MCTS(model, num_simulations=10, temperature=1)
    best_move = mcts.search(game, 1)

    print(f"\nBest move: {best_move}")
