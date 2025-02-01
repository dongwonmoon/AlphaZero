import math
import random
import torch
import numpy as np
import itertools
from game import ChessGame

FILES = "abcdefgh"
RANKS = "12345678"

# 64 positions on a chess board
SQUARES = [f + r for f, r in itertools.product(FILES, RANKS)]

# All possible moves (including promotional moves)
ALL_MOVES = [
    from_sq + to_sq for from_sq in SQUARES for to_sq in SQUARES if from_sq != to_sq
]
PROMOTIONS = ["q", "r", "b", "n"]
ALL_MOVES += [move + promo for move in ALL_MOVES for promo in PROMOTIONS]

# Map UCI string to index
uci_to_index = {move: i for i, move in enumerate(ALL_MOVES)}

class Node:
    """
    A node in the MCTS search tree.

    Attributes:
        state (ChessGame): The current chess state (using the organization's ChessGame class).
        parent (Node): Parent node, or None if root.
        children (List[Node]): Collection of child nodes.
        visits (int): Number of times this node has been visited.
        value (float): Accumulated value (win/loss from the perspective of cur_p).
        prior (float): Prior probability (from the policy network).
        cur_p (int): Current player's turn indicator (0=White, 1=Black).
    """

    def __init__(self, state: ChessGame, cur_p: int, parent=None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.cur_p = cur_p

    def is_fully_expanded(self) -> bool:
        """
        Returns True if all legal moves for this state have been expanded into child nodes.
        """
        return len(self.children) == len(self.state.get_legal_moves())

    def select_child(self, c_puct: float) -> "Node":
        """
        Select the best child node based on the UCB formula. The formula is:
            Q + cPUCT * P * (sqrt(N) / (1 + n))
        where:
          Q = value_score = child.value / child.visits (if visits>0 else 0)
          P = child.prior
          N = self.visits (parent visits)
          n = child.visits

        Args:
            c_puct (float): The UCB exploration parameter.

        Returns:
            Node: The child node with the highest UCB score.
        """
        best_child = None
        best_score = -float("inf")

        for child in self.children:
            if child.visits > 0:
                value_score = child.value / child.visits
            else:
                value_score = 0.0

            prior_score = c_puct * child.prior * math.sqrt(self.visits) / (child.visits + 1)
            ucb_score = value_score + prior_score

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def select_action(self, temperature: float) -> int:
        """
        After MCTS simulations, select which child node to pick as the next move.

        Args:
            temperature (float): If 0, pick the child with the most visits.
                                 Otherwise, sample from the visit counts with exponent 1/temperature.

        Returns:
            int: The index of the chosen child node.
        """
        visit_counts = np.array([child.visits for child in self.children])
        if temperature == 0:
            # Greedy selection
            return int(np.argmax(visit_counts))
        else:
            # Softmax-style sampling from visit counts
            exponentiated = visit_counts ** (1.0 / temperature)
            distribution = exponentiated / exponentiated.sum()
            return int(np.random.choice(len(self.children), p=distribution))

    def expand(self, policy: np.ndarray) -> None:
        """
        Expand this node by creating child nodes for every legal move from this state.

        Args:
            policy (np.ndarray): The prior distribution over all possible moves (shape = [action_size]).
        """
        legal_moves = self.state.get_legal_moves()

        # Map each legal move to an index in the policy array
        move_indices = []
        for move in legal_moves:
            # For promotions, strip the last character from move.uci() if length > 4
            uci_str = move.uci()
            if len(uci_str) > 4:
                index = uci_to_index.get(uci_str[:-1], -1)
            else:
                index = uci_to_index.get(uci_str, -1)
            move_indices.append(index)

        move_probs = {move: policy[idx] for move, idx in zip(legal_moves, move_indices)}

        for move in legal_moves:
            prob = move_probs.get(move, 0.0)
            # Apply the move to get a new state
            new_state = self.state.apply_move(move)
            # Switch player
            next_p = 1 if self.cur_p == 0 else 0

            child_node = Node(new_state, cur_p=next_p, parent=self, prior=prob)
            self.children.append(child_node)

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate the simulation result. Updates the value of this node (and its ancestors).
        In a two-player zero-sum game, we invert the value sign at each level so that
        a win for one is a loss for the other.

        Args:
            value (float): Result from the perspective of the current player. (+1=win, 0=draw, -1=loss)
        """
        self.visits += 1
        self.value += value
        if self.parent is not None and not isinstance(self.parent, (int, float)):
            # Flip the sign of value since the parent's perspective is the opponent
            self.parent.backpropagate(-value)

class MCTS:
    """
    Monte Carlo Tree Search wrapper for an AlphaZero-like model.

    Attributes:
        model (nn.Module): The neural network (policy, value) model.
        num_simulations (int): The number of MCTS rollouts to perform.
        temperature (float): Temperature for action selection.
        c_puct (float): Exploration parameter for UCB formula.
    """

    def __init__(self, model: torch.nn.Module, num_simulations: int = 5,
                 temperature: float = 1.0, c_puct: float = 1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.c_puct = c_puct

    def search(self, initial_state: ChessGame, cur_p: int = 0) -> tuple:
        """
        Perform MCTS by creating a root node and exploring using the neural network.

        Args:
            initial_state (ChessGame): The initial board state.
            cur_p (int): Current player index (0=White, 1=Black).

        Returns:
            A tuple (move, prior, policies):
                move (chess.Move): The best move decided by MCTS.
                prior (float): The prior probability assigned by the network for this move.
                policies (List[np.ndarray]): A list of policy vectors used during expansions.
        """
        # Create root node
        root_node = Node(initial_state, cur_p=cur_p)

        # Get policy from the model for the root state
        root_policy, _ = self._evaluate_state(root_node.state)
        root_node.expand(root_policy)

        for _ in range(self.num_simulations):
            node = root_node
            # Traverse until we find a leaf node or a terminal state
            while not node.state.is_game_over() and node.is_fully_expanded():
                node = node.select_child(self.c_puct)

            # If the game isn't over, expand this node
            if not node.state.is_game_over():
                new_policy, _ = self._evaluate_state(node.state)
                node.expand(new_policy)

            # Simulate to get a game result
            value = self._simulate(node.state, node.cur_p)
            # Backpropagate result
            node.backpropagate(value)

        # Choose the final move from root's children
        best_child_idx = root_node.select_action(self.temperature)
        best_child = root_node.children[best_child_idx]

        return best_child.state.board.peek(), root_policy

    def _simulate(self, state: ChessGame, cur_p: int) -> float:
        """
        Simulate the game until it ends, always choosing the move with the highest probability.

        Args:
            state (ChessGame): The current game state.
            cur_p (int): Current player's turn index.

        Returns:
            float: +1 if cur_p eventually wins, -1 if loss, 0 if draw.
        """
        while not state.is_game_over():
            # Evaluate state and choose the move with the largest probability
            policy, _ = self._evaluate_state(state)
            legal_moves = state.get_legal_moves()

            move_indices = []
            for move in legal_moves:
                uci_str = move.uci()
                if len(uci_str) > 4:
                    index = uci_to_index.get(uci_str[:-1], -1)
                else:
                    index = uci_to_index.get(uci_str, -1)
                move_indices.append(index)

            move_probs = {move: policy[idx] for move, idx in zip(legal_moves, move_indices)}
            best_move = max(move_probs, key=move_probs.get)
            state = state.apply_move(best_move)

        return state.get_result(cur_p)

    def _evaluate_state(self, state: ChessGame) -> tuple:
        """
        Forward pass through the model to get (policy, value) predictions.

        Args:
            state (ChessGame): Current board state.

        Returns:
            (np.ndarray, float): (policy vector, value)
        """
        board_state = state.get_board_state()  # shape (8,8,14) typically
        board_state = board_state[np.newaxis, ...]  # shape (1,8,8,14)
        policy, value = self.model(board_state)
        policy = policy.squeeze().detach().cpu().numpy()  # shape (action_size,)
        # value = value.squeeze().detach().cpu().numpy()  # shape (1,) if needed
        return policy, value

if __name__ == "__main__":
    from model import AlphaZeroNet

    # Example usage
    game = ChessGame()
    print("Initial board:")
    print(game)

    model = AlphaZeroNet(board_size=8,
                         num_ware=14,
                         action_size=4032,
                         num_res_blocks=5,
                         in_channels=64,
                         mid_channels=16)
    # Instantiate MCTS
    mcts = MCTS(model=model, num_simulations=1, temperature=1, c_puct=1.5)

    # Search for the best move from White's perspective
    chosen_move, policy = mcts.search(game, cur_p=0)
    print(f"\nChosen move: {chosen_move}, Prior: {policy}")