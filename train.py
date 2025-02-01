import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader, TensorDataset

from self_play import SelfPlay
from game import ChessGame
from model import AlphaZeroNet


class AlphaZeroTrainer:
    """
    A refactored trainer for an AlphaZero-like approach, incorporating A2C elements such as advantage and entropy bonus.

    Attributes:
        board_size (int): Board dimension (for chess, usually 8).
        num_ware (int): Number of channels in board representation (e.g., 14 for chess).
        action_size (int): Number of possible actions.
        num_simulations (int): Number of MCTS rollouts per move.
        temperature (float): Exploration parameter for MCTS.
        gamma (float): Discount factor for reward accumulation.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for regularization.
        batch_size (int): Mini-batch size for training.
        entropy_coef (float): Weight for entropy bonus.
        value_loss_coef (float): Weight for value loss in total loss.
        model (AlphaZeroNet): Neural network model for policy/value.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion_value (nn.Module): Value loss function (MSE by default).
    """

    def __init__(
        self,
        board_size: int,
        num_ware: int,
        action_size: int,
        num_simulations: int,
        num_res_blocks: int,
        in_channels: int,
        mid_channels: int,
        temperature: float = 1.0,
        gamma: float = 0.99,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
    ):
        """
        Initializes the AlphaZero-like trainer.

        Args:
            board_size (int): Chess board size, typically 8.
            num_ware (int): Number of input channels (planes).
            action_size (int): Number of possible moves (e.g., 4032).
            num_simulations (int): Number of MCTS rollouts.
            num_res_blocks (int): Number of residual blocks in the network.
            in_channels (int): Initial in-channels for conv layers.
            mid_channels (int): Mid-channels used inside each residual block.
            temperature (float): MCTS temperature for exploration.
            gamma (float): Discount factor.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer.
            batch_size (int): Training batch size.
            entropy_coef (float): Weight for entropy bonus.
            value_loss_coef (float): Weight for value loss in total loss.
        """
        self.board_size = board_size
        self.num_ware = num_ware
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AlphaZeroNet(
            board_size, num_ware, action_size, num_res_blocks, in_channels, mid_channels
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion_value = nn.MSELoss()

    def train(self, epochs: int, num_games_per_epoch: int, max_workers=None) -> None:
        """
        Orchestrates multiple epochs of self-play data generation and model training.

        Args:
            epochs (int): Number of epochs to run.
            num_games_per_epoch (int): Number of self-play games per epoch.
            max_workers (int|None): Maximum number of parallel workers for self-play.
        """
        for epoch in range(epochs):
            print(f"===== Epoch {epoch + 1}/{epochs} =====")
            training_data = self.generate_self_play_data_parallel(num_games_per_epoch, max_workers)
            loss = self.update_model(training_data)
            print(f"Loss after epoch {epoch + 1}: {loss:.6f}")

    def generate_self_play_data_parallel(self, num_games: int, max_workers=None):
        """
        Runs multiple self-play games in parallel to gather training data.

        Args:
            num_games (int): Number of self-play games to generate.
            max_workers (int|None): Max parallel processes.

        Returns:
            A list of (state, policy, reward) tuples.
        """
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate_single_game_data) for _ in range(num_games)]
            results = [f.result() for f in futures]

        data = [item for result in results for item in result]
        return data

    def _generate_single_game_data(self):
        """
        Runs one self-play game using MCTS and returns the collected (state, policy, reward) tuples.
        """
        game = ChessGame()
        self_play = SelfPlay(self.model, game, self.num_simulations, self.temperature)
        states, policies, rewards = self_play.play()
        return list(zip(states, policies, rewards))

    def update_model(self, training_data) -> float:
        """
        Single update cycle for the model, using A2C-like approach (policy + value + advantage + entropy).

        Args:
            training_data: A list of (state, policy, reward) tuples.

        Returns:
            float: The final loss value from the last training batch (for logging purposes).
        """
        self.model.train()
        states, policies, rewards = zip(*training_data)

        # Compute discounted returns
        R = 0.0
        accumulative_rewards = []
        for idx, reward in enumerate(reversed(rewards)):
            R = reward if idx == 0 else self.gamma * R
            accumulative_rewards.insert(0, R)

        # Convert to tensors
        device = self.model.device
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
        accumulative_rewards_tensor = torch.tensor(
            np.array(accumulative_rewards), dtype=torch.float32
        ).to(device).unsqueeze(1)

        dataset = TensorDataset(states_tensor, policies_tensor, accumulative_rewards_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        final_loss = 0.0

        for batch_states, batch_policies, batch_returns in dataloader:
            pred_policies, pred_values = self.model(batch_states)

            # Squeeze as necessary
            pred_values = pred_values.squeeze()
            batch_returns = batch_returns.squeeze()

            # Advantage = returns - predicted value
            advantage = batch_returns - pred_values

            # Policy loss with advantage
            log_probs = torch.log(pred_policies + 1e-10)
            # Summation across action dimension
            policy_loss_term = (batch_policies * log_probs).sum(dim=1)
            policy_loss = -torch.mean(policy_loss_term * advantage.detach())

            # Entropy bonus
            entropy = -torch.sum(pred_policies * log_probs, dim=1).mean()
            entropy_bonus = self.entropy_coef * entropy

            # Value loss (MSE)
            value_loss = self.criterion_value(pred_values, batch_returns)

            # Combine into final loss
            # A2C typical:  policy_loss + c1*value_loss - c2*entropy
            loss = policy_loss + (self.value_loss_coef * value_loss) - entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            final_loss = loss.item()

        return final_loss

    def save_model(self, filepath: str) -> None:
        """
        Saves the current model's state dictionary.

        Args:
            filepath (str): File path to save to.
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}.")

    def load_model(self, filepath: str) -> None:
        """
        Loads a model state dictionary from the specified file path.

        Args:
            filepath (str): File path from which to load.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.model.device))
        self.model.eval()
        print(f"Model loaded from {filepath}.")


if __name__ == "__main__":
    # Example usage
    trainer = AlphaZeroTrainer(
        board_size=8,
        num_ware=14,
        action_size=4032,
        num_simulations=3,
        num_res_blocks=2,
        in_channels=32,
        mid_channels=8,
        temperature=1.5,
        gamma=0.99,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=4,
        entropy_coef=0.01,
        value_loss_coef=1.0,
    )