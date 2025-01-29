import torch
import torch.nn as nn
import torch.optim as optim
from self_play import SelfPlay
from game import ChessGame
import numpy as np
from model import AlphaZeroNet
from concurrent.futures import ProcessPoolExecutor


class AlphaZeroTrainer:
    def __init__(
        self,
        board_size,
        action_size,
        num_simulations,
        num_res_blocks,
        in_channels,
        mid_channels,
        lr=0.001,
        weight_decay=1e-4,
        batch_size=8,
    ):
        self.board_size = board_size
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.model = AlphaZeroNet(
            board_size, action_size, num_res_blocks, in_channels, mid_channels
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()

    def train(self, epochs, num_games_per_epoch, temperature):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            training_data = self.generate_self_play_data_parallel(
                num_games_per_epoch, temperature
            )
            self.update_model(training_data)

    def generate_self_play_data_parallel(self, num_games, temperature):
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_single_game_data, temperature)
                for _ in range(num_games)
            ]
            results = [future.result() for future in futures]
        data = [item for result in results for item in result]
        return data

    def _generate_single_game_data(self, temperature):
        game = ChessGame()
        self_play = SelfPlay(self.model, game, self.num_simulations, temperature)
        states, policies, rewards = self_play.play()
        return list(zip(states, policies, rewards))

    def update_model(self, training_data):
        self.model.train()
        states, policies, rewards = zip(*training_data)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.model.device
        )
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(
            self.model.device
        )
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(
            self.model.device
        )
        dataset = torch.utils.data.TensorDataset(states, policies, rewards)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        for batch_states, batch_policies, batch_rewards in dataloader:
            pred_policies, pred_values = self.model(batch_states)
            loss_policy = -(batch_policies.detach() * torch.log(pred_policies)).mean()
            loss_value = self.criterion_value(pred_values.squeeze(), batch_rewards)
            loss = loss_policy + loss_value
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()


if __name__ == "__main__":
    board_size = 8
    action_size = 4672
    num_simulations = 100
    trainer = AlphaZeroTrainer(board_size, action_size, num_simulations)
    trainer.train(epochs=10, num_games_per_epoch=10, temperature=1.0)
    trainer.save_model("alphazero_chess.pth")
