import torch
import logging
from train import AlphaZeroTrainer
from model import AlphaZeroNet
from performance_evaluation import evaluate_model_parallel

logging.basicConfig(
    filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


def main():
    board_size = 8
    action_size = 4032
    num_simulations = 10
    num_epochs = 100
    num_games_per_epoch = 3
    temperature = 1
    evaluation_games = 5
    batch_size = 32

    num_res_blocks = 3
    in_channels = 64
    mid_channels = 16

    model = AlphaZeroNet(
        board_size,
        action_size,
        num_res_blocks=num_res_blocks,
        in_channels=in_channels,
        mid_channels=mid_channels,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    trainer = AlphaZeroTrainer(
        board_size=board_size,
        action_size=action_size,
        num_simulations=num_simulations,
        num_res_blocks=num_res_blocks,
        in_channels=in_channels,
        mid_channels=mid_channels,
        temperature=temperature,
        lr=3e-4,
        weight_decay=1e-4,
        batch_size=batch_size,
    )

    for epoch in range(num_epochs):
        logging.info(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        print("Generating Self-Play data...")
        training_data = trainer.generate_self_play_data_parallel(
            num_games=num_games_per_epoch,
            max_workers=num_games_per_epoch,
        )
        print(f"Generated {len(training_data)} training data points.")

        print("Training model...")
        loss = trainer.update_model(training_data)
        logging.info(f"Epoch {epoch + 1} Loss: {loss}")

        checkpoint_path = f"./model/alphazero_chess_epoch_{epoch + 1}.pth"
        trainer.save_model(checkpoint_path)
        print(f"Model saved to {checkpoint_path}.")

        if (epoch + 1) % 5 == 0:
            print("Evaluating model...")
            win_rate, loss_rate, draw_rate = evaluate_model_parallel(
                model,
                num_games=evaluation_games,
                num_simulations=num_simulations,
                temperature=0.1,
            )
            print(
                f"Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}"
            )
            logging.info(
                f"Evaluation - Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}"
            )

    print("\nTraining complete. Final model saved.")


if __name__ == "__main__":
    main()
