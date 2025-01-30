import concurrent.futures
import torch
import logging


def evaluate_single_game(model, num_simulations, temperature):
    from game import ChessGame
    from self_play import SelfPlay

    game = ChessGame()
    self_play = SelfPlay(
        model, game, num_simulations=num_simulations, temperature=temperature
    )
    actions, _, rewards = self_play.play()
    logging.info(f"Game finished - Actions: {len(actions)}, Result: {rewards[-1]}")
    return rewards[-1]  # 게임 결과 반환 (1: 승리, -1: 패배, 0: 무승부)


def evaluate_model_parallel(model, num_games=20, num_simulations=50, temperature=0.0):
    wins, losses, draws = 0, 0, 0

    # 병렬 실행 설정
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 각 게임 평가를 병렬로 실행
        futures = [
            executor.submit(evaluate_single_game, model, num_simulations, temperature)
            for _ in range(num_games)
        ]

        # 결과 수집
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

    # 통계 계산
    total_games = wins + losses + draws
    win_rate = wins / total_games
    loss_rate = losses / total_games
    draw_rate = draws / total_games
    logging.info(
        f"Total Games: {total_games}, Wins: {wins}, Losses: {losses}, Draws: {draws}"
    )
    return win_rate, loss_rate, draw_rate


if __name__ == "__main__":
    from model import AlphaZeroNet

    model = AlphaZeroNet(8, 4032, 5, 64, 16)
    model.eval()
    win_rate, loss_rate, draw_rate = evaluate_model_parallel(
        model, num_games=50, temperature=0.0
    )
    print(
        f"Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}"
    )
