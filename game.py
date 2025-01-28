import chess
import numpy as np


class ChessGame:
    def __init__(self, board=None):
        self.board = board if board else chess.Board()

    def get_initial_state(self):
        return self.board.fen()

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def apply_move(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        return ChessGame(new_board)

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self, player):
        if self.board.is_checkmate():
            return -1 if self.board.turn == player else 1
        return 0

    def get_board_state(self):
        board_state = np.zeros((8, 8, 14), dtype=np.float32)
        piece_to_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                layer = piece_to_index[piece.piece_type] + (
                    0 if piece.color == chess.WHITE else 6
                )
                row, col = divmod(square, 8)
                board_state[row, col, layer] = 1

        return board_state

    def __str__(self):
        return str(self.board)


if __name__ == "__main__":
    game = ChessGame()
    print("Initial board:")
    print(game)

    print("\nLegal moves:")
    print(game.get_legal_moves())

    move = game.get_legal_moves()[0]
    print(f"\nApplying move: {move}")
    game = game.apply_move(move)
    print(game)

    print("\nIs game over?", game.is_game_over())
    print("Result for player 1:", game.get_result(1))
