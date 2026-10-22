from shogi_ai.shogi_env import Board, encode_board, parse_move


def test_parse_apply_and_encode_move() -> None:
    board = Board.initial()
    move = parse_move("P7g-7f")

    assert move is not None
    assert board.apply(move)

    encoded = encode_board(board)
    assert encoded.shape == (42, 9, 9)
    assert board.turn == 1
