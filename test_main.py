from shogi_ai.shogi_env import Board, encode_board, parse_move
from server.app import DROPPABLE, build_legal_mask_729


def empty_board(turn: int = 0) -> Board:
    return Board(
        grid=[[None] * 9 for _ in range(9)],
        hands=[
            {piece: 0 for piece in DROPPABLE},
            {piece: 0 for piece in DROPPABLE},
        ],
        turn=turn,
    )


def move_label(row: int, col: int) -> int:
    return (row * 9 + col) * 9


def test_parse_apply_and_encode_move() -> None:
    board = Board.initial()
    move = parse_move("P7g-7f")

    assert move is not None
    assert board.apply(move)

    encoded = encode_board(board)
    assert encoded.shape == (42, 9, 9)
    assert board.turn == 1


def test_king_cannot_capture_a_defended_checking_piece() -> None:
    board = empty_board()
    board.grid[8][4] = ("K", 0, False)
    board.grid[7][4] = ("R", 1, False)
    board.grid[6][4] = ("G", 1, False)
    board.grid[0][0] = ("K", 1, False)

    mask = build_legal_mask_729(board, is_sente=True)

    assert not mask[move_label(7, 4)]


def test_king_can_capture_an_undefended_checking_piece() -> None:
    board = empty_board()
    board.grid[8][4] = ("K", 0, False)
    board.grid[7][4] = ("R", 1, False)
    board.grid[0][0] = ("K", 1, False)

    mask = build_legal_mask_729(board, is_sente=True)

    assert mask[move_label(7, 4)]


def test_pinned_piece_cannot_expose_its_king() -> None:
    board = empty_board()
    board.grid[8][4] = ("K", 0, False)
    board.grid[7][4] = ("G", 0, False)
    board.grid[0][4] = ("R", 1, False)
    board.grid[0][0] = ("K", 1, False)

    mask = build_legal_mask_729(board, is_sente=True)

    assert not mask[move_label(6, 3)]
    assert mask[move_label(6, 4)]


def test_drop_must_resolve_check() -> None:
    board = empty_board()
    board.grid[8][4] = ("K", 0, False)
    board.grid[0][4] = ("R", 1, False)
    board.grid[0][0] = ("K", 1, False)
    board.hands[0]["G"] = 1
    gold_drop_sub = 2 + DROPPABLE.index("G")

    mask = build_legal_mask_729(board, is_sente=True)

    blocking_label = (4 * 9 + 4) * 9 + gold_drop_sub
    unrelated_label = (4 * 9 + 3) * 9 + gold_drop_sub
    assert mask[blocking_label]
    assert not mask[unrelated_label]


def test_promoted_piece_can_move_to_last_rank() -> None:
    board = empty_board()
    board.grid[8][4] = ("K", 0, False)
    board.grid[1][4] = ("P", 0, True)
    board.grid[0][0] = ("K", 1, False)

    mask = build_legal_mask_729(board, is_sente=True)

    assert mask[move_label(0, 4)]


def test_pawn_drop_mate_is_rejected() -> None:
    board = empty_board()
    board.grid[8][4] = ("K", 0, False)
    board.grid[0][4] = ("K", 1, False)
    board.grid[2][3] = ("R", 0, False)
    board.grid[2][4] = ("G", 0, False)
    board.grid[2][5] = ("R", 0, False)
    board.hands[0]["P"] = 1
    pawn_drop_sub = 2 + DROPPABLE.index("P")
    pawn_drop_label = (1 * 9 + 4) * 9 + pawn_drop_sub

    mask = build_legal_mask_729(board, is_sente=True)
    assert not mask[pawn_drop_label]

    board.grid[2][3] = None
    mask_with_escape = build_legal_mask_729(board, is_sente=True)
    assert mask_with_escape[pawn_drop_label]
