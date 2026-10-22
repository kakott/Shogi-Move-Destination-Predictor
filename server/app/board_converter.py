"""フロントエンドの盤面表現を学習側のBoardへ変換する処理。"""

from typing import Optional

from shogi_ai.shogi_env import Board, DROPPABLE, PIECE_TYPES


def parse_piece_str(value: str) -> Optional[tuple[str, int, bool]]:
    """駒文字列を (駒種, 所有者, 成り状態) へ変換する。"""
    if not value:
        return None

    promoted = value.startswith("+")
    piece_text = value[1:] if promoted else value
    if not piece_text:
        return None

    is_sente = piece_text.isupper()
    piece = piece_text.upper()
    if piece not in PIECE_TYPES:
        return None
    return piece, 0 if is_sente else 1, promoted


def board_from_payload(
    grid_strs: list[list[Optional[str]]],
    hands: dict,
    is_sente: bool,
) -> Board:
    """APIの盤面表現からshogi_env.Boardを構築する。"""
    board = Board(
        grid=[[None] * 9 for _ in range(9)],
        hands=[
            {piece: 0 for piece in DROPPABLE},
            {piece: 0 for piece in DROPPABLE},
        ],
        turn=0 if is_sente else 1,
    )

    for row in range(9):
        for col in range(9):
            value = grid_strs[row][col]
            if not value:
                continue
            parsed = parse_piece_str(value)
            if parsed is not None:
                board.grid[row][col] = parsed

    sente_hand = hands.get("sente", {}) if isinstance(hands, dict) else {}
    gote_hand = hands.get("gote", {}) if isinstance(hands, dict) else {}
    for piece in DROPPABLE:
        board.hands[0][piece] = int(sente_hand.get(piece, 0) or 0)
        board.hands[1][piece] = int(gote_hand.get(piece, 0) or 0)

    return board
