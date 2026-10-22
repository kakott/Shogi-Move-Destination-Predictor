"""駒の移動規則と合法手マスクの生成処理。"""

from typing import Optional

import numpy as np

from shogi_ai.shogi_env import Board, DROPPABLE, PROMOTABLE

from .config import NUM_LABELS


def is_in_bounds(row: int, col: int) -> bool:
    return 0 <= row < 9 and 0 <= col < 9


def is_in_promotion_zone(row: int, is_sente: bool) -> bool:
    """先手なら0～2段、後手なら6～8段が敵陣。"""
    return row <= 2 if is_sente else row >= 6


def must_promote(piece: str, to_row: int, is_sente: bool) -> bool:
    """到達後に動けなくなるため成りが強制される位置か判定する。"""
    if piece in {"P", "L"}:
        return to_row == (0 if is_sente else 8)
    if piece == "N":
        return to_row in ({0, 1} if is_sente else {7, 8})
    return False


def belongs_to(board: Board, row: int, col: int, owner: int) -> bool:
    cell = board.grid[row][col]
    return cell is not None and cell[1] == owner


def add_step_dst(
    board: Board,
    destinations: set,
    row: int,
    col: int,
    deltas,
    owner: int,
) -> None:
    for delta_row, delta_col in deltas:
        next_row = row + delta_row
        next_col = col + delta_col
        if not is_in_bounds(next_row, next_col):
            continue
        if belongs_to(board, next_row, next_col, owner):
            continue
        destinations.add((row, col, next_row, next_col))


def add_slider_dst(
    board: Board,
    destinations: set,
    row: int,
    col: int,
    directions,
    owner: int,
) -> None:
    for delta_row, delta_col in directions:
        next_row = row + delta_row
        next_col = col + delta_col
        while is_in_bounds(next_row, next_col):
            if belongs_to(board, next_row, next_col, owner):
                break
            destinations.add((row, col, next_row, next_col))
            if board.grid[next_row][next_col] is not None:
                break
            next_row += delta_row
            next_col += delta_col


def piece_movement(
    board: Board,
    row: int,
    col: int,
    owner: int,
) -> set[tuple[int, int, int, int]]:
    """指定した駒の移動可能な始点・終点を返す。"""
    destinations: set[tuple[int, int, int, int]] = set()
    cell = board.grid[row][col]
    if cell is None or cell[1] != owner:
        return destinations

    piece, _, promoted = cell
    forward = -1 if owner == 0 else 1

    king_steps = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    gold_steps = [
        (forward, -1), (forward, 0), (forward, 1),
        (0, -1), (0, 1), (-forward, 0),
    ]
    silver_steps = [
        (forward, -1), (forward, 0), (forward, 1),
        (-forward, -1), (-forward, 1),
    ]
    knight_steps = [(forward * 2, -1), (forward * 2, 1)]

    if promoted and piece in {"P", "L", "N", "S"}:
        add_step_dst(
            board, destinations, row, col, gold_steps, owner
        )
        return destinations

    if piece == "K":
        add_step_dst(board, destinations, row, col, king_steps, owner)
    elif piece == "G":
        add_step_dst(board, destinations, row, col, gold_steps, owner)
    elif piece == "S":
        add_step_dst(board, destinations, row, col, silver_steps, owner)
    elif piece == "N":
        add_step_dst(board, destinations, row, col, knight_steps, owner)
    elif piece == "L":
        add_slider_dst(
            board, destinations, row, col, [(forward, 0)], owner
        )
    elif piece == "P":
        add_step_dst(
            board, destinations, row, col, [(forward, 0)], owner
        )
    elif piece == "B":
        add_slider_dst(
            board,
            destinations,
            row,
            col,
            [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            owner,
        )
        if promoted:
            add_step_dst(
                board,
                destinations,
                row,
                col,
                [(-1, 0), (0, -1), (0, 1), (1, 0)],
                owner,
            )
    elif piece == "R":
        add_slider_dst(
            board,
            destinations,
            row,
            col,
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            owner,
        )
        if promoted:
            add_step_dst(
                board,
                destinations,
                row,
                col,
                [(-1, -1), (-1, 1), (1, -1), (1, 1)],
                owner,
            )
    return destinations


def has_own_pawn_on_file(board: Board, col: int, owner: int) -> bool:
    """指定した筋に自分の成っていない歩がいるか判定する。"""
    for row in range(9):
        cell = board.grid[row][col]
        if cell is None:
            continue
        piece, piece_owner, promoted = cell
        if piece == "P" and piece_owner == owner and not promoted:
            return True
    return False


def find_king(board: Board, owner: int) -> Optional[tuple[int, int]]:
    """指定した手番の玉の座標を返す。"""
    for row in range(9):
        for col in range(9):
            cell = board.grid[row][col]
            if cell is not None and cell[0] == "K" and cell[1] == owner:
                return row, col
    return None


def is_square_attacked(
    board: Board,
    row: int,
    col: int,
    attacker: int,
) -> bool:
    """指定マスがattackerの駒に攻撃されているか判定する。"""
    for piece_row in range(9):
        for piece_col in range(9):
            cell = board.grid[piece_row][piece_col]
            if cell is None or cell[1] != attacker:
                continue
            if any(
                to_row == row and to_col == col
                for _, _, to_row, to_col in piece_movement(
                    board, piece_row, piece_col, attacker
                )
            ):
                return True
    return False


def copy_board(board: Board) -> Board:
    """合法手判定用に盤面を複製する。"""
    return Board(
        grid=[row.copy() for row in board.grid],
        hands=[hand.copy() for hand in board.hands],
        turn=board.turn,
    )


def keeps_king_safe_after_move(
    board: Board,
    owner: int,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    promote: bool = False,
) -> bool:
    """盤上の手を仮適用し、自玉が王手にならないか判定する。"""
    next_board = copy_board(board)
    cell = next_board.grid[from_row][from_col]
    if cell is None or cell[1] != owner:
        return False

    piece, _, already_promoted = cell
    next_board.grid[from_row][from_col] = None
    next_board.grid[to_row][to_col] = (
        piece,
        owner,
        already_promoted or promote,
    )

    king = find_king(next_board, owner)
    if king is None:
        return False
    return not is_square_attacked(next_board, *king, attacker=1 - owner)


def keeps_king_safe_after_drop(
    board: Board,
    owner: int,
    piece: str,
    to_row: int,
    to_col: int,
) -> bool:
    """駒打ちを仮適用し、自玉が王手にならないか判定する。"""
    next_board = copy_board(board)
    next_board.grid[to_row][to_col] = (piece, owner, False)

    king = find_king(next_board, owner)
    if king is None:
        return False
    return not is_square_attacked(next_board, *king, attacker=1 - owner)


def has_legal_board_move(board: Board, owner: int) -> bool:
    """ownerに自玉を安全にする盤上の応手があるか調べる。"""
    for row in range(9):
        for col in range(9):
            for from_row, from_col, to_row, to_col in piece_movement(
                board, row, col, owner
            ):
                target = board.grid[to_row][to_col]
                if target is not None and target[0] == "K":
                    continue
                if keeps_king_safe_after_move(
                    board,
                    owner,
                    from_row,
                    from_col,
                    to_row,
                    to_col,
                ):
                    return True
    return False


def is_pawn_drop_mate(
    board: Board,
    owner: int,
    to_row: int,
    to_col: int,
) -> bool:
    """その歩打ちが禁じ手の打ち歩詰めになるか判定する。"""
    opponent = 1 - owner
    opponent_king = find_king(board, opponent)
    pawn_attack_row = to_row + (-1 if owner == 0 else 1)
    if opponent_king != (pawn_attack_row, to_col):
        return False

    next_board = copy_board(board)
    next_board.grid[to_row][to_col] = ("P", owner, False)

    # 歩による王手は隣接しているため、駒打ちで遮ることはできない。
    return not has_legal_board_move(next_board, opponent)


def build_legal_mask_729(board: Board, is_sente: bool) -> np.ndarray:
    """自玉の安全も検証した729次元の合法手マスクを返す。"""
    mask = np.zeros(NUM_LABELS, dtype=bool)
    owner = 0 if is_sente else 1

    for row in range(9):
        for col in range(9):
            movements = piece_movement(board, row, col, owner)
            for from_row, from_col, to_row, to_col in movements:
                cell = board.grid[from_row][from_col]
                if cell is None:
                    continue
                piece, _, promoted = cell

                target = board.grid[to_row][to_col]
                if target is not None and target[0] == "K":
                    continue
                if not keeps_king_safe_after_move(
                    board,
                    owner,
                    from_row,
                    from_col,
                    to_row,
                    to_col,
                ):
                    continue

                if is_sente:
                    normalized_row, normalized_col = to_row, to_col
                else:
                    normalized_row = 8 - to_row
                    normalized_col = 8 - to_col
                square_index = normalized_row * 9 + normalized_col

                if promoted or not must_promote(piece, to_row, is_sente):
                    mask[square_index * 9] = True

                if (
                    piece in PROMOTABLE
                    and not promoted
                    and (
                        is_in_promotion_zone(to_row, is_sente)
                        or is_in_promotion_zone(from_row, is_sente)
                    )
                ):
                    mask[square_index * 9 + 1] = True

    for sub_offset, piece in enumerate(DROPPABLE):
        if board.hands[owner].get(piece, 0) <= 0:
            continue
        for row in range(9):
            for col in range(9):
                if board.grid[row][col] is not None:
                    continue
                if piece in {"P", "L"} and row == (
                    0 if is_sente else 8
                ):
                    continue
                if piece == "N" and row in (
                    {0, 1} if is_sente else {7, 8}
                ):
                    continue
                if (
                    piece == "P"
                    and has_own_pawn_on_file(board, col, owner)
                ):
                    continue
                if piece == "P" and is_pawn_drop_mate(
                    board, owner, row, col
                ):
                    continue
                if not keeps_king_safe_after_drop(
                    board, owner, piece, row, col
                ):
                    continue

                if is_sente:
                    normalized_row, normalized_col = row, col
                else:
                    normalized_row = 8 - row
                    normalized_col = 8 - col
                square_index = normalized_row * 9 + normalized_col
                mask[square_index * 9 + 2 + sub_offset] = True

    return mask
