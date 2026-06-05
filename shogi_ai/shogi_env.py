"""
shogi_env.py
============
将棋の盤面表現・棋譜パーサ・CNN用エンコーダ。

棋譜は西洋式 (Hodges) 表記を想定:
  P7g-7f      : 7gの歩 -> 7f
  P2fx2e      : 2fの歩が2eで駒を取る
  B3cx7g+    : 3cの角が7gで取って成る (成った場合は末尾 +)
  S'4e        : 持ち駒の銀を4eに打つ (' のあとに座標)
  +P8gx7h    : 成歩(と金)が8gから7hへ取って動く (先頭 + は成った駒)

座標: ファイル(筋)は 1-9 (右から左)、ランク(段)は a-i (上から下)。
本実装では (row, col) = (rank_index, file_index) で 0-8 とする。
sente(先手) は下から上に進む。CNNの入力テンソルは「手番側視点」に
正規化することで、先手・後手を区別せずに学習させる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# 駒の定義
# ----------------------------------------------------------------------------
# 8種類の生駒 + 持ち駒7種 + 成駒6種 = 学習用のチャンネル設計は後述
PIECE_TYPES = ["P", "L", "N", "S", "G", "B", "R", "K"]  # 歩香桂銀金角飛玉
PROMOTABLE = {"P", "L", "N", "S", "B", "R"}             # 金と玉は不成
PIECE_TO_IDX = {p: i for i, p in enumerate(PIECE_TYPES)}

# 持ち駒として持てるのは玉以外
DROPPABLE = ["P", "L", "N", "S", "G", "B", "R"]
DROP_TO_IDX = {p: i for i, p in enumerate(DROPPABLE)}

FILE_CHARS = "987654321"   # 棋譜文字 '9'..'1' を col 0..8 に対応させる
RANK_CHARS = "abcdefghi"   # 'a'..'i' を row 0..8 に対応させる


def sq_from_str(s: str) -> tuple[int, int]:
    """例: '7f' -> (row=5, col=2)。"""
    file_ch, rank_ch = s[0], s[1]
    col = FILE_CHARS.index(file_ch)
    row = RANK_CHARS.index(rank_ch)
    return row, col


# ----------------------------------------------------------------------------
# 1手の表現
# ----------------------------------------------------------------------------
@dataclass
class Move:
    piece: str          # 生駒の英字 'P','L',...'K' (移動元での駒種、成駒なら base のみ)
    from_sq: Optional[tuple[int, int]]   # 打ちなら None
    to_sq: tuple[int, int]
    promote: bool = False
    is_drop: bool = False
    is_capture: bool = False
    was_promoted: bool = False   # 動かす駒が既に成っていたか (+P など)


def parse_move(token: str) -> Optional[Move]:
    """棋譜1トークンを Move に変換。解析できなければ None。"""
    t = token.strip()
    if not t:
        return None
    # 中括弧コメント等を弾く
    if t.startswith("{") or t.startswith("(") or t.startswith("["):
        return None

    promote = False
    if t.endswith("+"):
        promote = True
        t = t[:-1]

    # 既に成った駒で始まる場合 (+P, +L, +N, +S, +B, +R)
    was_promoted = False
    if t.startswith("+") and len(t) > 1 and t[1] in PROMOTABLE:
        was_promoted = True
        t = t[1:]

    if len(t) < 4:
        return None

    piece = t[0]
    if piece not in PIECE_TO_IDX:
        return None

    rest = t[1:]
    # 打ち
    if rest.startswith("'"):
        to_str = rest[1:3] if len(rest) >= 3 else ""
        if len(to_str) != 2:
            return None
        try:
            to_sq = sq_from_str(to_str)
        except ValueError:
            return None
        return Move(piece=piece, from_sq=None, to_sq=to_sq,
                    promote=False, is_drop=True)

    # 通常移動: '7g-7f' か '2fx2e'
    if len(rest) < 5:
        return None
    from_str = rest[0:2]
    sep = rest[2]
    to_str = rest[3:5]
    if sep not in "-x":
        return None
    try:
        from_sq = sq_from_str(from_str)
        to_sq = sq_from_str(to_str)
    except ValueError:
        return None
    return Move(piece=piece, from_sq=from_sq, to_sq=to_sq,
                promote=promote, is_drop=False,
                is_capture=(sep == "x"), was_promoted=was_promoted)


# ----------------------------------------------------------------------------
# 盤面
# ----------------------------------------------------------------------------
@dataclass
class Board:
    # board[row][col] = (piece, owner, promoted) または None
    # owner: 0 = sente(先手, 下), 1 = gote(後手, 上)
    grid: list[list[Optional[tuple[str, int, bool]]]] = field(default_factory=list)
    hands: list[dict[str, int]] = field(default_factory=list)   # [sente_hand, gote_hand]
    turn: int = 0                                                # 0 = 先手番

    @classmethod
    def initial(cls) -> "Board":
        b = cls(
            grid=[[None] * 9 for _ in range(9)],
            hands=[{p: 0 for p in DROPPABLE}, {p: 0 for p in DROPPABLE}],
            turn=0,
        )
        # 後手陣 (上段, row 0-2)
        b.grid[0] = [
            ("L", 1, False), ("N", 1, False), ("S", 1, False), ("G", 1, False),
            ("K", 1, False), ("G", 1, False), ("S", 1, False), ("N", 1, False), ("L", 1, False),
        ]
        b.grid[1][1] = ("R", 1, False)
        b.grid[1][7] = ("B", 1, False)
        for c in range(9):
            b.grid[2][c] = ("P", 1, False)
        # 先手陣 (下段, row 6-8)
        for c in range(9):
            b.grid[6][c] = ("P", 0, False)
        b.grid[7][7] = ("R", 0, False)
        b.grid[7][1] = ("B", 0, False)
        b.grid[8] = [
            ("L", 0, False), ("N", 0, False), ("S", 0, False), ("G", 0, False),
            ("K", 0, False), ("G", 0, False), ("S", 0, False), ("N", 0, False), ("L", 0, False),
        ]
        return b

    def apply(self, mv: Move) -> bool:
        """Move を適用。失敗(不整合)なら False。"""
        r, c = mv.to_sq
        owner = self.turn
        if mv.is_drop:
            # 持ち駒から打つ
            if self.hands[owner].get(mv.piece, 0) <= 0:
                return False
            if self.grid[r][c] is not None:
                return False
            self.hands[owner][mv.piece] -= 1
            self.grid[r][c] = (mv.piece, owner, False)
        else:
            fr, fc = mv.from_sq
            src = self.grid[fr][fc]
            if src is None or src[1] != owner:
                return False
            # 取る
            tgt = self.grid[r][c]
            if tgt is not None:
                # 持ち駒に加える (成駒は生駒に戻す)
                captured_piece = tgt[0]
                if captured_piece in DROPPABLE:
                    self.hands[owner][captured_piece] = self.hands[owner].get(captured_piece, 0) + 1
            promoted = src[2] or mv.promote
            self.grid[r][c] = (src[0], owner, promoted)
            self.grid[fr][fc] = None
        self.turn = 1 - self.turn
        return True


# ----------------------------------------------------------------------------
# CNN 入力テンソルへのエンコード
# ----------------------------------------------------------------------------
# チャンネル設計 (合計 31ch):
#   0-7   : 手番側 8種の生駒/玉   (1 if 自駒, else 0)
#   8-13  : 手番側 成駒 (P,L,N,S,B,R 順)
#   14-21 : 相手側 8種の生駒/玉
#   22-27 : 相手側 成駒
#   28    : 手番側 持ち駒数 (P,L,N,S,G,B,R を合計... ではなく後で分割)
#
# 持ち駒は別途 7 + 7 = 14ch の「全マス同じ値」プレーンで持たせる。
# 最終的に 28 + 14 = 42ch。学習側でこの仕様を使う。
BOARD_CHANNELS = 28
HAND_CHANNELS = 14
TOTAL_CHANNELS = BOARD_CHANNELS + HAND_CHANNELS

PROMOTED_LIST = ["P", "L", "N", "S", "B", "R"]   # 成駒対象
PROMOTED_IDX = {p: i for i, p in enumerate(PROMOTED_LIST)}


def encode_board(board: Board) -> np.ndarray:
    """Board を (C, 9, 9) の float32 テンソルに変換 (手番側視点に正規化)。"""
    me = board.turn
    arr = np.zeros((TOTAL_CHANNELS, 9, 9), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            cell = board.grid[r][c]
            if cell is None:
                continue
            piece, owner, promoted = cell
            # 手番側視点に揃えるため、後手番のときは盤を 180度回す
            rr, cc = (r, c) if me == 0 else (8 - r, 8 - c)
            is_me = (owner == me)
            if promoted and piece in PROMOTED_IDX:
                base_offset = 8 + PROMOTED_IDX[piece] if is_me else 22 + PROMOTED_IDX[piece]
                arr[base_offset, rr, cc] = 1.0
            else:
                base_offset = PIECE_TO_IDX[piece] if is_me else 14 + PIECE_TO_IDX[piece]
                arr[base_offset, rr, cc] = 1.0

    # 持ち駒プレーン (全マスに同じ値)。学習しやすさのため /18 で正規化
    for i, p in enumerate(DROPPABLE):
        arr[BOARD_CHANNELS + i, :, :] = board.hands[me].get(p, 0) / 18.0
        arr[BOARD_CHANNELS + 7 + i, :, :] = board.hands[1 - me].get(p, 0) / 18.0

    return arr


# ----------------------------------------------------------------------------
# 行動空間 (policy のラベル)
# ----------------------------------------------------------------------------
# 単純化のため、出力ラベル = "移動先 9x9 = 81" + "成る/不成 = 2倍" + "持ち駒打ち = 7種 x 81"
# だが、学習を簡単にするためここでは以下のシンプルな表現を採用する:
#   ラベル = (to_row * 9 + to_col) * (2 + 7)
#     0..80*9-1 で、各 to_sq に対し:
#       0: 通常移動 (不成)
#       1: 通常移動 (成)
#       2..8: 持ち駒打ち (P,L,N,S,G,B,R)
# 合計 81 * 9 = 729 クラス
NUM_LABELS = 81 * 9


def move_to_label(mv: Move) -> int:
    r, c = mv.to_sq
    sq_idx = r * 9 + c
    if mv.is_drop:
        sub = 2 + DROP_TO_IDX[mv.piece]
    else:
        sub = 1 if mv.promote else 0
    return sq_idx * 9 + sub


def label_to_human(label: int) -> str:
    sq_idx, sub = divmod(label, 9)
    r, c = divmod(sq_idx, 9)
    to_str = f"{FILE_CHARS[c]}{RANK_CHARS[r]}"
    if sub == 0:
        return f"move->{to_str}"
    if sub == 1:
        return f"move->{to_str}+ (promote)"
    return f"drop {DROPPABLE[sub - 2]}->{to_str}"


# ラベルを「手番側視点」に正規化する関数
def normalize_label_for_turn(mv: Move, turn: int) -> int:
    """手番が後手なら盤を 180度回した上でラベル化する。"""
    if turn == 0:
        return move_to_label(mv)
    # 後手の場合: to_sq を回す
    r, c = mv.to_sq
    rr, cc = 8 - r, 8 - c
    sq_idx = rr * 9 + cc
    if mv.is_drop:
        sub = 2 + DROP_TO_IDX[mv.piece]
    else:
        sub = 1 if mv.promote else 0
    return sq_idx * 9 + sub
