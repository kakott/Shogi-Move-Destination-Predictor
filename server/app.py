"""
将棋 次の一手予測 API サーバー (選択肢A: 学習側に合わせた版)
=========================================================
- モデル: ShogiPolicyNet (729クラス, ResNet 6ブロック, val_acc=0.38)
- 学習側の shogi_env.py の Board / encode_board を「正解」として使う
- フロントから受け取るのは生の盤面表記(駒文字列の9x9と持ち駒の枚数dict)
- 向き正規化と持ち駒 /18.0 はサーバ側で encode_board が自動でやる
- 合法手マスクは 729次元に拡張

必要ライブラリ:
  pip install fastapi uvicorn torch numpy

起動:
  python app.py
  または
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 学習側のコードを再利用 (同じ仕様で推論するため)
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
SHOGI_AI_DIR = PROJECT_ROOT / "shogi_ai"

if str(SHOGI_AI_DIR) not in sys.path:
    sys.path.insert(0, str(SHOGI_AI_DIR))

from model import ShogiPolicyNet
from shogi_env import (
    Board,
    DROPPABLE,
    FILE_CHARS,
    PIECE_TYPES,
    PROMOTABLE,
    PROMOTED_LIST,
    RANK_CHARS,
    encode_board,
)


MODEL_PATH = SHOGI_AI_DIR / "best_model.pt"
NUM_LABELS = 729   # 81マス × 9サブアクション


# ============================================================
# モデル読み込み
# ============================================================
print("モデル読み込み中...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# state_dict から自動でブロック数を決める (メタの 'blocks' は信頼しない)
state_dict = ckpt["model"] if "model" in ckpt else ckpt
n_blocks_in_ckpt = sum(
    1 for k in state_dict.keys()
    if k.startswith("blocks.") and k.endswith(".conv1.weight")
)
n_blocks = n_blocks_in_ckpt if n_blocks_in_ckpt > 0 else ckpt.get("blocks", 6)
ch = ckpt.get("ch", 128)
in_channels = ckpt.get("in_channels", 42)

model = ShogiPolicyNet(
    in_channels=in_channels, ch=ch, n_blocks=n_blocks, n_labels=NUM_LABELS
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(
    f"モデル読み込み完了: device={device}, in_channels={in_channels}, "
    f"ch={ch}, blocks={n_blocks}, val_acc={ckpt.get('val_acc', '?')}"
)


# ============================================================
# 駒文字列 ↔ shogi_env.Board の cell 表現の変換
# ============================================================
# フロントの駒表記:
#   大文字 = 先手 (例: 'P', 'R', '+B')
#   小文字 = 後手 (例: 'p', 'r', '+b')
# Board の cell 表現:
#   (piece: str(大文字), owner: 0=先手 1=後手, promoted: bool)

def parse_piece_str(s: str) -> Optional[tuple[str, int, bool]]:
    if not s:
        return None
    promoted = False
    if s.startswith("+"):
        promoted = True
        s = s[1:]
    if not s:
        return None
    is_sente = s.isupper()
    piece = s.upper()
    if piece not in PIECE_TYPES:
        return None
    return (piece, 0 if is_sente else 1, promoted)


def board_from_payload(
    grid_strs: list[list[Optional[str]]],
    hands: dict,
    is_sente: bool,
) -> Board:
    """フロントの盤面表現から shogi_env.Board を構築。"""
    b = Board(
        grid=[[None] * 9 for _ in range(9)],
        hands=[
            {p: 0 for p in DROPPABLE},
            {p: 0 for p in DROPPABLE},
        ],
        turn=0 if is_sente else 1,
    )
    for r in range(9):
        for c in range(9):
            cell = grid_strs[r][c]
            if cell:
                parsed = parse_piece_str(cell)
                if parsed is not None:
                    b.grid[r][c] = parsed
    # 持ち駒
    sente_hand = hands.get("sente", {}) if isinstance(hands, dict) else {}
    gote_hand = hands.get("gote", {}) if isinstance(hands, dict) else {}
    for p in DROPPABLE:
        b.hands[0][p] = int(sente_hand.get(p, 0) or 0)
        b.hands[1][p] = int(gote_hand.get(p, 0) or 0)
    return b


# ============================================================
# 出力ラベル ↔ 人間が読める手の変換
# ============================================================
SUB_NAMES = [
    "move",         # 0: 普通の移動 (不成)
    "promote",      # 1: 移動して成る
    "drop_P", "drop_L", "drop_N", "drop_S", "drop_G", "drop_B", "drop_R",
]

PIECE_NAME_JP = {
    "P": "歩", "L": "香", "N": "桂", "S": "銀", "G": "金",
    "B": "角", "R": "飛",
}


def label_to_obj(label: int, is_sente: bool) -> dict:
    """ラベル -> {row, col, sub, sub_name, label_text}。
    手番が後手なら、ラベル内の座標を実盤面に戻すため 180度逆回転する。"""
    sq_idx, sub = divmod(label, 9)
    r_norm, c_norm = divmod(sq_idx, 9)
    if is_sente:
        r, c = r_norm, c_norm
    else:
        r, c = 8 - r_norm, 8 - c_norm

    sub_name = SUB_NAMES[sub]
    # ラベル説明文
    to_str = f"{FILE_CHARS[c]}{RANK_CHARS[r]}"
    if sub == 0:
        label_text = f"-> {to_str}"
    elif sub == 1:
        label_text = f"-> {to_str} 成"
    else:
        piece = DROPPABLE[sub - 2]
        label_text = f"{PIECE_NAME_JP.get(piece, piece)}打 -> {to_str}"
    return {
        "row": r, "col": c, "sub": sub, "sub_name": sub_name,
        "label_text": label_text,
    }


# ============================================================
# 合法手マスク (729次元)
# ============================================================
# 着地マス9x9の合法判定 (前バージョンと同じロジック) を流用しつつ、
# サブアクション(成/不成/打ち)も含めた 729次元のマスクを作る。

def is_in_bounds(row: int, col: int) -> bool:
    return 0 <= row < 9 and 0 <= col < 9


def is_in_promotion_zone(row: int, is_sente: bool) -> bool:
    """先手なら 0..2 段、後手なら 6..8 段が敵陣。"""
    return row <= 2 if is_sente else row >= 6


def must_promote(piece: str, to_row: int, is_sente: bool) -> bool:
    """到達後に動けなくなるため成りが強制される位置か。"""
    if piece in {"P", "L"}:
        return to_row == (0 if is_sente else 8)
    if piece == "N":
        return to_row in ({0, 1} if is_sente else {7, 8})
    return False


def belongs_to(board: Board, r: int, c: int, owner: int) -> bool:
    cell = board.grid[r][c]
    return cell is not None and cell[1] == owner


def add_step_dst(board: Board, dsts: set, r: int, c: int, deltas, owner: int) -> None:
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if not is_in_bounds(nr, nc):
            continue
        if belongs_to(board, nr, nc, owner):
            continue
        dsts.add((r, c, nr, nc))


def add_slider_dst(board: Board, dsts: set, r: int, c: int, directions, owner: int) -> None:
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while is_in_bounds(nr, nc):
            if belongs_to(board, nr, nc, owner):
                break
            dsts.add((r, c, nr, nc))
            if board.grid[nr][nc] is not None:
                break
            nr += dr
            nc += dc


def piece_movement(board: Board, r: int, c: int, owner: int) -> set[tuple[int, int, int, int]]:
    """(from_r, from_c, to_r, to_c) のセットを返す。"""
    dsts: set = set()
    cell = board.grid[r][c]
    if cell is None or cell[1] != owner:
        return dsts
    piece, _, promoted = cell
    # owner=0(先手) は -1 方向が前、owner=1(後手) は +1 方向が前
    fwd = -1 if owner == 0 else 1

    king_steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    gold_steps = [(fwd, -1), (fwd, 0), (fwd, 1),
                  (0, -1), (0, 1), (-fwd, 0)]
    silver_steps = [(fwd, -1), (fwd, 0), (fwd, 1),
                    (-fwd, -1), (-fwd, 1)]
    knight_steps = [(fwd * 2, -1), (fwd * 2, 1)]

    if promoted and piece in {"P", "L", "N", "S"}:
        add_step_dst(board, dsts, r, c, gold_steps, owner)
        return dsts

    if piece == "K":
        add_step_dst(board, dsts, r, c, king_steps, owner)
    elif piece == "G":
        add_step_dst(board, dsts, r, c, gold_steps, owner)
    elif piece == "S":
        add_step_dst(board, dsts, r, c, silver_steps, owner)
    elif piece == "N":
        add_step_dst(board, dsts, r, c, knight_steps, owner)
    elif piece == "L":
        add_slider_dst(board, dsts, r, c, [(fwd, 0)], owner)
    elif piece == "P":
        add_step_dst(board, dsts, r, c, [(fwd, 0)], owner)
    elif piece == "B":
        add_slider_dst(board, dsts, r, c,
                       [(-1, -1), (-1, 1), (1, -1), (1, 1)], owner)
        if promoted:
            add_step_dst(board, dsts, r, c,
                         [(-1, 0), (0, -1), (0, 1), (1, 0)], owner)
    elif piece == "R":
        add_slider_dst(board, dsts, r, c,
                       [(-1, 0), (1, 0), (0, -1), (0, 1)], owner)
        if promoted:
            add_step_dst(board, dsts, r, c,
                         [(-1, -1), (-1, 1), (1, -1), (1, 1)], owner)
    return dsts


def has_own_pawn_on_file(board: Board, col: int, owner: int) -> bool:
    """指定した筋に自分の(成っていない)歩がいるか。"""
    for r in range(9):
        cell = board.grid[r][col]
        if cell is None:
            continue
        piece, own, promoted = cell
        if piece == "P" and own == owner and not promoted:
            return True
    return False


def build_legal_mask_729(board: Board, is_sente: bool) -> np.ndarray:
    """729次元の合法手マスク。手番側視点(=encode_board と同じ向き)で返す。"""
    mask = np.zeros(NUM_LABELS, dtype=bool)
    owner = 0 if is_sente else 1

    # --- 盤上の駒の移動 ---
    for r in range(9):
        for c in range(9):
            for fr, fc, tr, tc in piece_movement(board, r, c, owner):
                cell = board.grid[fr][fc]
                if cell is None:
                    continue
                piece, _, promoted = cell

                # 「手番側視点」に座標を回転
                if is_sente:
                    tr_n, tc_n = tr, tc
                    fr_n = fr  # 成判定にはfromも必要
                else:
                    tr_n, tc_n = 8 - tr, 8 - tc
                    fr_n = 8 - fr  # 後手視点では fr_n も反転後の座標で判定

                sq_idx = tr_n * 9 + tc_n

                # 1) 不成移動: 必ず成らねばならない位置でなければ可
                if not must_promote(piece, tr, is_sente):
                    mask[sq_idx * 9 + 0] = True

                # 2) 成る移動: 成れる駒で、敵陣絡みなら可
                if (
                    piece in PROMOTABLE and not promoted
                    and (
                        is_in_promotion_zone(tr, is_sente)
                        or is_in_promotion_zone(fr, is_sente)
                    )
                ):
                    mask[sq_idx * 9 + 1] = True

    # --- 持ち駒打ち ---
    for sub_offset, p in enumerate(DROPPABLE):
        if board.hands[owner].get(p, 0) <= 0:
            continue
        for r in range(9):
            for c in range(9):
                if board.grid[r][c] is not None:
                    continue
                # 行き場のない打ち禁止
                if p in {"P", "L"} and r == (0 if is_sente else 8):
                    continue
                if p == "N" and r in (
                    {0, 1} if is_sente else {7, 8}
                ):
                    continue
                # 二歩禁止
                if p == "P" and has_own_pawn_on_file(board, c, owner):
                    continue
                # ラベルは手番側視点
                if is_sente:
                    tr_n, tc_n = r, c
                else:
                    tr_n, tc_n = 8 - r, 8 - c
                sq_idx = tr_n * 9 + tc_n
                mask[sq_idx * 9 + (2 + sub_offset)] = True

    return mask


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="将棋 次の一手予測API (v2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    # 9x9 の盤面。各セルは駒文字列 (例: "P", "+r", "k") か None/空文字
    grid: list[list[Optional[str]]]
    # 持ち駒: {"sente": {"P": 2, ...}, "gote": {"R": 1, ...}}
    hands: dict
    is_sente: bool
    top_k: int = 5


class Prediction(BaseModel):
    row: int
    col: int
    sub: int
    sub_name: str
    label_text: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        # 1) フロントの表現 -> Board へ
        board = board_from_payload(req.grid, req.hands, req.is_sente)

        # 2) encode_board で 42ch テンソル (手番側視点に正規化済)
        x = encode_board(board)
        x = torch.from_numpy(x).unsqueeze(0).to(device)

        # 3) 推論
        with torch.no_grad():
            logits = model(x)[0]

        # 4) 合法手マスクを適用 (-inf で不可手を潰してから softmax)
        mask = build_legal_mask_729(board, req.is_sente)
        mask_t = torch.from_numpy(mask).to(device)
        logits = logits.masked_fill(~mask_t, float("-inf"))

        if not torch.isfinite(logits).any():
            # 合法手が一つもない場合 (詰みなど) は素のsoftmaxにフォールバック
            probs = F.softmax(model(x)[0], dim=0)
        else:
            probs = F.softmax(logits, dim=0)

        # 5) 上位 top_k を返す
        k = max(1, min(int(req.top_k), 20))
        topv, topi = probs.topk(k)
        preds = []
        for prob, idx in zip(topv.tolist(), topi.tolist()):
            if prob <= 0.0:
                continue
            info = label_to_obj(int(idx), req.is_sente)
            info["confidence"] = float(prob)
            preds.append(info)

        return {"predictions": preds}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
