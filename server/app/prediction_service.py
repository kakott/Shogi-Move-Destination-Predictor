"""学習済みモデルの読み込みと合法手に限定した推論処理。"""

import numpy as np
import torch
import torch.nn.functional as F

from shogi_ai.model import ShogiPolicyNet
from shogi_ai.shogi_env import (
    Board,
    DROPPABLE,
    FILE_CHARS,
    RANK_CHARS,
    encode_board,
)

from .config import MAX_TOP_K, MODEL_PATH, NUM_LABELS
from .legal_moves import build_legal_mask_729


SUB_NAMES = [
    "move",
    "promote",
    "drop_P",
    "drop_L",
    "drop_N",
    "drop_S",
    "drop_G",
    "drop_B",
    "drop_R",
]

PIECE_NAME_JP = {
    "P": "歩",
    "L": "香",
    "N": "桂",
    "S": "銀",
    "G": "金",
    "B": "角",
    "R": "飛",
}


def load_model() -> tuple[ShogiPolicyNet, torch.device]:
    """チェックポイントを読み込み、推論用モデルを構築する。"""
    print("モデル読み込み中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )
    state_dict = (
        checkpoint["model"] if "model" in checkpoint else checkpoint
    )
    checkpoint_blocks = sum(
        1
        for key in state_dict.keys()
        if key.startswith("blocks.") and key.endswith(".conv1.weight")
    )
    block_count = (
        checkpoint_blocks
        if checkpoint_blocks > 0
        else checkpoint.get("blocks", 6)
    )
    channels = checkpoint.get("ch", 128)
    input_channels = checkpoint.get("in_channels", 42)

    model = ShogiPolicyNet(
        in_channels=input_channels,
        ch=channels,
        n_blocks=block_count,
        n_labels=NUM_LABELS,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"モデル読み込み完了: device={device}, "
        f"in_channels={input_channels}, ch={channels}, "
        f"blocks={block_count}, val_acc={checkpoint.get('val_acc', '?')}"
    )
    return model, device


MODEL, DEVICE = load_model()


def label_to_obj(label: int, is_sente: bool) -> dict:
    """モデルのラベルをAPIレスポンス用の情報へ変換する。"""
    square_index, sub = divmod(label, 9)
    normalized_row, normalized_col = divmod(square_index, 9)
    if is_sente:
        row, col = normalized_row, normalized_col
    else:
        row, col = 8 - normalized_row, 8 - normalized_col

    sub_name = SUB_NAMES[sub]
    destination = f"{FILE_CHARS[col]}{RANK_CHARS[row]}"
    if sub == 0:
        label_text = f"-> {destination}"
    elif sub == 1:
        label_text = f"-> {destination} 成"
    else:
        piece = DROPPABLE[sub - 2]
        label_text = (
            f"{PIECE_NAME_JP.get(piece, piece)}打 -> {destination}"
        )

    return {
        "row": row,
        "col": col,
        "sub": sub,
        "sub_name": sub_name,
        "label_text": label_text,
    }


def predict_position(
    board: Board,
    is_sente: bool,
    top_k: int,
) -> list[dict]:
    """局面を推論し、合法手だけを確率順に返す。"""
    legal_mask = build_legal_mask_729(board, is_sente)
    if not legal_mask.any():
        return []

    encoded = encode_board(board)
    model_input = torch.from_numpy(encoded).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(model_input)[0]

    mask_tensor = torch.from_numpy(legal_mask).to(DEVICE)
    logits = logits.masked_fill(~mask_tensor, float("-inf"))
    probabilities = F.softmax(logits, dim=0)

    count = max(1, min(int(top_k), MAX_TOP_K))
    top_values, top_indices = probabilities.topk(count)
    predictions = []
    for probability, index in zip(
        top_values.tolist(), top_indices.tolist()
    ):
        if probability <= 0.0 or not np.isfinite(probability):
            continue
        prediction = label_to_obj(int(index), is_sente)
        prediction["confidence"] = float(probability)
        predictions.append(prediction)

    return predictions
