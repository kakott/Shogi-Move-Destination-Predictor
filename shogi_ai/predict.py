"""
predict.py
==========
学習済みモデルで、棋譜の途中から次の一手を予測する。

使い方:
    python predict.py --model best_model.pt --moves "P7g-7f P3c-3d P6g-6f"
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import ShogiPolicyNet
from shogi_env import (
    Board,
    encode_board,
    label_to_human,
    parse_move,
)


def predict_top_k(model: ShogiPolicyNet, board: Board, k: int = 5,
                  device: str = "cpu") -> list[tuple[int, float, str]]:
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    top = probs.argsort()[::-1][:k]
    return [(int(i), float(probs[i]), label_to_human(int(i))) for i in top]


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    ckpt = torch.load(args.model, map_location=device)
    model = ShogiPolicyNet(
        in_channels=ckpt.get("in_channels", 42),
        ch=ckpt.get("ch", 128),
        n_blocks=ckpt.get("blocks", 6),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"モデル読み込み済 (epoch={ckpt.get('epoch','?')}, "
          f"val_acc={ckpt.get('val_acc',0):.4f})")

    # 棋譜を進める
    board = Board.initial()
    for tok in args.moves.split():
        mv = parse_move(tok)
        if mv is None:
            print(f"[警告] パースできない手: {tok}")
            continue
        if not board.apply(mv):
            print(f"[警告] 適用できない手: {tok}")
            continue
    print(f"局面の手番: {'先手' if board.turn == 0 else '後手'}")

    print(f"\n=== 上位 {args.k} 候補 ===")
    for i, (lab, prob, desc) in enumerate(predict_top_k(model, board, k=args.k, device=device)):
        print(f"  {i+1}. ({prob:.4f}) {desc}   [label={lab}]")
    # 注意: 出力ラベルは「手番側視点」に正規化されている。後手番のとき
    # ラベルの座標は実盤面と上下左右逆になっている点に注意。


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="best_model.pt")
    p.add_argument("--moves", default="",
                   help="スペース区切りの棋譜。空なら初期局面の手を予測。")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    main(args)
