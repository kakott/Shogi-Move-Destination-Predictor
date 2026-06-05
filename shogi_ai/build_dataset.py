"""
build_dataset.py
================
CSVの棋譜列をパースして「各局面の (盤面テンソル, 次の手ラベル)」を
numpy 配列にまとめて保存する。

使い方:
    python build_dataset.py --csv shogi_games.csv --out dataset.npz

生成物:
    dataset.npz
        X: (N, 42, 9, 9) float32  -- 盤面テンソル
        y: (N,)         int64    -- 次の手のラベル (0..728)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from shogi_env import (
    Board,
    encode_board,
    normalize_label_for_turn,
    parse_move,
)


def build(csv_path: str, out_path: str, min_moves: int = 10) -> None:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    print(f"読み込んだ対局数: {len(df)}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    n_games_ok = 0
    n_games_skipped = 0
    n_parse_fail = 0
    n_apply_fail = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        moves = row.get("moves")
        if not isinstance(moves, str):
            n_games_skipped += 1
            continue
        tokens = moves.split()
        if len(tokens) < min_moves:
            n_games_skipped += 1
            continue

        board = Board.initial()
        ok_in_game = 0
        for tok in tokens:
            mv = parse_move(tok)
            if mv is None:
                n_parse_fail += 1
                break  # 1手でも壊れたらこの対局は打ち切り (整合性のため)
            # ラベル化 (適用前の手番でラベルを取る)
            label = normalize_label_for_turn(mv, board.turn)
            # この局面の入力テンソル
            x = encode_board(board)
            # 適用
            if not board.apply(mv):
                n_apply_fail += 1
                break
            X_list.append(x)
            y_list.append(label)
            ok_in_game += 1

        if ok_in_game > 0:
            n_games_ok += 1

        if (i + 1) % 100 == 0:
            print(f"  進捗: {i + 1}/{len(df)} 対局処理, "
                  f"局面数={len(X_list)}")

    print(f"\n=== 集計 ===")
    print(f"  完走/部分使用 対局: {n_games_ok}")
    print(f"  スキップ対局: {n_games_skipped}")
    print(f"  パース失敗 (途中で打ち切り): {n_parse_fail}")
    print(f"  適用失敗 (途中で打ち切り): {n_apply_fail}")
    print(f"  最終 局面数: {len(X_list)}")

    if not X_list:
        print("局面が0件です。中断します。", file=sys.stderr)
        sys.exit(1)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    print(f"  X.shape={X.shape}, y.shape={y.shape}")
    print(f"  X.dtype={X.dtype}, y.dtype={y.dtype}")
    print(f"  保存先: {out_path}")
    np.savez_compressed(out_path, X=X, y=y)
    print("完了")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="dataset.npz")
    p.add_argument("--min-moves", type=int, default=10)
    args = p.parse_args()
    build(args.csv, args.out, args.min_moves)
