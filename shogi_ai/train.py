"""
train.py
========
CNN を学習するスクリプト。Google Colab の GPU で動くことを想定。

使い方:
    python train.py --data dataset.npz --epochs 20 --batch 256

ベスト時に best_model.pt を保存する。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import ShogiPolicyNet


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使うデバイス: {device}")

    # データロード
    print(f"データを読み込み中: {args.data}")
    data = np.load(args.data)
    X, y = data["X"], data["y"]
    print(f"  X.shape={X.shape}, y.shape={y.shape}")

    # tensorに変換
    X_t = torch.from_numpy(X)         # float32
    y_t = torch.from_numpy(y).long()  # int64
    dataset = TensorDataset(X_t, y_t)

    # train/val 分割 (固定seed)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    print(f"  train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device == "cuda"))

    model = ShogiPolicyNet(in_channels=X.shape[1], ch=args.ch,
                           n_blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"モデルのパラメータ数: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        t0 = time.time()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
        sched.step()

        # ---- val ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        top5_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                # top5
                top5 = logits.topk(5, dim=1).indices
                top5_correct += (top5 == yb.unsqueeze(1)).any(dim=1).sum().item()
                val_total += xb.size(0)

        dt = time.time() - t0
        tr_acc = train_correct / train_total
        va_acc = val_correct / val_total
        top5_acc = top5_correct / val_total
        print(
            f"[epoch {epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss / train_total:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss / val_total:.4f} val_acc={va_acc:.4f} "
            f"top5={top5_acc:.4f} | {dt:.1f}s"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "in_channels": X.shape[1],
                "ch": args.ch,
                "blocks": args.blocks,
                "val_acc": va_acc,
                "epoch": epoch,
            }, args.out)
            print(f"  -> ベスト更新: val_acc={va_acc:.4f} を {args.out} に保存")

    print(f"\n完了。ベスト val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset.npz")
    p.add_argument("--out", default="best_model.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    main(args)
