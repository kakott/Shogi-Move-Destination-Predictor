# Shogi-Move-Destination-Predictor
将棋AI（次の盤面の次の手を上から5個予測するAI）

## Dataset

This project uses the **Elite Shogi Games** dataset published on Kaggle by suayptalha.

Dataset: Elite Shogi Games  
Author: suayptalha  
Source: Kaggle  
URL: https://www.kaggle.com/datasets/suayptalha/elite-shogi-games

The dataset is used only for training and evaluation purposes.
The raw dataset files are not included in this repository.
Please download the dataset directly from Kaggle and follow the license terms specified on the dataset page.

## データセットについて

本プロジェクトでは、Kaggleで公開されている suayptalha 氏の
**Elite Shogi Games** データセットを学習および評価に使用しました。

Dataset: Elite Shogi Games  
Author: suayptalha  
Source: Kaggle  
URL: https://www.kaggle.com/datasets/suayptalha/elite-shogi-games

本リポジトリには、元データセットのファイルは含めていません。
データセットを利用する場合は、Kaggle上の配布ページから直接取得し、
当該ページに記載されたライセンス条件に従ってください。

# Shogi AI Next-Move Predictor & Visualizer

深層学習 (PyTorch) を用いた将棋の次の一手予測モデルと、それをブラウザ上でインタラクティブに確認できるReactベースのビジュアライザーです。

## ✨ 特徴

* **Deep Learning Model**: AlphaZeroのアプローチにインスパイアされた軽量なResNetベースのPolicy Network (PyTorch)。42チャンネルの盤面エンコーディングを使用し、729クラスの行動空間から次の一手を予測します。
* **FastAPI Backend**: モデルをホストし、フロントエンドからの推論リクエストを処理する高速なAPIサーバー。合法手マスクを適用し、あり得ない手を自動的に除外します。
* **React Visualizer**: 西洋式棋譜 (Hodges表記) を読み込み、インタラクティブに盤面を進めながらAIの予測結果（候補手と確率、ヒートマップ）を可視化します。

## 📁 プロジェクト構成

* `model.py`: PyTorchによるPolicy Network (CNN) の定義。
* `shogi_env.py`: 盤面の管理、棋譜のパース、CNN用テンソルへのエンコード処理。
* `build_dataset.py`: CSVの棋譜データから学習用の `dataset.npz` を生成。
* `predict.py`: CUIベースでの推論テストスクリプト。
* `server.py`: FastAPIを使った推論バックエンドサーバー。
* `src/` (React Files): `App.jsx`, `shogi_visualizer.jsx` などのフロントエンド実装。

## 🚀 セットアップと起動方法

### 1. バックエンド (APIサーバー) の起動

Python 3.8以上を推奨します。

```bash
# 依存関係のインストール
pip install torch torchvision numpy fastapi uvicorn pandas

# モデルの配置 (ご自身で学習したモデルを配置してください)
# 例: best_model.pt をプロジェクトルートに置く

# サーバーの起動
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
