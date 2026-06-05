# shogi_ai

`shogi_ai` だけで、将棋の次の一手予測モデルのデータ作成、学習、推論確認までできます。
この README は Google Colab で動かす前提の最小ガイドです。

## 含まれているもの

- `build_dataset.py`: `shogi_games.csv` から `dataset.npz` を作る
- `train.py`: `dataset.npz` を使って学習し、`best_model.pt` を保存する
- `predict.py`: 学習済みモデルで次の一手候補を出す
- `model.py`: モデル定義
- `shogi_env.py`: 盤面表現、棋譜パーサ、特徴量化

## Colab で使う流れ

1. ランタイムを `GPU` にする
2. この `shogi_ai` フォルダを Colab に持ち込む
3. `dataset.npz` を作る
4. `train.py` で学習する
5. `predict.py` で推論確認する
6. 必要なら Drive に成果物を保存する

## 方法A: GitHub から読み込む

リポジトリを GitHub に置いているなら、Colab の最初のセルで次を実行します。

```bash
!git clone <YOUR_REPO_URL>
%cd shogi-Move-Destination-Predictor/shogi_ai
!ls
```

`<YOUR_REPO_URL>` は自分の GitHub URL に置き換えてください。

## 方法B: `shogi_ai` フォルダだけアップロードする

GitHub を使わないなら、Colab 左側のファイルペインから `shogi_ai` 一式をアップロードして、
次のように移動します。

```python
%cd /content/shogi_ai
!ls
```

## 依存関係

Colab には `torch` が入っていることが多いですが、最低限これを入れておくと安全です。

```bash
!pip install -q numpy pandas
```

`torch` が無い環境なら追加で入れます。

```bash
!pip install -q torch
```

## 1. データセットを作る

`shogi_games.csv` から学習用データを作ります。
この CSV 自体はリポジトリに含めず、Kaggle から各自ダウンロードする前提です。

取得元:

- https://www.kaggle.com/datasets/suayptalha/elite-shogi-games

```bash
!python build_dataset.py --csv shogi_games.csv --out dataset.npz
```

手数が短い対局を除外したいときは `--min-moves` を変えます。

```bash
!python build_dataset.py --csv shogi_games.csv --out dataset.npz --min-moves 10
```

生成物:

- `dataset.npz`
  - `X`: `(N, 42, 9, 9)`
  - `y`: `(N,)`

## 2. 学習する

まずは標準設定で十分です。

```bash
!python train.py --data dataset.npz --epochs 20 --batch 256 --out best_model.pt
```

GPU を使うと Colab ではかなり速くなります。

よく触る引数:

- `--epochs`: 学習エポック数
- `--batch`: バッチサイズ
- `--lr`: 学習率
- `--ch`: チャンネル数
- `--blocks`: ResNet ブロック数
- `--val-ratio`: 検証データ割合
- `--workers`: DataLoader の worker 数

軽めに試す例:

```bash
!python train.py --data dataset.npz --epochs 5 --batch 128 --workers 2
```

学習が進むと、最良の検証精度で `best_model.pt` が更新されます。

## 3. 推論確認をする

初期局面で候補を見たいとき:

```bash
!python predict.py --model best_model.pt --k 5
```

途中局面から見たいとき:

```bash
!python predict.py --model best_model.pt --moves "P7g-7f P3c-3d P6g-6f" --k 5
```

## 4. Google Drive に保存する

学習結果を残すなら Drive を mount します。

```python
from google.colab import drive
drive.mount('/content/drive')
```

保存例:

```bash
!mkdir -p /content/drive/MyDrive/shogi_ai_outputs
!cp best_model.pt /content/drive/MyDrive/shogi_ai_outputs/
!cp dataset.npz /content/drive/MyDrive/shogi_ai_outputs/
```

## Colab でのおすすめ手順

最短だとこの順です。

```bash
!git clone <YOUR_REPO_URL>
%cd shogi-Move-Destination-Predictor/shogi_ai
!pip install -q numpy pandas
!python build_dataset.py --csv shogi_games.csv --out dataset.npz
!python train.py --data dataset.npz --epochs 20 --batch 256 --out best_model.pt
!python predict.py --model best_model.pt --k 5
```

## よくある詰まりどころ

- `FileNotFoundError`
  - `shogi_ai` ディレクトリにいるか確認してください。
- `CUDA is not available`
  - Colab のランタイムが CPU のままです。`GPU` に変えてください。
- メモリ不足
  - `--batch` を `256` から `128` や `64` に下げてください。
- 学習結果が消えた
  - Colab セッション終了で `/content` は消えます。Drive にコピーしてください。

## 補足

- `dataset.npz` と `best_model.pt` がすでにあるなら、毎回 `build_dataset.py` や `train.py` を回す必要はありません。
- `shogi_ai` は学習用で、`web` と `server` は可視化と API 用です。Colab で AI を作るだけなら `shogi_ai` だけで足ります。
- リポジトリ同梱の `best_model.pt` をそのまま使う運用もできます。
- `shogi_games.csv` は Kaggle 取得前提なので、GitHub には載せない運用を想定しています。
