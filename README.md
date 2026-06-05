# Shogi Move Destination Predictor

将棋の次の一手予測モデルと、その推論結果を確認するための Web アプリです。

## Structure

- `web/`: React + Vite のフロントエンド
- `server/`: FastAPI の推論 API
- `shogi_ai/`: 学習コード、前処理、モデル定義、学習成果物

## Dataset

このプロジェクトは Kaggle の `Elite Shogi Games` を学習データ取得元として想定しています。

- Dataset: `Elite Shogi Games`
- Author: `suayptalha`
- URL: `https://www.kaggle.com/datasets/suayptalha/elite-shogi-games`

方針:

- `shogi_games.csv` はリポジトリに含めません
- 必要な場合は Kaggle から各自ダウンロードしてください
- 学習済みモデル `shogi_ai/best_model.pt` はリポジトリに同梱します

## Local Development

### API server

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

`GET /health` で疎通確認できます。

### Web app

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

`VITE_API_BASE_URL` を変えると、接続先 API を切り替えられます。未指定時は `http://localhost:8000` を使います。

## Notes

- `server/` は `shogi_ai/` の `model.py`、`shogi_env.py`、`best_model.pt` を参照します
- `shogi_ai/dataset.npz` は再生成できる中間生成物として `.gitignore` に入れています
- `shogi_ai/shogi_games.csv` は Kaggle 取得前提なので `.gitignore` に入れています

## License

- このリポジトリは [MIT License](./LICENSE) です
- 同梱している `shogi_ai/best_model.pt` も、特に別記がない限り同じ MIT License で扱う想定です
- 学習元データ `shogi_games.csv` は第三者データなので、このリポジトリから再配布しない方針です
