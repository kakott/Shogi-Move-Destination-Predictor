# Shogi Move Destination Predictor

将棋の次の一手予測を扱うリポジトリです。責務ごとにディレクトリを分けています。

## Structure

- `web/`: React + Vite のフロントエンド
- `server/`: FastAPI の推論 API
- `shogi_ai/`: 学習コード、前処理、モデル定義、学習成果物

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

- `server/` は `shogi_ai/` の `model.py`、`shogi_env.py`、`best_model.pt` を参照します。
- `shogi_ai/best_model.pt` はリポジトリに同梱する想定です。
- `shogi_ai/dataset.npz` は再生成できる中間生成物として `.gitignore` に入れています。
- `shogi_ai/shogi_games.csv` は Kaggle から各自取得する前提で、リポジトリには含めない方針です。
- 学習データの取得元は Kaggle の `Elite Shogi Games` を想定します:
  https://www.kaggle.com/datasets/suayptalha/elite-shogi-games

## License

- このリポジトリは [MIT License](./LICENSE) です。
- コードと同梱している `shogi_ai/best_model.pt` は、特に別記がない限り同じ MIT License で扱う想定です。
- ただし、学習元データ `shogi_games.csv` は第三者データなので、このリポジトリから再配布しない方針にします。
