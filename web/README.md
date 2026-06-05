# Web

将棋の局面ビジュアライザー兼、次の一手予測 UI です。

## Run

```bash
cp .env.example .env.local
npm install
npm run dev
```

## Environment

- `VITE_API_BASE_URL`
  - 既定値: `http://localhost:8000`
  - FastAPI サーバのベース URL
