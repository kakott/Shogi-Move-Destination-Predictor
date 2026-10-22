"""将棋の次の一手予測APIのエントリーポイント。"""

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .board_converter import board_from_payload
from .prediction_service import DEVICE, predict_position
from .schemas import PredictRequest, PredictResponse


app = FastAPI(title="将棋 次の一手予測API (v2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        board = board_from_payload(req.grid, req.hands, req.is_sente)
        predictions = predict_position(
            board=board,
            is_sente=req.is_sente,
            top_k=req.top_k,
        )
        return {"predictions": predictions}
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
