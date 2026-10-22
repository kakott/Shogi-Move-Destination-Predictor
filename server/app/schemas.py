"""APIで使用するリクエスト・レスポンス定義。"""

from typing import Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    # 9x9 の盤面。各セルは駒文字列 (例: "P", "+r", "k") か None/空文字
    grid: list[list[Optional[str]]]
    # 持ち駒: {"sente": {"P": 2, ...}, "gote": {"R": 1, ...}}
    hands: dict
    is_sente: bool
    top_k: int = 5


class Prediction(BaseModel):
    row: int
    col: int
    sub: int
    sub_name: str
    label_text: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
