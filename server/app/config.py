"""APIサーバーの設定値。"""

from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SERVER_DIR.parent
SHOGI_AI_DIR = PROJECT_ROOT / "shogi_ai"
MODEL_PATH = SHOGI_AI_DIR / "best_model.pt"

NUM_LABELS = 729
MAX_TOP_K = 20
