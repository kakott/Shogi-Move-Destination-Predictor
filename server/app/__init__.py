"""将棋予測APIパッケージ。"""

from __future__ import annotations

import sys
from pathlib import Path

# `cd server && uvicorn app.main:app` でも学習側パッケージを読めるようにする。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
