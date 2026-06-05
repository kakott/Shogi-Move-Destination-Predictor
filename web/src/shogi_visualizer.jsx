import { useState, useEffect, useCallback } from "react";

// ============================================================
// 定数・初期盤面
// ============================================================
const PIECES = [
  "P", "L", "N", "S", "G", "B", "R", "K",
  "p", "l", "n", "s", "g", "b", "r", "k",
  "+P", "+L", "+N", "+S", "+B", "+R",
  "+p", "+l", "+n", "+s", "+b", "+r",
];

const HAND_PIECES = ["P", "L", "N", "S", "G", "B", "R"];
const HAND_DISPLAY_ORDER = ["R", "B", "G", "S", "N", "L", "P"];
const ROW_LABELS = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];

const PIECES_JP = {
  K: "王", k: "玉", R: "飛", r: "飛", B: "角", b: "角",
  G: "金", g: "金", S: "銀", s: "銀", N: "桂", n: "桂",
  L: "香", l: "香", P: "歩", p: "歩",
  "+R": "龍", "+r": "龍", "+B": "馬", "+b": "馬",
  "+S": "全", "+s": "全", "+N": "圭", "+n": "圭",
  "+L": "杏", "+l": "杏", "+P": "と", "+p": "と",
};

const INITIAL_BOARD = () => {
  const board = Array(9).fill(null).map(() => Array(9).fill(null));
  const back = ["l", "n", "s", "g", "k", "g", "s", "n", "l"];

  back.forEach((piece, col) => {
    board[0][col] = piece;
  });
  board[1][1] = "r";
  board[1][7] = "b";
  for (let col = 0; col < 9; col += 1) board[2][col] = "p";

  back.forEach((piece, col) => {
    board[8][col] = piece.toUpperCase();
  });
  board[7][1] = "B";
  board[7][7] = "R";
  for (let col = 0; col < 9; col += 1) board[6][col] = "P";

  return board;
};

const MOVE_RE = /^(\+?[PBRGLSNKpbrglsnk])([\d][a-i])[-x]([\d][a-i])(\+?)$/;
const DROP_RE = /^([PBRGLSNKpbrglsnk])'([\d][a-i])$/;

function sqToRC(sq) {
  return { row: sq.charCodeAt(1) - 97, col: 9 - parseInt(sq[0], 10) };
}

function parseMove(token) {
  let matched = token.match(MOVE_RE);
  if (matched) {
    const [, piece, from, to, promote] = matched;
    return {
      piece,
      from: sqToRC(from),
      to: sqToRC(to),
      promote: promote === "+",
      drop: false,
    };
  }

  matched = token.match(DROP_RE);
  if (matched) {
    const [, piece, to] = matched;
    return {
      piece,
      from: null,
      to: sqToRC(to),
      promote: false,
      drop: true,
    };
  }

  return null;
}

function createEmptyHandCounts() {
  return HAND_PIECES.reduce((acc, piece) => {
    acc[piece] = 0;
    return acc;
  }, {});
}

function createEmptyHands() {
  return {
    sente: createEmptyHandCounts(),
    gote: createEmptyHandCounts(),
  };
}

function cloneBoard(boardState) {
  return boardState.map((row) => [...row]);
}

function cloneHands(handsState) {
  return {
    sente: { ...handsState.sente },
    gote: { ...handsState.gote },
  };
}

function createInitialPosition() {
  return {
    board: INITIAL_BOARD(),
    hands: createEmptyHands(),
  };
}

function normalizeBasePiece(piece) {
  return piece.replace("+", "");
}

function normalizeHandPiece(piece) {
  return normalizeBasePiece(piece).toUpperCase();
}

function toOwnedPiece(piece, isSente, promote = false) {
  const normalized = isSente
    ? normalizeBasePiece(piece).toUpperCase()
    : normalizeBasePiece(piece).toLowerCase();

  return promote ? `+${normalized}` : normalized;
}

function HandPanel({ label, hand, accentColor, borderColor }) {
  const entries = HAND_DISPLAY_ORDER.filter((piece) => hand[piece] > 0);

  return (
    <div style={{
      width: "100%",
      background: "rgba(0,0,0,0.28)",
      border: `1px solid ${borderColor}`,
      borderRadius: "8px",
      padding: "12px 14px",
      boxSizing: "border-box",
      minHeight: "74px",
    }}>
      <div style={{
        fontSize: "12px",
        color: accentColor,
        letterSpacing: "0.08em",
        marginBottom: "10px",
        fontWeight: "600",
      }}>
        {label}
      </div>
      {entries.length > 0 ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
          {entries.map((piece) => (
            <div
              key={piece}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px",
                padding: "6px 10px",
                minWidth: "58px",
                borderRadius: "999px",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(240,192,96,0.16)",
              }}
            >
              <span style={{ fontSize: "17px", color: "#f5dfae", lineHeight: 1 }}>
                {PIECES_JP[piece]}
              </span>
              <span style={{ fontSize: "12px", color: "#c8a86a" }}>
                x{hand[piece]}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: "12px", color: "#6a5a3a", lineHeight: "32px" }}>
          なし
        </div>
      )}
    </div>
  );
}

const SAMPLE_GAME = "P7g-7f P3c-3d P6g-6f P8c-8d R2h-6h S7a-6b K5i-4h K5a-4b K4h-3h K4b-3b K3h-2h G6a-5b S3i-3h S3a-4b S7i-7h P5c-5d B8h-7g S6b-5c P1g-1f P3d-3e G6i-5h S4b-3c P6f-6e P4c-4d P4g-4f S3c-3d G5h-4g G5b-4c S7h-6g P1c-1d S6g-5f B2b-3c P2g-2f P2c-2d S3h-2g K3b-3a P3g-3f P2d-2e";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export default function ShogiVisualizer() {
  const [moves, setMoves] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [board, setBoard] = useState(() => INITIAL_BOARD());
  const [hands, setHands] = useState(createEmptyHands);
  const [prediction, setPrediction] = useState(null);
  const [gameInput, setGameInput] = useState(SAMPLE_GAME);
  const [positionHistory, setPositionHistory] = useState(() => [createInitialPosition()]);
  const [moveHistory, setMoveHistory] = useState([]);

  const loadGame = useCallback((inputText) => {
    const tokens = inputText.trim() ? inputText.trim().split(/\s+/) : [];
    const parsed = tokens.map(parseMove).filter(Boolean);
    const initialPosition = createInitialPosition();

    setMoves(parsed);
    setCurrentStep(0);
    setBoard(initialPosition.board);
    setHands(initialPosition.hands);
    setPositionHistory([initialPosition]);
    setMoveHistory([]);
    setPrediction(null);
  }, []);

  useEffect(() => {
    loadGame(SAMPLE_GAME);
  }, [loadGame]);

  const applyMove = (boardState, handsState, mv, isSente) => {
    const nextBoard = cloneBoard(boardState);
    const nextHands = cloneHands(handsState);
    const handKey = isSente ? "sente" : "gote";
    const { row: tr, col: tc } = mv.to;

    if (!mv.drop && mv.from) {
      const { row: fr, col: fc } = mv.from;
      const movingPiece = nextBoard[fr][fc];
      const capturedPiece = nextBoard[tr][tc];

      if (!movingPiece) {
        return { board: nextBoard, hands: nextHands };
      }

      if (capturedPiece) {
        const capturedForHand = normalizeHandPiece(capturedPiece);
        if (HAND_PIECES.includes(capturedForHand)) {
          nextHands[handKey][capturedForHand] += 1;
        }
      }

      nextBoard[fr][fc] = null;
      nextBoard[tr][tc] = mv.promote ? toOwnedPiece(movingPiece, isSente, true) : movingPiece;
    } else {
      const droppedPiece = normalizeHandPiece(mv.piece);
      if (HAND_PIECES.includes(droppedPiece) && nextHands[handKey][droppedPiece] > 0) {
        nextHands[handKey][droppedPiece] -= 1;
      }
      nextBoard[tr][tc] = toOwnedPiece(mv.piece, isSente);
    }

    return { board: nextBoard, hands: nextHands };
  };

  // ----------------------------------------------------------------
  // 予測API呼び出し
  // - サーバ側 (server/app.py の /predict) には盤面と持ち駒の生データを送る
  // - 学習側と完全に同じテンソル化はサーバが encode_board で行うので
  //   フロントではテンソル化を行わない (整合性のため)
  // ----------------------------------------------------------------
  const fetchPrediction = useCallback(async (boardState, handsState, isSente, expectedMove) => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grid: boardState,
          hands: handsState,
          is_sente: isSente,
          top_k: 5,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      // 正解ラベルを計算 (3要素一致: row, col, sub)
      // expectedMove は parseMove の結果 ({piece, from, to, promote, drop})
      let correctRow = -1, correctCol = -1, correctSub = -1;
      if (expectedMove) {
        correctRow = expectedMove.to.row;
        correctCol = expectedMove.to.col;
        if (expectedMove.drop) {
          // 持ち駒打ち: sub=2..8 (P,L,N,S,G,B,R)
          const dropIdx = HAND_PIECES.indexOf(expectedMove.piece.replace("+", "").toUpperCase());
          correctSub = dropIdx >= 0 ? 2 + dropIdx : -1;
        } else if (expectedMove.promote) {
          correctSub = 1;
        } else {
          correctSub = 0;
        }
      }
      return data.predictions.map((item) => ({
        ...item,
        isCorrect:
          item.row === correctRow &&
          item.col === correctCol &&
          item.sub === correctSub,
      }));
    } catch (error) {
      console.error("予測取得エラー:", error);
      return null;
    }
  }, []);

  const stepForward = () => {
    if (currentStep >= moves.length) return;

    const mv = moves[currentStep];
    const isSente = currentStep % 2 === 0;
    const nextPosition = applyMove(board, hands, mv, isSente);

    setBoard(nextPosition.board);
    setHands(nextPosition.hands);
    setPositionHistory((prev) => [...prev, nextPosition]);
    setMoveHistory((prev) => [...prev, { mv, isSente, step: currentStep }]);
    setCurrentStep((prev) => prev + 1);

    if (currentStep + 1 < moves.length) {
      const nextIsSente = (currentStep + 1) % 2 === 0;
      const expectedMove = moves[currentStep + 1];
      fetchPrediction(nextPosition.board, nextPosition.hands, nextIsSente, expectedMove).then((candidates) => {
        if (candidates) setPrediction(candidates);
      });
    } else {
      setPrediction(null);
    }
  };

  const stepBackward = () => {
    if (currentStep <= 0) return;

    const newStep = currentStep - 1;
    const previousPosition = positionHistory[newStep];

    setCurrentStep(newStep);
    setBoard(previousPosition.board);
    setHands(previousPosition.hands);
    setPositionHistory((prev) => prev.slice(0, newStep + 1));
    setMoveHistory((prev) => prev.slice(0, newStep));

    if (newStep < moves.length) {
      const nextIsSente = newStep % 2 === 0;
      const expectedMove = moves[newStep];
      fetchPrediction(previousPosition.board, previousPosition.hands, nextIsSente, expectedMove).then((candidates) => {
        if (candidates) setPrediction(candidates);
      });
    } else {
      setPrediction(null);
    }
  };

  const isSenteTurn = currentStep % 2 === 0;
  const lastMove = moveHistory[moveHistory.length - 1];

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0f0c08 0%, #1a1209 50%, #0f0c08 100%)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Noto Serif JP', 'Hiragino Mincho ProN', serif",
      padding: "24px",
      color: "#e8d5a3",
    }}>
      <div style={{ textAlign: "center", marginBottom: "28px" }}>
        <h1 style={{
          fontSize: "28px",
          fontWeight: "700",
          letterSpacing: "0.15em",
          color: "#f0c060",
          textShadow: "0 0 30px rgba(240,192,96,0.4)",
          margin: 0,
        }}>
          将棋 次の盤面予測AI
        </h1>
        <p style={{ fontSize: "13px", color: "#8a7a5a", marginTop: "6px", letterSpacing: "0.1em" }}>
          赤いマス = モデルが予測した移動先候補
        </p>
      </div>

      <div style={{ display: "flex", gap: "32px", alignItems: "flex-start", flexWrap: "wrap", justifyContent: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "12px", width: "fit-content" }}>
          <HandPanel
            label="△ 後手の持ち駒"
            hand={hands.gote}
            accentColor="#90b0ff"
            borderColor="rgba(144,176,255,0.22)"
          />

          <div>
            <div style={{ display: "flex", justifyContent: "flex-end", paddingRight: "2px", marginBottom: "2px" }}>
              {[9, 8, 7, 6, 5, 4, 3, 2, 1].map((num) => (
                <div key={num} style={{ width: "52px", textAlign: "center", fontSize: "12px", color: "#8a7a5a" }}>
                  {num}
                </div>
              ))}
            </div>

            <div style={{ display: "flex" }}>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(9, 52px)",
                gridTemplateRows: "repeat(9, 52px)",
                border: "2px solid #6b5a2a",
                boxShadow: "0 0 40px rgba(240,192,96,0.15), inset 0 0 20px rgba(0,0,0,0.5)",
                background: "linear-gradient(145deg, #d4a851 0%, #c49640 50%, #b8882e 100%)",
                borderRadius: "2px",
              }}>
                {board.map((row, r) =>
                  row.map((piece, c) => {
                    const pred = prediction?.find((item) => item.row === r && item.col === c);
                    const isLastTo = lastMove && lastMove.mv.to.row === r && lastMove.mv.to.col === c;
                    const isGotePiece = piece && (
                      piece === piece.toLowerCase() ||
                      (piece.startsWith("+") && piece.slice(1) === piece.slice(1).toLowerCase())
                    );

                    return (
                      <div
                        key={`${r}-${c}`}
                        style={{
                          width: "52px",
                          height: "52px",
                          border: "0.5px solid rgba(107,90,42,0.6)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          position: "relative",
                          background: pred?.isCorrect
                            ? "rgba(220,50,50,0.35)"
                            : pred
                              ? `rgba(220,50,50,${pred.confidence * 0.25})`
                              : isLastTo
                                ? "rgba(255,220,80,0.3)"
                                : "transparent",
                          transition: "background 0.3s ease",
                          cursor: "default",
                        }}
                      >
                        {pred && (
                          <div style={{
                            position: "absolute",
                            inset: "2px",
                            border: pred.isCorrect ? "2px solid #ff3333" : "1.5px solid rgba(255,80,80,0.6)",
                            borderRadius: "2px",
                            pointerEvents: "none",
                            boxShadow: pred.isCorrect ? "inset 0 0 8px rgba(255,50,50,0.3)" : "none",
                          }} />
                        )}

                        {pred && (
                          <div style={{
                            position: "absolute",
                            bottom: "2px",
                            left: "2px",
                            right: "2px",
                            height: "3px",
                            background: "rgba(0,0,0,0.3)",
                            borderRadius: "2px",
                          }}>
                            <div style={{
                              height: "100%",
                              width: `${pred.confidence * 100}%`,
                              background: pred.isCorrect ? "#ff4444" : "#ff8888",
                              borderRadius: "2px",
                              transition: "width 0.5s ease",
                            }} />
                          </div>
                        )}

                        {piece && (
                          <div style={{
                            fontSize: piece.startsWith("+") ? "14px" : "18px",
                            fontWeight: "700",
                            color: "#1a0a00",
                            transform: isGotePiece ? "rotate(180deg)" : "none",
                            textShadow: piece.startsWith("+") ? "0 0 6px rgba(180,50,0,0.8)" : "none",
                            filter: isLastTo ? "drop-shadow(0 0 4px rgba(255,200,0,0.8))" : "none",
                            lineHeight: 1,
                            userSelect: "none",
                            position: "relative",
                            zIndex: 1,
                          }}>
                            {PIECES_JP[piece] || piece}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>

              <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-around", paddingLeft: "6px" }}>
                {ROW_LABELS.map((label) => (
                  <div key={label} style={{ fontSize: "12px", color: "#8a7a5a", lineHeight: "52px" }}>
                    {label}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <HandPanel
            label="▲ 先手の持ち駒"
            hand={hands.sente}
            accentColor="#f0c060"
            borderColor="rgba(240,192,96,0.24)"
          />

          <div style={{ display: "flex", gap: "12px", marginTop: "8px", justifyContent: "center", alignItems: "center" }}>
            <button
              onClick={stepBackward}
              disabled={currentStep === 0}
              style={{
                padding: "10px 24px",
                background: currentStep === 0 ? "rgba(255,255,255,0.05)" : "rgba(240,192,96,0.15)",
                border: "1px solid rgba(240,192,96,0.3)",
                borderRadius: "4px",
                color: currentStep === 0 ? "#4a3a2a" : "#f0c060",
                cursor: currentStep === 0 ? "not-allowed" : "pointer",
                fontSize: "16px",
                transition: "all 0.2s",
              }}
            >
              ◀ 前
            </button>

            <div style={{
              padding: "8px 20px",
              background: "rgba(0,0,0,0.3)",
              border: "1px solid rgba(240,192,96,0.2)",
              borderRadius: "4px",
              fontSize: "14px",
              color: "#c8a86a",
              minWidth: "80px",
              textAlign: "center",
            }}>
              {currentStep} / {moves.length} 手
            </div>

            <button
              onClick={stepForward}
              disabled={currentStep >= moves.length}
              style={{
                padding: "10px 24px",
                background: currentStep >= moves.length ? "rgba(255,255,255,0.05)" : "rgba(240,192,96,0.15)",
                border: "1px solid rgba(240,192,96,0.3)",
                borderRadius: "4px",
                color: currentStep >= moves.length ? "#4a3a2a" : "#f0c060",
                cursor: currentStep >= moves.length ? "not-allowed" : "pointer",
                fontSize: "16px",
                transition: "all 0.2s",
              }}
            >
              次 ▶
            </button>
          </div>

          <div style={{ textAlign: "center", marginTop: "4px", fontSize: "14px" }}>
            <span style={{
              padding: "4px 16px",
              background: isSenteTurn ? "rgba(240,192,96,0.2)" : "rgba(100,150,255,0.2)",
              border: `1px solid ${isSenteTurn ? "rgba(240,192,96,0.5)" : "rgba(100,150,255,0.5)"}`,
              borderRadius: "20px",
              color: isSenteTurn ? "#f0c060" : "#90b0ff",
            }}>
              {isSenteTurn ? "▲ 先手番" : "△ 後手番"}
            </span>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "16px", minWidth: "240px", maxWidth: "280px" }}>
          <div style={{
            background: "rgba(0,0,0,0.4)",
            border: "1px solid rgba(240,192,96,0.2)",
            borderRadius: "8px",
            padding: "16px",
          }}>
            <h3 style={{ margin: "0 0 12px 0", fontSize: "13px", color: "#f0c060", letterSpacing: "0.1em" }}>
              🎯 モデルの予測候補
            </h3>
            {prediction ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {prediction.map((item, index) => {
                  const col = 9 - item.col;
                  // 表示: "7六 不成" / "3四 成" / "歩打 -> 5五" など
                  const squareLabel = `${col}${ROW_LABELS[item.row]}`;
                  let subLabel = "";
                  if (item.sub === 1) subLabel = " 成";
                  else if (item.sub >= 2) {
                    const piece = HAND_PIECES[item.sub - 2];
                    const pieceJp = { P: "歩", L: "香", N: "桂", S: "銀", G: "金", B: "角", R: "飛" }[piece] || piece;
                    subLabel = ` (${pieceJp}打)`;
                  }
                  return (
                    <div key={index} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span style={{ fontSize: "11px", color: item.isCorrect ? "#ff6666" : "#8a7a5a", minWidth: "16px" }}>
                        {index + 1}.
                      </span>
                      <span style={{
                        fontSize: "13px",
                        color: item.isCorrect ? "#ff8888" : "#c8a86a",
                        minWidth: "70px",
                        fontWeight: item.isCorrect ? "700" : "400",
                      }}>
                        {squareLabel}{subLabel}
                        {item.isCorrect && " ✓"}
                      </span>
                      <div style={{ flex: 1, height: "6px", background: "rgba(255,255,255,0.1)", borderRadius: "3px" }}>
                        <div style={{
                          height: "100%",
                          width: `${item.confidence * 100}%`,
                          background: item.isCorrect
                            ? "linear-gradient(90deg, #ff4444, #ff8888)"
                            : "linear-gradient(90deg, #6a5a3a, #8a7a5a)",
                          borderRadius: "3px",
                          transition: "width 0.5s ease",
                        }} />
                      </div>
                      <span style={{ fontSize: "11px", color: "#8a7a5a", minWidth: "36px", textAlign: "right" }}>
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p style={{ fontSize: "13px", color: "#6a5a3a", margin: 0 }}>
                「次へ」を押すと予測が出ます
              </p>
            )}
          </div>

          <div style={{
            background: "rgba(0,0,0,0.4)",
            border: "1px solid rgba(240,192,96,0.2)",
            borderRadius: "8px",
            padding: "16px",
          }}>
            <h3 style={{ margin: "0 0 10px 0", fontSize: "13px", color: "#f0c060", letterSpacing: "0.1em" }}>
              📋 棋譜を入力
            </h3>
            <textarea
              value={gameInput}
              onChange={(event) => setGameInput(event.target.value)}
              style={{
                width: "100%",
                height: "100px",
                background: "rgba(0,0,0,0.5)",
                border: "1px solid rgba(240,192,96,0.2)",
                borderRadius: "4px",
                color: "#c8a86a",
                fontSize: "11px",
                padding: "8px",
                resize: "vertical",
                fontFamily: "monospace",
                boxSizing: "border-box",
              }}
              placeholder="棋譜を貼り付け..."
            />
            <button
              onClick={() => loadGame(gameInput)}
              style={{
                marginTop: "8px",
                width: "100%",
                padding: "8px",
                background: "rgba(240,192,96,0.2)",
                border: "1px solid rgba(240,192,96,0.4)",
                borderRadius: "4px",
                color: "#f0c060",
                cursor: "pointer",
                fontSize: "13px",
              }}
            >
              読み込む
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
