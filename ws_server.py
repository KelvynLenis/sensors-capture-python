# ws_server.py
from fastapi import WebSocket, WebSocketDisconnect
import json
from ASAnalyzer import ASAnalyzer
from imu_to_asanalyzer_txt import process_imu_buffer
import numpy as np

FRAME_SIZE = 9
WINDOW = 32

async def imu_ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] conectado")

    analyzer = ASAnalyzer()

    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)

            # print("[WS] payload:", payload)

            if payload.get("type") != "imu_frame":
                continue

            data = payload.get("data")

            # =========================
            # validação mínima
            # =========================
            if (
                not isinstance(data, list)
                or len(data) != FRAME_SIZE
            ):
                print("[WS] frame inválido:", data)
                continue

            try:
                frame = np.array(data, dtype=float)
            except Exception:
                print("[WS] erro ao converter frame")
                continue

            # =========================
            # adiciona ao ASAnalyzer
            # =========================
            analyzer.add_data(frame)

            # debug útil
            # print("len(raw):", len(analyzer.raw))

            # =========================
            # janela pronta
            # =========================
            if len(analyzer.raw) >= WINDOW:
                try:
                    feats = analyzer.extract_window_features(WINDOW)
                except ZeroDivisionError:
                    # janela numericamente inválida
                    # print("[WS] janela numericamente inválida")
                    continue

                await websocket.send_text(json.dumps({
                    "type": "result",
                    "features": feats
                }))

    except WebSocketDisconnect:
        print("[WS] desconectado")
    except Exception as e:
        print("[WS] erro:", e)
