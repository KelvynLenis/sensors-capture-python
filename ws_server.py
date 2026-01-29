# ws_server.py
from fastapi import WebSocket, WebSocketDisconnect
import json
from ASAnalyzer import ASAnalyzer
from imu_to_asanalyzer_txt import process_window
import numpy as np

FRAME_SIZE = 9
WINDOW = 32
MIN_IMU_SAMPLES = 128
HOP_SIZE = 16          # overlap de 50%
WARMUP_WINDOWS = 5

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


async def imu_ws_endpoint2(websocket: WebSocket):
    await websocket.accept()
    print("[WS] conectado")

    analyzer = ASAnalyzer()

    buffer = []

    features = []

    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)

            data = payload.get("data")

            buffer.append(data)

            # print("len(buffer):", len(buffer))

            if len(buffer) >= MIN_IMU_SAMPLES:
                window_nd = np.array(buffer, dtype=float)

                frames = process_window(window_nd)

                print("len(frames):", len(frames))

                for frame in frames:
                    analyzer.add_data(frame)

                    if len(analyzer.raw) >= WARMUP_WINDOWS:
                        feats = analyzer.extract_window_features(WINDOW)
                        features.append(feats)
                    
                    if len(buffer) > WINDOW + HOP_SIZE:
                        buffer = buffer[-(WINDOW + HOP_SIZE):]

                if len(features) >= 50:
                    features = features[-50:]

                print("OK | Features:", len(features))
                # print("Sample feature:", features)

                await websocket.send_text(json.dumps({
                    "type": "result",
                    "features": features
                }))

    except WebSocketDisconnect:
        print("[WS] desconectado")
    except Exception as e:
        print("[WS] erro:", e)
