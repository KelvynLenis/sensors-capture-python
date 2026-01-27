# ws_server.py
from fastapi import WebSocket, WebSocketDisconnect
import json
from imu_to_asanalyzer_txt import process_imu_buffer

WINDOW_SIZE = 10

async def imu_ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS conectado")

    buffer = []

    try:
        while True:
            msg = await websocket.receive_text()
            sample = json.loads(msg)

            print("sample:", sample)

            buffer.append(sample.get("data"))

            print("buffer len:", len(buffer))

            # exemplo: janela de 50 amostras
            if len(buffer) >= 32:
                result = process_imu_buffer(buffer)
                buffer.clear()

                await websocket.send_text(
                    json.dumps({"result": result})
                )

    except WebSocketDisconnect:
        print("WS desconectado")