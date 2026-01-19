from fastapi import FastAPI, UploadFile, File
import tempfile
from pydantic import BaseModel
from typing import List
import os
import json
from ASAnalyzer.utils import load_raw_file
from imu_to_asanalyzer_txt import process_imu_txt

import math
from ASAnalyzer import ASAnalyzer
from audio_adapter import wav_to_raw_matrix
from analyzer import AudioAnalyzerService

app = FastAPI()
service = AudioAnalyzerService()
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
DEFAULT_TEST_FILE = "state_data.json"

app = FastAPI()
analyser = ASAnalyzer()

WINDOW_SIZE = 5
WINDOW = 5

def init_analyser_buffers(analyser):
    analyser.raw = []
    analyser.delta = []

    analyser.est_d = []
    analyser.obs_d = []

    analyser.est_p = []
    analyser.obs_p = []

def update_internal_states(analyser, state_vec):
    px, py, pz, vx, vy, vz, ax, ay, az = state_vec

    speed = math.sqrt(vx*vx + vy*vy + vz*vz)
    pos = math.sqrt(px*px + py*py + pz*pz)

    analyser.est_d.append(speed)
    analyser.obs_d.append(speed)

    analyser.est_p.append(pos)
    analyser.obs_p.append(pos)

def load_state_json(path):
    with open(path, "r") as f:
        return json.load(f)

def run_local_test():
    analyser = ASAnalyzer()

    init_analyser_buffers(analyser)

    samples = load_state_json("assets/state_data.json")

    for s in samples:
        state_vec = [
            s["px"], s["py"], s["pz"],
            s["vx"], s["vy"], s["vz"],
            s["ax"], s["ay"], s["az"],
        ]

        analyser.raw.append(state_vec)

        if len(analyser.raw) > 1:
            prev = analyser.raw[-2]
            delta = [a - b for a, b in zip(state_vec, prev)]
        else:
            delta = [0.0] * len(state_vec)

        analyser.delta.append(delta)

        update_internal_states(analyser, state_vec)

        if len(analyser.raw) >= WINDOW_SIZE:
            feats = analyser.extract_window_features(WINDOW_SIZE)
            print("Features:", feats)


def test_real_world_file(path):
    data = load_raw_file(path, delimiter=',')  # 🔑 A FUNÇÃO CERTA

    analyzer = ASAnalyzer()

    features = []

    for frame in data:
        analyzer.add_data(frame)

        if len(analyzer.raw) >= WINDOW:
            feats = analyzer.extract_window_features(WINDOW)
            features.append(feats)

    print("OK | windows:", len(features))
    print("Sample feature:", features[0])

@app.get("/test")
def analyse_window():
    run_local_test()

@app.get("/test2")
def analyse_window():
    test_real_world_file("imu_as.txt")

@app.get("/convert2")
def analyse_window():
    process_imu_txt()

class StateSample(BaseModel):
    timestamp: int
    ax: float
    ay: float
    az: float
    vx: float
    vy: float
    vz: float
    px: float
    py: float
    pz: float

class WindowPayload(BaseModel):
    samples: List[StateSample]

@app.post("/analyse")
def analyse_window(payload: WindowPayload):
    samples_file = os.path.join(ASSETS_DIR, DEFAULT_TEST_FILE)

    for s in payload.samples:
        # 🔑 ORDEM IMPORTANTE
        state_vec = [
            s.px, s.py, s.pz,
            s.vx, s.vy, s.vz,
            s.ax, s.ay, s.az,
        ]

        analyser.raw.append(state_vec)

        # delta (diferença entre estados)
        if len(analyser.raw) > 1:
            prev = analyser.raw[-2]
            delta = [a - b for a, b in zip(state_vec, prev)]
        else:
            delta = [0.0] * len(state_vec)

        analyser.delta.append(delta)

    if len(analyser.raw) < 5:
        return { "status": "waiting_for_more_data" }

    features = analyser.extract_window_features(window_size=5)

    return {
        "status": "ok",
        "features": features
    }

@app.get("/analyze-test")
def analyze_fixed_file():
    csv_path = os.path.join(ASSETS_DIR, DEFAULT_TEST_FILE)

    if not os.path.exists(csv_path):
        return {"error": f"File not found: {csv_path}"}
    
    features = service.test(csv_path)

    # features, meta = service.extract_features_from_csv(csv_path)

    # return {
    #     "file": DEFAULT_TEST_FILE,
    #     "windows": len(features),
    #     "features": features,
    #     "meta": meta,
    # }


@app.get("/convert")
def convertJsonToTxt():
    INPUT_JSON = "assets/state_data.json"
    OUTPUT_TXT = "assets/state_data_converted.txt"

    FIELDS = [
        "px", "py", "pz",
        "vx", "vy", "vz",
        "ax", "ay", "az"
    ]

    with open(INPUT_JSON, "r") as f:
        samples = json.load(f)

    with open(OUTPUT_TXT, "w") as f:
        for s in samples:
            row = [str(float(s[k])) for k in FIELDS]
            f.write(",".join(row) + "\n")