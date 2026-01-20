from fastapi import FastAPI
import os
import json
from ASAnalyzer.utils import load_raw_file
from imu_to_asanalyzer_txt import process_imu_txt

from ASAnalyzer import ASAnalyzer
from analyzer import AudioAnalyzerService

app = FastAPI()
service = AudioAnalyzerService()
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "dataset")
DEFAULT_TEST_FILE = "state_data.json"

app = FastAPI()
analyser = ASAnalyzer()

WINDOW = 5

# v1.0.1

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

@app.get("/test2")
def analyse_window():
    test_real_world_file("imu_as.txt")

@app.get("/convert2")
def analyse_window():
    process_imu_txt()