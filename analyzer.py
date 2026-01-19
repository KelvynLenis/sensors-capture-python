from ASAnalyzer import ASAnalyzer
from ASAnalyzer.constant import SETTINGS_PRESET
from csv_loader import load_sensor_csv
import numpy as np

class AudioAnalyzerService:
    def __init__(self):
        self.settings = SETTINGS_PRESET["GENERAL"]


    def test(self, file):
        asa = ASAnalyzer(self.settings)
        asa.extract_window_features(file)
        print(asa)

    def extract_features_from_csv(self, csv_path, window=5):
        raw = load_sensor_csv(csv_path)

        print("raw shape:", raw.shape)

        asa = ASAnalyzer(self.settings)
        features = []

        WARMUP = window * 2  # 🔥 aquecimento mínimo

        for i in range(raw.shape[0]):
            asa.add_data(raw[i])

            # 🟡 warm-up: não extrai nada
            if i < WARMUP:
                continue

            print("delta len:", len(asa.delta))
            print("delta shape:", np.array(asa.delta).shape)

            # 🟢 só extrai após warm-up e em múltiplos de window
            if len(asa.delta) >= window and len(asa.delta) % window == 0:
                feat = asa.extract_window_features(window)
                if feat is not None:
                    features.append(feat)

        return features, {
            "samples": raw.shape[0],
            "windows": len(features),
            "warmup": WARMUP,
        }
