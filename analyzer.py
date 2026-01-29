from ASAnalyzer import ASAnalyzer
from ASAnalyzer.constant import SETTINGS_PRESET
import numpy as np

class AudioAnalyzerService:
    def __init__(self):
        self.settings = SETTINGS_PRESET["GENERAL"]


    def test(self, file):
        asa = ASAnalyzer(self.settings)
        asa.extract_window_features(file)
        print(asa)
