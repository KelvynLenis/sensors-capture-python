import numpy as np
import librosa

def wav_to_raw_matrix(
    wav_path: str,
    target_sr: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160
):
    """
    Converte WAV em matriz compatível com ASAnalyzer.
    Retorna array shape (N, 4)
    """

    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    # RMS
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    features = np.vstack([rms, zcr, centroid, bandwidth]).T

    return features.astype(np.float32)
