import numpy as np
import librosa


def wav_to_features(
    wav_path: str,
    sr: int = 16000,
    frame_ms: int = 25,
    hop_ms: int = 10
) -> np.ndarray:
    """
    Converte um arquivo WAV em uma matriz de features (NxM)
    compatível com o ASAnalyzer.

    Retorno:
        np.ndarray shape (num_frames, num_features)
    """

    # 1️⃣ Carregar áudio (mono, 16kHz)
    audio, sr = librosa.load(
        wav_path,
        sr=sr,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Arquivo de áudio vazio")

    # 2️⃣ Parâmetros de frame
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    # 3️⃣ Framing
    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    ).T  # shape: (N, frame_length)

    # 4️⃣ Features básicas (leves e eficazes)
    energy = np.sum(frames ** 2, axis=1)

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    # 5️⃣ Garantir alinhamento de tamanho
    min_len = min(
        len(energy),
        len(zcr),
        len(centroid),
        len(bandwidth)
    )

    features = np.stack([
        energy[:min_len],
        zcr[:min_len],
        centroid[:min_len],
        bandwidth[:min_len]
    ], axis=1)

    # 6️⃣ Normalização simples (opcional, mas recomendada)
    features = np.nan_to_num(features)
    features = (features - features.mean(axis=0)) / (
        features.std(axis=0) + 1e-8
    )

    return features
