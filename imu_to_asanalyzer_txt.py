import numpy as np

# ==============================
# CONFIGURAÇÃO
# ==============================
INPUT_FILE = "assets/imu_data.txt"
OUTPUT_FILE = "imu_as.txt"

WINDOW_SIZE = 32     # igual ao Java (ajustável)
FS = 50               # Hz (assumido fixo, sem timestamp)

# ==============================
# LOAD TXT (ax ay az)
# ==============================
def load_imu_txt(path):
    return np.loadtxt(path)  # shape: (N, 3)

# ==============================
# FFT MAGNITUDE
# ==============================
def fft_magnitude(window):
    fft = np.fft.rfft(window)
    return np.abs(fft)

# ==============================
# PIPELINE
# ==============================
def process_imu_txt():
    imu = load_imu_txt(INPUT_FILE)

    if imu.ndim != 2 or imu.shape[1] != 3:
        raise ValueError("Formato inválido: esperado TXT com 3 colunas (ax ay az)")

    # magnitude da aceleração
    acc_mag = np.linalg.norm(imu, axis=1)

    frames = []

    for i in range(0, len(acc_mag) - WINDOW_SIZE, WINDOW_SIZE):
        window = acc_mag[i:i + WINDOW_SIZE]

        # rejeição de silêncio
        if np.std(window) < 1e-6:
            continue

        mag = fft_magnitude(window)

        # normalização segura
        norm = np.linalg.norm(mag)
        if norm == 0:
            continue

        mag = mag / norm
        frames.append(mag)

    frames = np.array(frames)

    if frames.size == 0:
        raise RuntimeError("Nenhuma janela válida gerada")

    # ==============================
    # EXPORTA TXT (real_world-like)
    # ==============================
    with open(OUTPUT_FILE, "w") as f:
        for row in frames:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")

    print(f"[OK] {OUTPUT_FILE} gerado | shape={frames.shape}")

# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    process_imu_txt()
