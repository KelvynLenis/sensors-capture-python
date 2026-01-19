import numpy as np
import csv

def load_sensor_csv(path):
    data = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = [
                float(row["ax"]),
                float(row["ay"]),
                float(row["az"]),
                float(row["gx"]),
                float(row["gy"]),
                float(row["gz"]),
            ]
            data.append(sample)

    data = np.asarray(data, dtype=np.float32)

    # 🔴 MUITO IMPORTANTE
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Invalid data shape: {data.shape}")

    return data
