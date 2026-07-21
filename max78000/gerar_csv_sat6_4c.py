import os

import numpy as np
import pandas as pd
import scipy.io

# AJUSTE ESTE CAMINHO se o .mat estiver em outro lugar
MAT_PATH = "./ai8x-training/data/SAT-6/sat-6-full.mat"

# Pasta onde o disparador_hil.py já procura os CSVs
OUT_DIR = "./dataset"


def export_split(images: np.ndarray, labels: np.ndarray, prefix: str) -> None:
    # Mantém os 4 canais: R, G, B, NIR
    # Formato original do .mat: (H, W, C, N)
    images_4c = images[:, :, :4, :]

    # Converte para (N, C, H, W), igual ao sat6.py
    images_nchw = np.transpose(images_4c, (3, 2, 0, 1)).astype(np.uint8)

    # Achata cada amostra para uma linha do CSV
    x_flat = images_nchw.reshape(images_nchw.shape[0], -1)

    # Labels one-hot: (6, N) -> (N, 6)
    y_rows = labels.T.astype(np.uint8)

    os.makedirs(OUT_DIR, exist_ok=True)

    x_path = os.path.join(OUT_DIR, f"X_{prefix}_sat6.csv")
    y_path = os.path.join(OUT_DIR, f"y_{prefix}_sat6.csv")

    pd.DataFrame(x_flat).to_csv(x_path, header=False, index=False)
    pd.DataFrame(y_rows).to_csv(y_path, header=False, index=False)

    print(f"{prefix}:")
    print(f"  X salvo em: {x_path}")
    print(f"  y salvo em: {y_path}")
    print(f"  Shape X: {x_flat.shape}")
    print(f"  Shape y: {y_rows.shape}")

    sample = images_nchw[0]
    print(f"  Sample 0 shape (C,H,W): {sample.shape}")
    print(f"  Canal 0 primeiros 8 valores: {sample[0].flatten()[:8].tolist()}")
    print(f"  Canal 1 primeiros 8 valores: {sample[1].flatten()[:8].tolist()}")
    print(f"  Canal 2 primeiros 8 valores: {sample[2].flatten()[:8].tolist()}")
    print(f"  Canal 3 primeiros 8 valores: {sample[3].flatten()[:8].tolist()}")
    print(f"  Label sample 0: {y_rows[0].tolist()}")
    print("")


def main() -> None:
    print(f"Carregando arquivo MAT: {MAT_PATH}")
    mat_data = scipy.io.loadmat(MAT_PATH)

    train_x = mat_data["train_x"]
    train_y = mat_data["train_y"]
    test_x = mat_data["test_x"]
    test_y = mat_data["test_y"]

    print("Shapes originais:")
    print(f"  train_x: {train_x.shape}")
    print(f"  train_y: {train_y.shape}")
    print(f"  test_x: {test_x.shape}")
    print(f"  test_y: {test_y.shape}")
    print("")

    export_split(train_x, train_y, "train")
    export_split(test_x, test_y, "test")

    print("CSV 4 canais gerados com sucesso.")


if __name__ == "__main__":
    main()