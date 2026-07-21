import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent
AI8X_DIR = ROOT_DIR / "ai8x-training"

sys.path.insert(0, str(AI8X_DIR))

import ai8x  # noqa: E402
from models.sat6_net import sat6_net  # noqa: E402

N_IMAGENS = 200
CHECKPOINT = AI8X_DIR / "logs" / "2026.05.25-151645" / "checkpoint-q.pth.tar"


def preparar_imagem(img_chw: np.ndarray) -> np.ndarray:
    img_hwc = np.transpose(img_chw, (1, 2, 0))

    img_signed = np.clip(
        img_hwc.astype(np.int16) - 128,
        -128,
        127,
    ).astype(np.int8)

    return np.transpose(img_signed, (2, 0, 1))


def main() -> None:
    print("Configurando MAX78000 em modo simulado...")
    ai8x.set_device(85, simulate=True, round_avg=False, verbose=False)

    print("Carregando CSV...")
    df_x = pd.read_csv(
        ROOT_DIR / "dataset" / "X_test_sat6.csv",
        header=None,
        nrows=N_IMAGENS,
        dtype=np.uint8,
    )
    df_y = pd.read_csv(
        ROOT_DIR / "dataset" / "y_test_sat6.csv",
        header=None,
        nrows=N_IMAGENS,
    )

    print("Carregando modelo...")
    model = sat6_net(pretrained=False)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    acertos = 0

    with torch.no_grad():
        for i in range(N_IMAGENS):
            img_chw = df_x.values[i].reshape(4, 28, 28)
            img = preparar_imagem(img_chw)

            x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            y_real = int(np.argmax(df_y.values[i]))

            out = model(x)
            pred = int(torch.argmax(out, dim=1).item())

            if pred == y_real:
                acertos += 1
            else:
                logits = out.numpy().tolist()[0]
                print(f"ERRO img {i + 1}: real={y_real}, pc={pred}, logits={logits}")

    acc = (acertos / N_IMAGENS) * 100

    print("=" * 40)
    print(f"Acurácia PC no CSV: {acc:.2f}%")
    print(f"Acertos: {acertos}/{N_IMAGENS}")
    print("=" * 40)


if __name__ == "__main__":
    main()