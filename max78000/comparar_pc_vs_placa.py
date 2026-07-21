import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial
import torch

ROOT_DIR = Path(__file__).resolve().parent
AI8X_TRAINING_DIR = ROOT_DIR / "ai8x-training"

sys.path.insert(0, str(AI8X_TRAINING_DIR))

import ai8x  # noqa: E402
from models.sat6_net import sat6_net  # noqa: E402

PORTA_COM = "COM8"
BAUD_RATE = 115200
N_IMAGENS = 50

CHECKPOINT_PATH = ROOT_DIR / "ai8x-training" / "logs" / "2026.04.14-122703" / "best-q.pth.tar"
X_TEST_PATH = ROOT_DIR / "dataset" / "X_test_sat6.csv"
Y_TEST_PATH = ROOT_DIR / "dataset" / "y_test_sat6.csv"


def carregar_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Checkpoint em formato inesperado.")

    state_dict_limpo = {}

    for key, value in state_dict.items():
        nova_key = key
        if nova_key.startswith("module."):
            nova_key = nova_key[len("module."):]
        state_dict_limpo[nova_key] = value

    missing, unexpected = model.load_state_dict(state_dict_limpo, strict=False)

    print("Checkpoint carregado.")
    if missing:
        print("Chaves faltando no modelo:")
        for item in missing:
            print(f"  - {item}")

    if unexpected:
        print("Chaves inesperadas no checkpoint:")
        for item in unexpected:
            print(f"  - {item}")


def preparar_input_pc_float(img_chw_uint8: np.ndarray) -> torch.Tensor:
    img_float = (img_chw_uint8.astype(np.float32) - 128.0) / 128.0
    return torch.tensor(img_float, dtype=torch.float32).unsqueeze(0)


def preparar_input_pc_simulado(img_chw_uint8: np.ndarray) -> torch.Tensor:
    img_int = img_chw_uint8.astype(np.float32) - 128.0
    return torch.tensor(img_int, dtype=torch.float32).unsqueeze(0)


def preparar_bytes_placa(img_chw_uint8: np.ndarray) -> bytes:
    img_hwc = np.transpose(img_chw_uint8, (1, 2, 0))
    img_signed = np.clip(
        img_hwc.astype(np.int16) - 128,
        -128,
        127,
    ).astype(np.int8)

    return img_signed.tobytes()


def esperar_ack(ser: serial.Serial) -> bool:
    inicio_espera = time.time()

    while True:
        if ser.in_waiting > 0:
            if ser.read(1) == b"K":
                return True

        if time.time() - inicio_espera > 2:
            return False


def enviar_bytes(ser: serial.Serial, payload: bytes) -> None:
    chunk_size = 16

    for i in range(0, len(payload), chunk_size):
        ser.write(payload[i:i + chunk_size])
        time.sleep(0.001)


def ler_resposta(ser: serial.Serial, timeout: float = 2.0) -> str:
    resposta_bruta = ""
    tempo_limite = time.time() + timeout

    while time.time() < tempo_limite:
        if ser.in_waiting > 0:
            resposta_bruta += ser.read(ser.in_waiting).decode("utf-8", errors="ignore")

            if re.search(r"R:\d", resposta_bruta):
                break

        time.sleep(0.01)

    return resposta_bruta


def extrair_classe_placa(resposta_bruta: str) -> int | None:
    match = re.search(r"R:(\d)", resposta_bruta)

    if match:
        return int(match.group(1))

    return None


def criar_modelo(simulate: bool) -> torch.nn.Module:
    ai8x.set_device(device=85, simulate=simulate, round_avg=False, verbose=False)
    model = sat6_net(pretrained=False)
    model.eval()
    carregar_checkpoint(model, CHECKPOINT_PATH)
    return model


def inferencia_pc_float(model: torch.nn.Module, img_chw: np.ndarray) -> tuple[int, np.ndarray]:
    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)

    with torch.no_grad():
        x = preparar_input_pc_float(img_chw)
        logits = model(x)
        classe = int(torch.argmax(logits, dim=1).item())
        logits_np = logits.squeeze(0).detach().cpu().numpy()

    return classe, logits_np


def inferencia_pc_simulada(model: torch.nn.Module, img_chw: np.ndarray) -> tuple[int, np.ndarray]:
    ai8x.set_device(device=85, simulate=True, round_avg=False, verbose=False)

    with torch.no_grad():
        x = preparar_input_pc_simulado(img_chw)
        logits = model(x)
        classe = int(torch.argmax(logits, dim=1).item())
        logits_np = logits.squeeze(0).detach().cpu().numpy()

    return classe, logits_np


def main() -> None:
    print("Carregando CSVs...")
    df_x = pd.read_csv(X_TEST_PATH, header=None, nrows=N_IMAGENS, dtype=np.uint8)
    df_y = pd.read_csv(Y_TEST_PATH, header=None, nrows=N_IMAGENS)

    print("Montando modelo PC float...")
    model_pc_float = criar_modelo(simulate=False)

    print("Montando modelo PC simulate=True...")
    model_pc_sim = criar_modelo(simulate=True)

    try:
        ser = serial.Serial(PORTA_COM, BAUD_RATE, timeout=0.5)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Erro ao abrir a porta serial: {e}")
        sys.exit()

    acertos_pc_float = 0
    acertos_pc_sim = 0
    acertos_placa = 0
    concordancia_float_placa = 0
    concordancia_sim_placa = 0
    concordancia_float_sim = 0
    respostas_invalidas = 0

    print("\n--- COMPARAÇÃO PC FLOAT x PC SIMULATE x PLACA ---")

    for i in range(N_IMAGENS):
        img_chw = df_x.values[i].reshape(4, 28, 28)
        classe_real = int(np.argmax(df_y.values[i]))

        classe_pc_float, logits_pc_float = inferencia_pc_float(model_pc_float, img_chw)
        classe_pc_sim, logits_pc_sim = inferencia_pc_simulada(model_pc_sim, img_chw)

        bytes_para_enviar = preparar_bytes_placa(img_chw)

        ser.reset_input_buffer()
        ser.write(b"S")

        if not esperar_ack(ser):
            print(f"Img {i + 1}: timeout esperando ACK da placa.")
            respostas_invalidas += 1
            continue

        enviar_bytes(ser, bytes_para_enviar)
        resposta_bruta = ler_resposta(ser, timeout=2.0)
        classe_placa = extrair_classe_placa(resposta_bruta)

        if i < 10:
            print(f"\nIMG {i + 1}")
            print(f"  Real      : {classe_real}")
            print(f"  PC float  : {classe_pc_float}")
            print(f"  PC sim    : {classe_pc_sim}")
            print(f"  Placa     : {classe_placa}")
            print(f"  Logits PC float: {np.round(logits_pc_float, 3).tolist()}")
            print(f"  Logits PC sim  : {np.round(logits_pc_sim, 3).tolist()}")

            for linha in resposta_bruta.splitlines():
                linha = linha.strip()
                if linha.startswith("PROBE UART:"):
                    print(f"  {linha}")
                if linha.startswith("PROBE SRAM:"):
                    print(f"  {linha}")
                if linha.startswith("LOGITS"):
                    print(f"  {linha}")
                if linha.startswith("R:"):
                    print(f"  {linha}")

        if classe_pc_float == classe_real:
            acertos_pc_float += 1

        if classe_pc_sim == classe_real:
            acertos_pc_sim += 1

        if classe_pc_float == classe_pc_sim:
            concordancia_float_sim += 1

        if classe_placa is None:
            respostas_invalidas += 1
            continue

        if classe_placa == classe_real:
            acertos_placa += 1

        if classe_pc_float == classe_placa:
            concordancia_float_placa += 1

        if classe_pc_sim == classe_placa:
            concordancia_sim_placa += 1

    ser.close()

    print("\n" + "=" * 60)
    print(f"Total de imagens: {N_IMAGENS}")
    print(
        f"Acurácia PC float: {acertos_pc_float}/{N_IMAGENS} = "
        f"{(acertos_pc_float / N_IMAGENS) * 100:.2f}%"
    )
    print(
        f"Acurácia PC simulate=True: {acertos_pc_sim}/{N_IMAGENS} = "
        f"{(acertos_pc_sim / N_IMAGENS) * 100:.2f}%"
    )
    print(
        f"Acurácia placa: {acertos_placa}/{N_IMAGENS} = "
        f"{(acertos_placa / N_IMAGENS) * 100:.2f}%"
    )
    print(
        f"Concordância PC float x PC sim: {concordancia_float_sim}/{N_IMAGENS} = "
        f"{(concordancia_float_sim / N_IMAGENS) * 100:.2f}%"
    )
    print(
        f"Concordância PC float x placa: {concordancia_float_placa}/{N_IMAGENS} = "
        f"{(concordancia_float_placa / N_IMAGENS) * 100:.2f}%"
    )
    print(
        f"Concordância PC sim x placa: {concordancia_sim_placa}/{N_IMAGENS} = "
        f"{(concordancia_sim_placa / N_IMAGENS) * 100:.2f}%"
    )
    print(f"Respostas inválidas da placa: {respostas_invalidas}")
    print("=" * 60)


if __name__ == "__main__":
    main()