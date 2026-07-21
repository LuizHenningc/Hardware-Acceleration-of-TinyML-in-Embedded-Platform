import re
import sys
import time

import numpy as np
import pandas as pd
import serial

PORTA_COM = "COM8"
BAUD_RATE = 115200
N_IMAGENS = 1000
MODO_ENTRADA = "original"
MOSTRAR_DEBUG_ATE = 0


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


def ler_resposta(ser: serial.Serial, timeout: float = 2.5) -> str:
    resposta_bruta = ""
    tempo_limite = time.time() + timeout

    while time.time() < tempo_limite:
        if ser.in_waiting > 0:
            resposta_bruta += ser.read(ser.in_waiting).decode("utf-8", errors="ignore")

            if re.search(r"R:\d", resposta_bruta):
                break

        time.sleep(0.01)

    return resposta_bruta


def extrair_classe(resposta_bruta: str) -> int | None:
    match = re.search(r"R:(\d)", resposta_bruta)

    if match:
        return int(match.group(1))

    return None


def preparar_imagem(img_chw: np.ndarray, modo_entrada: str) -> np.ndarray:
    img_hwc = np.transpose(img_chw, (1, 2, 0))

    if modo_entrada == "original":
        img_hwc_modo = img_hwc.copy()
    elif modo_entrada == "nir_zero":
        img_hwc_modo = img_hwc.copy()
        img_hwc_modo[:, :, 3] = 0
    elif modo_entrada == "bgrn":
        img_hwc_modo = img_hwc[:, :, [2, 1, 0, 3]].copy()
    elif modo_entrada == "nir_first":
        img_hwc_modo = img_hwc[:, :, [3, 0, 1, 2]].copy()
    else:
        raise ValueError(f"Modo de entrada inválido: {modo_entrada}")

    img_signed = np.clip(
        img_hwc_modo.astype(np.int16) - 128,
        -128,
        127,
    ).astype(np.int8)

    return img_signed


def main() -> None:
    print("Carregando dataset...")

    df_x = pd.read_csv("./dataset/X_test_sat6.csv", header=None, nrows=N_IMAGENS, dtype=np.uint8)
    df_y = pd.read_csv("./dataset/y_test_sat6.csv", header=None, nrows=N_IMAGENS)

    try:
        ser = serial.Serial(PORTA_COM, BAUD_RATE, timeout=0.5)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Erro ao abrir a porta: {e}")
        sys.exit()

    print("\n--- TESTE HIL FINAL ---")
    print(f"Modo atual: {MODO_ENTRADA}")
    print(f"Quantidade de imagens: {N_IMAGENS}")

    acertos = 0
    respostas_invalidas = 0

    for i in range(N_IMAGENS):
        img_chw = df_x.values[i].reshape(4, 28, 28)
        img_signed = preparar_imagem(img_chw, MODO_ENTRADA)
        bytes_para_enviar = img_signed.tobytes()

        if i < MOSTRAR_DEBUG_ATE:
            print("\n" + "-" * 40)
            print(f"IMG {i + 1}")
            print(f"PY BYTES: {list(bytes_para_enviar[:4])}")

        ser.reset_input_buffer()
        ser.write(b"S")

        if not esperar_ack(ser):
            print(f"IMG {i + 1}: sem ACK")
            respostas_invalidas += 1
            continue

        enviar_bytes(ser, bytes_para_enviar)
        resposta_bruta = ler_resposta(ser, timeout=2.5)

        classe_placa = extrair_classe(resposta_bruta)
        classe_real = int(np.argmax(df_y.values[i]))

        if i < MOSTRAR_DEBUG_ATE:
            print(resposta_bruta.strip())
            print(f"Real: {classe_real}")
            print(f"Placa: {classe_placa}")

        if classe_placa is None:
            respostas_invalidas += 1
            continue

        if classe_placa == classe_real:
            acertos += 1

    total_validas = N_IMAGENS - respostas_invalidas
    acuracia_total = (acertos / N_IMAGENS) * 100

    if total_validas > 0:
        acuracia_validas = (acertos / total_validas) * 100
    else:
        acuracia_validas = 0.0

    print("\n" + "=" * 40)
    print(f"Modo testado: {MODO_ENTRADA}")
    print(f"Acurácia total: {acuracia_total:.2f}%")
    print(f"Acurácia nas válidas: {acuracia_validas:.2f}%")
    print(f"Acertos: {acertos}/{N_IMAGENS}")
    print(f"Respostas inválidas: {respostas_invalidas}")
    print("=" * 40)

    ser.close()


if __name__ == "__main__":
    main()