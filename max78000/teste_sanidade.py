import re
import sys
import time

import numpy as np
import serial

PORTA_COM = "COM8"
BAUD_RATE = 115200
REPETICOES = 3


def esperar_ack(ser: serial.Serial) -> bool:
    inicio_espera = time.time()

    while True:
        if ser.in_waiting > 0:
            if ser.read(1) == b'K':
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


def extrair_classe(resposta_bruta: str) -> int | None:
    match = re.search(r"R:(\d)", resposta_bruta)

    if match:
        return int(match.group(1))

    return None


def main() -> None:
    try:
        ser = serial.Serial(PORTA_COM, BAUD_RATE, timeout=0.5)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Erro ao abrir a porta: {e}")
        sys.exit()

    print("\n--- TESTE DE SANIDADE: INJEÇÃO DE RUÍDO ---")

    for tentativa in range(REPETICOES):
        print(f"\n========== SANIDADE {tentativa + 1} ==========")

        img_fake = np.random.randint(-128, 128, size=(28, 28, 4), dtype=np.int8)
        bytes_para_enviar = img_fake.tobytes()

        ser.reset_input_buffer()
        ser.write(b"S")

        if not esperar_ack(ser):
            print("Timeout esperando ACK 'K' da placa.")
            continue

        print("Enviando bytes aleatorios...")
        enviar_bytes(ser, bytes_para_enviar)

        resposta_bruta = ler_resposta(ser, timeout=2.0)

        for linha in resposta_bruta.splitlines():
            linha = linha.strip()

            if linha.startswith("PROBE UART:"):
                print(linha)

            if linha.startswith("PROBE SRAM:"):
                print(linha)

            if linha.startswith("LOGITS_Q15:"):
                print(linha)

            if linha.startswith("R:"):
                print(linha)

        classe_placa = extrair_classe(resposta_bruta)

        if classe_placa is None:
            print("Nao foi possivel interpretar a classe retornada pela placa.")
        else:
            print(f"Classe prevista pela placa: {classe_placa}")

    ser.close()


if __name__ == "__main__":
    main()