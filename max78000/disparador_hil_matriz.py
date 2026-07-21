import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial

PORTA_COM = "COM8"
BAUD_RATE = 115200
N_IMAGENS = 1000

MODO_ENTRADA = "original"
MOSTRAR_DEBUG_ATE = 0
TOTAL_CLASSES = 6

ROOT_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = ROOT_DIR / "resultados_max78000"


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


def ler_startup(ser: serial.Serial, timeout: float = 2.0) -> str:
    resposta_bruta = ""
    tempo_limite = time.time() + timeout

    while time.time() < tempo_limite:
        if ser.in_waiting > 0:
            resposta_bruta += ser.read(ser.in_waiting).decode("utf-8", errors="ignore")

        time.sleep(0.01)

    return resposta_bruta


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


def extrair_valor_inteiro(resposta_bruta: str, chave: str) -> int | None:
    match = re.search(rf"{chave}:(\d+)", resposta_bruta)

    if match:
        return int(match.group(1))

    return None


def extrair_classe(resposta_bruta: str) -> int | None:
    match = re.search(r"R:(\d)", resposta_bruta)

    if match:
        return int(match.group(1))

    return None


def extrair_init_time(resposta_bruta: str) -> int | None:
    return extrair_valor_inteiro(resposta_bruta, "INIT_TIME")


def extrair_cnn_time(resposta_bruta: str) -> int | None:
    return extrair_valor_inteiro(resposta_bruta, "CNN_TIME")


def extrair_proc_time(resposta_bruta: str) -> int | None:
    return extrair_valor_inteiro(resposta_bruta, "PROC_TIME")


def extrair_fw_time(resposta_bruta: str) -> int | None:
    return extrair_valor_inteiro(resposta_bruta, "FW_TIME")


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

    return np.clip(
        img_hwc_modo.astype(np.int16) - 128,
        -128,
        127,
    ).astype(np.int8)


def gerar_matriz_confusao(reais: list[int], preditas: list[int]) -> np.ndarray:
    matriz = np.zeros((TOTAL_CLASSES, TOTAL_CLASSES), dtype=np.int32)

    for real, predita in zip(reais, preditas, strict=False):
        matriz[real, predita] += 1

    return matriz


def salvar_grafico_matriz(matriz: np.ndarray, caminho_saida: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    imagem = ax.imshow(matriz)

    ax.set_title("Matriz de confusão — MAX78000 HIL")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")

    classes = [f"Classe {i}" for i in range(TOTAL_CLASSES)]

    ax.set_xticks(np.arange(TOTAL_CLASSES))
    ax.set_yticks(np.arange(TOTAL_CLASSES))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(TOTAL_CLASSES):
        for j in range(TOTAL_CLASSES):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center")

    fig.colorbar(imagem, ax=ax)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=300)
    plt.close(fig)


def salvar_resumo_latencia_cnn(cnn_times: list[int]) -> dict[str, float]:
    if not cnn_times:
        resumo = {
            "cnn_time_medio_us": 0.0,
            "cnn_time_min_us": 0.0,
            "cnn_time_max_us": 0.0,
            "latencia_cnn_media_ms": 0.0,
            "latencia_cnn_min_ms": 0.0,
            "latencia_cnn_max_ms": 0.0,
            "throughput_puro_cnn": 0.0,
            "medicoes_cnn": 0,
        }
    else:
        cnn_times_np = np.array(cnn_times)

        cnn_time_medio_us = float(np.mean(cnn_times_np))
        cnn_time_min_us = float(np.min(cnn_times_np))
        cnn_time_max_us = float(np.max(cnn_times_np))

        latencia_cnn_media_ms = cnn_time_medio_us / 1000.0
        latencia_cnn_min_ms = cnn_time_min_us / 1000.0
        latencia_cnn_max_ms = cnn_time_max_us / 1000.0

        throughput_puro_cnn = (
            1000.0 / latencia_cnn_media_ms
            if latencia_cnn_media_ms > 0
            else 0.0
        )

        resumo = {
            "cnn_time_medio_us": cnn_time_medio_us,
            "cnn_time_min_us": cnn_time_min_us,
            "cnn_time_max_us": cnn_time_max_us,
            "latencia_cnn_media_ms": latencia_cnn_media_ms,
            "latencia_cnn_min_ms": latencia_cnn_min_ms,
            "latencia_cnn_max_ms": latencia_cnn_max_ms,
            "throughput_puro_cnn": throughput_puro_cnn,
            "medicoes_cnn": len(cnn_times),
        }

    caminho_txt = RESULTADOS_DIR / "latencia_pura_cnn_max78000.txt"

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Latência pura da CNN — MAX78000\n")
        arquivo.write("=" * 40)
        arquivo.write("\n\n")
        arquivo.write(f"Quantidade de medições válidas: {resumo['medicoes_cnn']}\n")
        arquivo.write(f"CNN_TIME médio: {resumo['cnn_time_medio_us']:.3f} us\n")
        arquivo.write(f"CNN_TIME mínimo: {resumo['cnn_time_min_us']:.3f} us\n")
        arquivo.write(f"CNN_TIME máximo: {resumo['cnn_time_max_us']:.3f} us\n")
        arquivo.write(f"Latência média CNN: {resumo['latencia_cnn_media_ms']:.3f} ms/imagem\n")
        arquivo.write(f"Latência mínima CNN: {resumo['latencia_cnn_min_ms']:.3f} ms\n")
        arquivo.write(f"Latência máxima CNN: {resumo['latencia_cnn_max_ms']:.3f} ms\n")
        arquivo.write(f"Throughput puro CNN: {resumo['throughput_puro_cnn']:.3f} imagens/s\n")
        arquivo.write("\nObservação: esta medição considera apenas o tempo interno da CNN, ")
        arquivo.write("medido pelo timer embarcado da MAX78000. Não inclui envio da imagem ")
        arquivo.write("por UART, leitura serial, processamento no computador, unload ou argmax.\n")

    pd.DataFrame([resumo]).to_csv(
        RESULTADOS_DIR / "latencia_pura_cnn_max78000.csv",
        index=False,
    )

    return resumo


def salvar_resumo_processamento(proc_times: list[int]) -> dict[str, float]:
    if not proc_times:
        resumo = {
            "proc_time_medio_us": 0.0,
            "proc_time_min_us": 0.0,
            "proc_time_max_us": 0.0,
            "latencia_processamento_media_ms": 0.0,
            "latencia_processamento_min_ms": 0.0,
            "latencia_processamento_max_ms": 0.0,
            "throughput_processamento": 0.0,
            "medicoes_processamento": 0,
        }
    else:
        proc_times_np = np.array(proc_times)

        proc_time_medio_us = float(np.mean(proc_times_np))
        proc_time_min_us = float(np.min(proc_times_np))
        proc_time_max_us = float(np.max(proc_times_np))

        latencia_processamento_media_ms = proc_time_medio_us / 1000.0
        latencia_processamento_min_ms = proc_time_min_us / 1000.0
        latencia_processamento_max_ms = proc_time_max_us / 1000.0

        throughput_processamento = (
            1000.0 / latencia_processamento_media_ms
            if latencia_processamento_media_ms > 0
            else 0.0
        )

        resumo = {
            "proc_time_medio_us": proc_time_medio_us,
            "proc_time_min_us": proc_time_min_us,
            "proc_time_max_us": proc_time_max_us,
            "latencia_processamento_media_ms": latencia_processamento_media_ms,
            "latencia_processamento_min_ms": latencia_processamento_min_ms,
            "latencia_processamento_max_ms": latencia_processamento_max_ms,
            "throughput_processamento": throughput_processamento,
            "medicoes_processamento": len(proc_times),
        }

    caminho_txt = RESULTADOS_DIR / "latencia_processamento_max78000.txt"

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Latência de processamento embarcado — MAX78000\n")
        arquivo.write("=" * 55)
        arquivo.write("\n\n")
        arquivo.write(f"Quantidade de medições PROC_TIME: {resumo['medicoes_processamento']}\n")
        arquivo.write(f"PROC_TIME médio: {resumo['proc_time_medio_us']:.3f} us\n")
        arquivo.write(f"PROC_TIME mínimo: {resumo['proc_time_min_us']:.3f} us\n")
        arquivo.write(f"PROC_TIME máximo: {resumo['proc_time_max_us']:.3f} us\n")
        arquivo.write(f"Latência média processamento: {resumo['latencia_processamento_media_ms']:.3f} ms/imagem\n")
        arquivo.write(f"Latência mínima processamento: {resumo['latencia_processamento_min_ms']:.3f} ms\n")
        arquivo.write(f"Latência máxima processamento: {resumo['latencia_processamento_max_ms']:.3f} ms\n")
        arquivo.write(f"Throughput processamento: {resumo['throughput_processamento']:.3f} imagens/s\n")
        arquivo.write("\nObservação: PROC_TIME considera o processamento embarcado após a imagem ")
        arquivo.write("já ter sido recebida pela UART. Inclui load_cnn_input, inferência CNN, ")
        arquivo.write("cnn_unload, conversão da saída e argmax. Não inclui a transmissão da imagem ")
        arquivo.write("pela UART nem etapas do Python.\n")

    pd.DataFrame([resumo]).to_csv(
        RESULTADOS_DIR / "latencia_processamento_max78000.csv",
        index=False,
    )

    return resumo


def salvar_resumo_firmware(
    fw_times: list[int],
    init_time: int | None,
) -> dict[str, float]:
    if not fw_times:
        resumo = {
            "init_time_us": float(init_time) if init_time is not None else 0.0,
            "init_time_ms": (float(init_time) / 1000.0) if init_time is not None else 0.0,
            "fw_time_medio_us": 0.0,
            "fw_time_min_us": 0.0,
            "fw_time_max_us": 0.0,
            "latencia_firmware_media_ms": 0.0,
            "latencia_firmware_min_ms": 0.0,
            "latencia_firmware_max_ms": 0.0,
            "throughput_firmware": 0.0,
            "medicoes_firmware": 0,
        }
    else:
        fw_times_np = np.array(fw_times)

        fw_time_medio_us = float(np.mean(fw_times_np))
        fw_time_min_us = float(np.min(fw_times_np))
        fw_time_max_us = float(np.max(fw_times_np))

        latencia_firmware_media_ms = fw_time_medio_us / 1000.0
        latencia_firmware_min_ms = fw_time_min_us / 1000.0
        latencia_firmware_max_ms = fw_time_max_us / 1000.0

        throughput_firmware = (
            1000.0 / latencia_firmware_media_ms
            if latencia_firmware_media_ms > 0
            else 0.0
        )

        resumo = {
            "init_time_us": float(init_time) if init_time is not None else 0.0,
            "init_time_ms": (float(init_time) / 1000.0) if init_time is not None else 0.0,
            "fw_time_medio_us": fw_time_medio_us,
            "fw_time_min_us": fw_time_min_us,
            "fw_time_max_us": fw_time_max_us,
            "latencia_firmware_media_ms": latencia_firmware_media_ms,
            "latencia_firmware_min_ms": latencia_firmware_min_ms,
            "latencia_firmware_max_ms": latencia_firmware_max_ms,
            "throughput_firmware": throughput_firmware,
            "medicoes_firmware": len(fw_times),
        }

    caminho_txt = RESULTADOS_DIR / "latencia_firmware_max78000.txt"

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Latência do firmware embarcado — MAX78000\n")
        arquivo.write("=" * 48)
        arquivo.write("\n\n")
        arquivo.write(f"INIT_TIME: {resumo['init_time_us']:.3f} us\n")
        arquivo.write(f"Tempo de inicialização: {resumo['init_time_ms']:.3f} ms\n")
        arquivo.write(f"Quantidade de medições FW_TIME: {resumo['medicoes_firmware']}\n")
        arquivo.write(f"FW_TIME médio: {resumo['fw_time_medio_us']:.3f} us\n")
        arquivo.write(f"FW_TIME mínimo: {resumo['fw_time_min_us']:.3f} us\n")
        arquivo.write(f"FW_TIME máximo: {resumo['fw_time_max_us']:.3f} us\n")
        arquivo.write(f"Latência média firmware: {resumo['latencia_firmware_media_ms']:.3f} ms/imagem\n")
        arquivo.write(f"Latência mínima firmware: {resumo['latencia_firmware_min_ms']:.3f} ms\n")
        arquivo.write(f"Latência máxima firmware: {resumo['latencia_firmware_max_ms']:.3f} ms\n")
        arquivo.write(f"Throughput firmware: {resumo['throughput_firmware']:.3f} imagens/s\n")
        arquivo.write("\nObservação: FW_TIME considera o tempo medido dentro da MAX78000 ")
        arquivo.write("após o recebimento do comando S. Inclui ACK, recepção UART da imagem, ")
        arquivo.write("cópia para a SRAM da CNN, inferência, unload, conversão da saída e argmax. ")
        arquivo.write("Não inclui a preparação da imagem no Python nem o envio final do resultado R:x.\n")

    pd.DataFrame([resumo]).to_csv(
        RESULTADOS_DIR / "latencia_firmware_max78000.csv",
        index=False,
    )

    return resumo


def salvar_resumo_sistema(
    tempos_sistema_ms: list[float],
    tempo_total_sistema_s: float,
    total_validas: int,
) -> dict[str, float]:
    if not tempos_sistema_ms:
        resumo = {
            "tempo_total_sistema_s": tempo_total_sistema_s,
            "latencia_sistema_media_ms": 0.0,
            "latencia_sistema_min_ms": 0.0,
            "latencia_sistema_max_ms": 0.0,
            "throughput_sistema_hil": 0.0,
            "medicoes_sistema": 0,
        }
    else:
        tempos_sistema_np = np.array(tempos_sistema_ms)

        latencia_sistema_media_ms = float(np.mean(tempos_sistema_np))
        latencia_sistema_min_ms = float(np.min(tempos_sistema_np))
        latencia_sistema_max_ms = float(np.max(tempos_sistema_np))

        throughput_sistema_hil = (
            total_validas / tempo_total_sistema_s
            if tempo_total_sistema_s > 0
            else 0.0
        )

        resumo = {
            "tempo_total_sistema_s": tempo_total_sistema_s,
            "latencia_sistema_media_ms": latencia_sistema_media_ms,
            "latencia_sistema_min_ms": latencia_sistema_min_ms,
            "latencia_sistema_max_ms": latencia_sistema_max_ms,
            "throughput_sistema_hil": throughput_sistema_hil,
            "medicoes_sistema": len(tempos_sistema_ms),
        }

    caminho_txt = RESULTADOS_DIR / "latencia_sistema_hil_max78000.txt"

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Latência do sistema HIL — MAX78000\n")
        arquivo.write("=" * 40)
        arquivo.write("\n\n")
        arquivo.write(f"Quantidade de medições: {resumo['medicoes_sistema']}\n")
        arquivo.write(f"Tempo total sistema HIL: {resumo['tempo_total_sistema_s']:.3f} s\n")
        arquivo.write(f"Latência média sistema HIL: {resumo['latencia_sistema_media_ms']:.3f} ms/imagem\n")
        arquivo.write(f"Latência mínima sistema HIL: {resumo['latencia_sistema_min_ms']:.3f} ms\n")
        arquivo.write(f"Latência máxima sistema HIL: {resumo['latencia_sistema_max_ms']:.3f} ms\n")
        arquivo.write(f"Throughput sistema HIL: {resumo['throughput_sistema_hil']:.3f} imagens/s\n")
        arquivo.write("\nObservação: esta medição considera o ciclo completo do teste HIL, ")
        arquivo.write("incluindo preparação no Python, envio do comando, espera do ACK, transmissão da imagem ")
        arquivo.write("por UART, inferência na MAX78000, resposta da placa e leitura serial no computador.\n")

    pd.DataFrame([resumo]).to_csv(
        RESULTADOS_DIR / "latencia_sistema_hil_max78000.csv",
        index=False,
    )

    return resumo


def main() -> None:
    RESULTADOS_DIR.mkdir(exist_ok=True)

    print("Carregando dataset...")

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

    try:
        ser = serial.Serial(PORTA_COM, BAUD_RATE, timeout=0.5)
        time.sleep(2)

        startup_msg = ler_startup(ser, timeout=1.0)
        init_time = extrair_init_time(startup_msg)

        ser.reset_input_buffer()
    except Exception as e:
        print(f"Erro ao abrir a porta: {e}")
        sys.exit()

    print("\n--- TESTE HIL COM MATRIZ DE CONFUSÃO ---")
    print(f"Modo atual: {MODO_ENTRADA}")
    print(f"Quantidade de imagens: {N_IMAGENS}")

    if startup_msg.strip():
        print("\nMensagens iniciais da placa:")
        print(startup_msg.strip())

    if init_time is not None:
        print(f"INIT_TIME capturado: {init_time} us ({init_time / 1000.0:.3f} ms)")
    else:
        print("INIT_TIME não capturado. Se precisar dele, reinicie a placa e rode o script novamente.")

    acertos = 0
    respostas_invalidas = 0

    classes_reais = []
    classes_placa = []
    linhas_resultado = []
    cnn_times = []
    proc_times = []
    fw_times = []
    tempos_sistema_ms = []

    inicio_total_sistema = time.perf_counter()

    for i in range(N_IMAGENS):
        inicio_imagem_sistema = time.perf_counter()

        img_chw = df_x.values[i].reshape(4, 28, 28)
        img_signed = preparar_imagem(img_chw, MODO_ENTRADA)
        bytes_para_enviar = img_signed.tobytes()
        classe_real = int(np.argmax(df_y.values[i]))

        ser.reset_input_buffer()
        ser.write(b"S")

        if not esperar_ack(ser):
            fim_imagem_sistema = time.perf_counter()
            tempo_imagem_sistema_ms = (fim_imagem_sistema - inicio_imagem_sistema) * 1000.0
            tempos_sistema_ms.append(tempo_imagem_sistema_ms)

            respostas_invalidas += 1
            linhas_resultado.append({
                "indice": i + 1,
                "classe_real": classe_real,
                "classe_placa": None,
                "acertou": False,
                "resposta_valida": False,
                "cnn_time_us": None,
                "latencia_cnn_ms": None,
                "proc_time_us": None,
                "latencia_processamento_ms": None,
                "fw_time_us": None,
                "latencia_firmware_ms": None,
                "latencia_sistema_ms": tempo_imagem_sistema_ms,
            })
            continue

        enviar_bytes(ser, bytes_para_enviar)
        resposta_bruta = ler_resposta(ser, timeout=2.5)

        fim_imagem_sistema = time.perf_counter()
        tempo_imagem_sistema_ms = (fim_imagem_sistema - inicio_imagem_sistema) * 1000.0
        tempos_sistema_ms.append(tempo_imagem_sistema_ms)

        classe_placa = extrair_classe(resposta_bruta)
        cnn_time = extrair_cnn_time(resposta_bruta)
        proc_time = extrair_proc_time(resposta_bruta)
        fw_time = extrair_fw_time(resposta_bruta)

        if cnn_time is not None:
            cnn_times.append(cnn_time)

        if proc_time is not None:
            proc_times.append(proc_time)

        if fw_time is not None:
            fw_times.append(fw_time)

        if i < MOSTRAR_DEBUG_ATE:
            print("\n" + "-" * 40)
            print(f"IMG {i + 1}")
            print(f"PY BYTES: {list(bytes_para_enviar[:4])}")
            print(resposta_bruta.strip())
            print(f"Real: {classe_real}")
            print(f"Placa: {classe_placa}")
            print(f"CNN_TIME: {cnn_time}")
            print(f"PROC_TIME: {proc_time}")
            print(f"FW_TIME: {fw_time}")
            print(f"Latência sistema HIL: {tempo_imagem_sistema_ms:.3f} ms")

        if classe_placa is None:
            respostas_invalidas += 1
            linhas_resultado.append({
                "indice": i + 1,
                "classe_real": classe_real,
                "classe_placa": None,
                "acertou": False,
                "resposta_valida": False,
                "cnn_time_us": cnn_time,
                "latencia_cnn_ms": cnn_time / 1000.0 if cnn_time is not None else None,
                "proc_time_us": proc_time,
                "latencia_processamento_ms": proc_time / 1000.0 if proc_time is not None else None,
                "fw_time_us": fw_time,
                "latencia_firmware_ms": fw_time / 1000.0 if fw_time is not None else None,
                "latencia_sistema_ms": tempo_imagem_sistema_ms,
            })
            continue

        acertou = classe_placa == classe_real

        if acertou:
            acertos += 1

        classes_reais.append(classe_real)
        classes_placa.append(classe_placa)

        linhas_resultado.append({
            "indice": i + 1,
            "classe_real": classe_real,
            "classe_placa": classe_placa,
            "acertou": acertou,
            "resposta_valida": True,
            "cnn_time_us": cnn_time,
            "latencia_cnn_ms": cnn_time / 1000.0 if cnn_time is not None else None,
            "proc_time_us": proc_time,
            "latencia_processamento_ms": proc_time / 1000.0 if proc_time is not None else None,
            "fw_time_us": fw_time,
            "latencia_firmware_ms": fw_time / 1000.0 if fw_time is not None else None,
            "latencia_sistema_ms": tempo_imagem_sistema_ms,
        })

    fim_total_sistema = time.perf_counter()
    tempo_total_sistema_s = fim_total_sistema - inicio_total_sistema

    ser.close()

    total_validas = N_IMAGENS - respostas_invalidas
    acuracia_total = (acertos / N_IMAGENS) * 100

    if total_validas > 0:
        acuracia_validas = (acertos / total_validas) * 100
    else:
        acuracia_validas = 0.0

    matriz = gerar_matriz_confusao(classes_reais, classes_placa)

    df_resultados = pd.DataFrame(linhas_resultado)
    df_matriz = pd.DataFrame(
        matriz,
        index=[f"real_{i}" for i in range(TOTAL_CLASSES)],
        columns=[f"pred_{i}" for i in range(TOTAL_CLASSES)],
    )

    caminho_resultados = RESULTADOS_DIR / "resultados_hil_max78000.csv"
    caminho_matriz_csv = RESULTADOS_DIR / "matriz_confusao_max78000.csv"
    caminho_matriz_png = RESULTADOS_DIR / "matriz_confusao_max78000.png"

    df_resultados.to_csv(caminho_resultados, index=False)
    df_matriz.to_csv(caminho_matriz_csv)

    resumo_latencia_cnn = salvar_resumo_latencia_cnn(cnn_times)
    resumo_processamento = salvar_resumo_processamento(proc_times)
    resumo_firmware = salvar_resumo_firmware(fw_times, init_time)
    resumo_sistema = salvar_resumo_sistema(
        tempos_sistema_ms,
        tempo_total_sistema_s,
        total_validas,
    )

    try:
        salvar_grafico_matriz(matriz, caminho_matriz_png)
        grafico_msg = str(caminho_matriz_png)
    except Exception as e:
        grafico_msg = f"Não foi possível gerar PNG: {e}"

    print("\n" + "=" * 40)
    print(f"Modo testado: {MODO_ENTRADA}")
    print(f"Acurácia total: {acuracia_total:.2f}%")
    print(f"Acurácia nas válidas: {acuracia_validas:.2f}%")
    print(f"Acertos: {acertos}/{N_IMAGENS}")
    print(f"Respostas inválidas: {respostas_invalidas}")

    if init_time is not None:
        print(f"Tempo de inicialização CNN/modelo: {init_time / 1000.0:.3f} ms")
    else:
        print("Tempo de inicialização CNN/modelo: não capturado")

    if cnn_times:
        print(f"Medições CNN_TIME capturadas: {len(cnn_times)}")
        print(f"Latência pura média CNN: {resumo_latencia_cnn['latencia_cnn_media_ms']:.3f} ms/imagem")
        print(f"Throughput puro CNN: {resumo_latencia_cnn['throughput_puro_cnn']:.3f} imagens/s")

        if resumo_latencia_cnn["cnn_time_medio_us"] <= 1.0:
            print("ATENÇÃO: CNN_TIME médio <= 1. O timer provavelmente não está ativo.")
    else:
        print("Nenhum CNN_TIME foi capturado.")
        print("Confira se o main.c está imprimindo CNN_TIME antes do R:x.")

    if proc_times:
        print(f"Medições PROC_TIME capturadas: {len(proc_times)}")
        print(f"Latência média processamento embarcado: {resumo_processamento['latencia_processamento_media_ms']:.3f} ms/imagem")
        print(f"Throughput processamento embarcado: {resumo_processamento['throughput_processamento']:.3f} imagens/s")
    else:
        print("Nenhum PROC_TIME foi capturado.")
        print("Confira se o main.c está imprimindo PROC_TIME antes do R:x.")

    if fw_times:
        print(f"Medições FW_TIME capturadas: {len(fw_times)}")
        print(f"Latência média firmware: {resumo_firmware['latencia_firmware_media_ms']:.3f} ms/imagem")
        print(f"Throughput firmware: {resumo_firmware['throughput_firmware']:.3f} imagens/s")
    else:
        print("Nenhum FW_TIME foi capturado.")
        print("Confira se o main.c está imprimindo FW_TIME antes do R:x.")

    print(f"Tempo total sistema HIL: {resumo_sistema['tempo_total_sistema_s']:.3f} s")
    print(f"Latência média sistema HIL: {resumo_sistema['latencia_sistema_media_ms']:.3f} ms/imagem")
    print(f"Latência mínima sistema HIL: {resumo_sistema['latencia_sistema_min_ms']:.3f} ms")
    print(f"Latência máxima sistema HIL: {resumo_sistema['latencia_sistema_max_ms']:.3f} ms")
    print(f"Throughput sistema HIL: {resumo_sistema['throughput_sistema_hil']:.3f} imagens/s")
    print("=" * 40)

    print("\nMatriz de confusão:")
    print(df_matriz)

    print("\nArquivos gerados:")
    print(caminho_resultados)
    print(caminho_matriz_csv)
    print(grafico_msg)
    print(RESULTADOS_DIR / "latencia_pura_cnn_max78000.csv")
    print(RESULTADOS_DIR / "latencia_pura_cnn_max78000.txt")
    print(RESULTADOS_DIR / "latencia_processamento_max78000.csv")
    print(RESULTADOS_DIR / "latencia_processamento_max78000.txt")
    print(RESULTADOS_DIR / "latencia_firmware_max78000.csv")
    print(RESULTADOS_DIR / "latencia_firmware_max78000.txt")
    print(RESULTADOS_DIR / "latencia_sistema_hil_max78000.csv")
    print(RESULTADOS_DIR / "latencia_sistema_hil_max78000.txt")


if __name__ == "__main__":
    main()