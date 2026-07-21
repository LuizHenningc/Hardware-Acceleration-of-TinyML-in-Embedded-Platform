import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "graficos_benchmark_sat6"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCHES = [1, 32, 256, 1024]
MODOS = ["7W", "15W"]

SAMPLES = 10000


def formatar_decimal(valor, casas=4):
    return f"{valor:.{casas}f}".replace(".", ",")


def extrair_metricas_modelo(caminho_log):
    texto = caminho_log.read_text(encoding="utf-8", errors="ignore")

    tempo_match = re.search(r"Execution time:\s*([\d.]+)\s*seconds", texto)
    latencia_match = re.search(r"Average latency:\s*([\d.]+)\s*ms/image", texto)
    acuracia_match = re.search(r"Accuracy:\s*([\d.]+)%", texto)

    if not tempo_match or not latencia_match or not acuracia_match:
        raise ValueError(f"Não consegui extrair métricas de {caminho_log}")

    return {
        "tempo_s": float(tempo_match.group(1)),
        "latencia_ms_img": float(latencia_match.group(1)),
        "acuracia_pct": float(acuracia_match.group(1)),
    }


def extrair_metricas_power(caminho_log):
    linhas = caminho_log.read_text(encoding="utf-8", errors="ignore").splitlines()

    vdd_todas = []
    vdd_gpu_ativa = []
    gr3d_todas = []
    gr3d_gpu_ativa = []

    for linha in linhas:
        vdd_match = re.search(r"VDD_IN\s+(\d+)mW/(\d+)mW", linha)
        gr3d_match = re.search(r"GR3D_FREQ\s+(\d+)%@", linha)

        if not vdd_match or not gr3d_match:
            continue

        vdd_atual_w = int(vdd_match.group(1)) / 1000
        gr3d_pct = int(gr3d_match.group(1))

        vdd_todas.append(vdd_atual_w)
        gr3d_todas.append(gr3d_pct)

        if gr3d_pct > 10:
            vdd_gpu_ativa.append(vdd_atual_w)
            gr3d_gpu_ativa.append(gr3d_pct)

    if not vdd_todas:
        raise ValueError(f"Não consegui extrair VDD_IN de {caminho_log}")

    if not vdd_gpu_ativa:
        vdd_gpu_ativa = vdd_todas

    if not gr3d_gpu_ativa:
        gr3d_gpu_ativa = gr3d_todas

    return {
        "vdd_in_medio_log_w": sum(vdd_todas) / len(vdd_todas),
        "vdd_in_medio_gpu_ativa_w": sum(vdd_gpu_ativa) / len(vdd_gpu_ativa),
        "linhas_power_log": len(vdd_todas),
        "gr3d_max_pct": max(gr3d_todas),
        "gr3d_medio_gpu_ativa_pct": sum(gr3d_gpu_ativa) / len(gr3d_gpu_ativa),
    }


def carregar_resultados():
    resultados = []

    for modo in MODOS:
        modo_lower = modo.lower()
        modo_dir = BASE_DIR / f"resultados_sat6_{modo_lower}"

        for batch in BATCHES:
            model_log = modo_dir / f"model_{modo_lower}_batch_{batch}.log"
            power_log = modo_dir / f"power_{modo_lower}_batch_{batch}.log"

            if not model_log.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {model_log}")

            if not power_log.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {power_log}")

            modelo = extrair_metricas_modelo(model_log)
            power = extrair_metricas_power(power_log)

            energia_j = power["vdd_in_medio_gpu_ativa_w"] * modelo["tempo_s"]
            energia_mj_img = (energia_j / SAMPLES) * 1000

            resultados.append(
                {
                    "modo": modo,
                    "batch": batch,
                    **modelo,
                    **power,
                    "energia_aprox_j": energia_j,
                    "energia_aprox_mj_img": energia_mj_img,
                }
            )

    return sorted(resultados, key=lambda item: (item["modo"], item["batch"]))


def salvar_csv(resultados):
    caminho_csv = OUTPUT_DIR / "resumo_resultados_sat6_jetson.csv"

    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

    print(f"CSV salvo em: {caminho_csv}")


def obter_dados_modo(resultados, modo):
    return [item for item in resultados if item["modo"] == modo]


def calcular_limites_y(resultados, campo, margem_superior=0.22, margem_inferior=0.12):
    valores = [item[campo] for item in resultados]
    menor = min(valores)
    maior = max(valores)
    intervalo = maior - menor

    if intervalo == 0:
        intervalo = maior if maior != 0 else 1

    limite_inferior = menor - intervalo * margem_inferior
    limite_superior = maior + intervalo * margem_superior

    if limite_inferior < 0 and menor >= 0:
        limite_inferior = 0

    return limite_inferior, limite_superior


def adicionar_rotulos(ax, x_posicoes, valores, casas, modo):
    y_min, y_max = ax.get_ylim()
    intervalo = y_max - y_min

    for posicao_x, valor_y in zip(x_posicoes, valores):
        deslocamento_y = 12
        alinhamento_vertical = "bottom"

        if valor_y > y_max - intervalo * 0.12:
            deslocamento_y = -18
            alinhamento_vertical = "top"

        if modo == "15W":
            deslocamento_y -= 4

        if modo == "7W":
            deslocamento_y += 4

        ax.annotate(
            formatar_decimal(valor_y, casas),
            (posicao_x, valor_y),
            textcoords="offset points",
            xytext=(0, deslocamento_y),
            ha="center",
            va=alinhamento_vertical,
            fontsize=11,
        )


def configurar_eixo_x(ax):
    x_posicoes = list(range(len(BATCHES)))
    labels = [str(batch) for batch in BATCHES]

    ax.set_xticks(x_posicoes)
    ax.set_xticklabels(labels)

    return x_posicoes


def gerar_grafico_linha(
    resultados,
    campo,
    titulo,
    eixo_y,
    nome_arquivo,
    casas=4,
):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    x_posicoes = configurar_eixo_x(ax)
    limite_inferior, limite_superior = calcular_limites_y(resultados, campo)

    ax.set_ylim(limite_inferior, limite_superior)

    for modo in MODOS:
        dados_modo = obter_dados_modo(resultados, modo)
        valores = [item[campo] for item in dados_modo]

        ax.plot(
            x_posicoes,
            valores,
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=modo,
        )

        adicionar_rotulos(ax, x_posicoes, valores, casas, modo)

    ax.set_xlabel("Tamanho do batch", fontsize=13)
    ax.set_ylabel(eixo_y, fontsize=13)
    ax.set_title(titulo, fontsize=16, pad=18)
    ax.legend(title="Modo", fontsize=12, title_fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=12)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    fig.tight_layout()

    caminho = OUTPUT_DIR / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico salvo em: {caminho}")


def gerar_grafico_unico(resultados):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))

    fig.suptitle(
        "Jetson Orin — Resultados SAT-6 (10.000 amostras)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    graficos = [
        {
            "campo": "latencia_ms_img",
            "titulo": "Latência média",
            "eixo_y": "ms/imagem",
            "casas": 4,
        },
        {
            "campo": "vdd_in_medio_gpu_ativa_w",
            "titulo": "Potência média",
            "eixo_y": "W",
            "casas": 2,
        },
        {
            "campo": "energia_aprox_j",
            "titulo": "Energia aproximada",
            "eixo_y": "J",
            "casas": 2,
        },
    ]

    for ax, grafico in zip(axes, graficos):
        x_posicoes = configurar_eixo_x(ax)

        limite_inferior, limite_superior = calcular_limites_y(
            resultados,
            grafico["campo"],
            margem_superior=0.25,
            margem_inferior=0.14,
        )

        ax.set_ylim(limite_inferior, limite_superior)

        for modo in MODOS:
            dados_modo = obter_dados_modo(resultados, modo)
            valores = [item[grafico["campo"]] for item in dados_modo]

            ax.plot(
                x_posicoes,
                valores,
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=modo,
            )

            adicionar_rotulos(ax, x_posicoes, valores, grafico["casas"], modo)

        ax.set_title(grafico["titulo"], fontsize=15, pad=16)
        ax.set_xlabel("Tamanho do batch", fontsize=12)
        ax.set_ylabel(grafico["eixo_y"], fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Modo", fontsize=10, title_fontsize=10)
        ax.tick_params(axis="both", labelsize=11)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    fig.text(
        0.5,
        0.02,
        "Fonte: medições do autor com TensorFlow C API e tegrastats.",
        ha="center",
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.92])

    caminho = OUTPUT_DIR / "grafico_unico_benchmark_sat6_jetson.png"
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico único salvo em: {caminho}")


def gerar_todos_graficos(resultados):
    gerar_grafico_linha(
        resultados,
        "latencia_ms_img",
        "SAT-6 na Jetson: latência média por batch",
        "Latência média (ms/imagem)",
        "grafico_latencia_sat6_jetson.png",
        casas=4,
    )

    gerar_grafico_linha(
        resultados,
        "vdd_in_medio_gpu_ativa_w",
        "SAT-6 na Jetson: potência média por batch",
        "Potência média (W)",
        "grafico_potencia_sat6_jetson.png",
        casas=2,
    )

    gerar_grafico_linha(
        resultados,
        "energia_aprox_j",
        "SAT-6 na Jetson: energia aproximada por batch",
        "Energia aproximada (J)",
        "grafico_energia_sat6_jetson.png",
        casas=2,
    )

    gerar_grafico_linha(
        resultados,
        "tempo_s",
        "SAT-6 na Jetson: tempo total por batch",
        "Tempo total (s)",
        "grafico_tempo_sat6_jetson.png",
        casas=4,
    )

    gerar_grafico_linha(
        resultados,
        "energia_aprox_mj_img",
        "SAT-6 na Jetson: energia aproximada por imagem",
        "Energia aproximada (mJ/imagem)",
        "grafico_energia_por_imagem_sat6_jetson.png",
        casas=4,
    )

    gerar_grafico_linha(
        resultados,
        "gr3d_medio_gpu_ativa_pct",
        "SAT-6 na Jetson: uso médio da GPU por batch",
        "Uso médio da GPU (%)",
        "grafico_gpu_sat6_jetson.png",
        casas=2,
    )

    gerar_grafico_unico(resultados)


def imprimir_resumo(resultados):
    print("\nResumo extraído dos logs:\n")

    for item in resultados:
        print(
            f"{item['modo']} | batch {item['batch']} | "
            f"latência {formatar_decimal(item['latencia_ms_img'], 4)} ms/img | "
            f"potência {formatar_decimal(item['vdd_in_medio_gpu_ativa_w'], 2)} W | "
            f"energia {formatar_decimal(item['energia_aprox_j'], 2)} J | "
            f"acurácia {formatar_decimal(item['acuracia_pct'], 2)}%"
        )


def main():
    resultados = carregar_resultados()

    salvar_csv(resultados)
    gerar_todos_graficos(resultados)
    imprimir_resumo(resultados)

    print("\nFinalizado.")
    print(f"Arquivos gerados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
