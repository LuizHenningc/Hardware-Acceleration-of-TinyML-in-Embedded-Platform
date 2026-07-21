import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent

LOG_PATH = (
    ROOT_DIR
    / "ai8x-training"
    / "logs"
    / "2026.05.25-151645"
    / "2026.05.25-151645.log"
)

RESULTADOS_DIR = ROOT_DIR / "resultados_max78000"


def extrair_metricas_log(log_path: Path) -> pd.DataFrame:
    linhas = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    registros: dict[int, dict[str, float]] = {}

    secao_atual = None
    epoch_validacao = None

    regex_treino = re.compile(
        r"Epoch:\s*\[(\d+)\]\[\s*\d+/\s*\d+\].*?"
        r"(?:Overall Loss|Loss)\s+([0-9.]+).*?"
        r"Top1\s+([0-9.]+)"
    )

    regex_validate_inicio = re.compile(r"--- validate \(epoch=(\d+)\)")
    regex_resultado = re.compile(r"==>\s*Top1:\s*([0-9.]+).*?Loss:\s*([0-9.]+)")

    for linha in linhas:
        if "Training epoch:" in linha:
            secao_atual = "treino"
            continue

        match_validate_inicio = regex_validate_inicio.search(linha)

        if match_validate_inicio:
            secao_atual = "validacao"
            epoch_validacao = int(match_validate_inicio.group(1))
            registros.setdefault(epoch_validacao, {"epoch": epoch_validacao})
            continue

        if secao_atual == "treino":
            match_treino = regex_treino.search(linha)

            if match_treino:
                epoch = int(match_treino.group(1))
                loss = float(match_treino.group(2))
                top1 = float(match_treino.group(3))

                registros.setdefault(epoch, {"epoch": epoch})
                registros[epoch]["treino_loss"] = loss
                registros[epoch]["treino_top1"] = top1

            continue

        if secao_atual == "validacao" and epoch_validacao is not None:
            match_resultado = regex_resultado.search(linha)

            if match_resultado:
                top1 = float(match_resultado.group(1))
                loss = float(match_resultado.group(2))

                registros.setdefault(epoch_validacao, {"epoch": epoch_validacao})
                registros[epoch_validacao]["validacao_top1"] = top1
                registros[epoch_validacao]["validacao_loss"] = loss

                secao_atual = None
                epoch_validacao = None

    df = pd.DataFrame(registros.values()).sort_values("epoch")

    return df


def salvar_grafico_acuracia(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    if "treino_top1" in df:
        ax.plot(df["epoch"], df["treino_top1"], marker="o", label="Treino")

    if "validacao_top1" in df:
        ax.plot(df["epoch"], df["validacao_top1"], marker="o", label="Validação")

    ax.set_title("Acurácia de Treino e Validação — MAX78000")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "treino_validacao_acuracia_max78000.png", dpi=300)
    plt.close(fig)


def salvar_grafico_loss(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    if "treino_loss" in df:
        ax.plot(df["epoch"], df["treino_loss"], marker="o", label="Treino")

    if "validacao_loss" in df:
        ax.plot(df["epoch"], df["validacao_loss"], marker="o", label="Validação")

    ax.set_title("Loss de Treino e Validação — MAX78000")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "treino_validacao_loss_max78000.png", dpi=300)
    plt.close(fig)


def salvar_resumo(df: pd.DataFrame) -> None:
    caminho_csv = RESULTADOS_DIR / "treino_validacao_metricas_max78000.csv"
    caminho_txt = RESULTADOS_DIR / "treino_validacao_resumo_max78000.txt"

    df.to_csv(caminho_csv, index=False)

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Treinamento do modelo utilizado na MAX78000\n")
        arquivo.write("=" * 48)
        arquivo.write("\n\n")
        arquivo.write(f"Log utilizado: {LOG_PATH}\n")
        arquivo.write(f"Quantidade de épocas encontradas: {len(df)}\n\n")

        if len(df) <= 1:
            arquivo.write(
                "Observação: o log contém apenas uma época registrada. "
                "Portanto, os gráficos representam o ponto disponível do treinamento "
                "do modelo usado na MAX78000, e não uma curva longa de evolução por épocas.\n\n"
            )

        arquivo.write(df.to_string(index=False))


def main() -> None:
    RESULTADOS_DIR.mkdir(exist_ok=True)

    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log não encontrado: {LOG_PATH}")

    df = extrair_metricas_log(LOG_PATH)

    if df.empty:
        raise RuntimeError("Nenhuma métrica de treino/validação foi encontrada no log.")

    salvar_grafico_acuracia(df)
    salvar_grafico_loss(df)
    salvar_resumo(df)

    print("Gráficos de treino/validação da MAX78000 gerados com sucesso.")
    print(f"Pasta: {RESULTADOS_DIR}")
    print("\nArquivos gerados:")
    print("treino_validacao_acuracia_max78000.png")
    print("treino_validacao_loss_max78000.png")
    print("treino_validacao_metricas_max78000.csv")
    print("treino_validacao_resumo_max78000.txt")


if __name__ == "__main__":
    main()