from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = ROOT_DIR / "resultados_max78000"

ARQUIVO_RESULTADOS = RESULTADOS_DIR / "resultados_hil_max78000.csv"
ARQUIVO_MATRIZ = RESULTADOS_DIR / "matriz_confusao_max78000.csv"

CLASSES = [f"Classe {i}" for i in range(6)]


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    resultados = pd.read_csv(ARQUIVO_RESULTADOS)
    matriz = pd.read_csv(ARQUIVO_MATRIZ, index_col=0)

    return resultados, matriz


def calcular_metricas(resultados: pd.DataFrame, matriz: pd.DataFrame) -> pd.DataFrame:
    metricas = []

    for indice_classe in range(len(CLASSES)):
        verdadeiros_positivos = matriz.iloc[indice_classe, indice_classe]
        total_real = matriz.iloc[indice_classe].sum()
        total_predito = matriz.iloc[:, indice_classe].sum()

        acuracia_classe = verdadeiros_positivos / total_real if total_real else 0
        precisao = verdadeiros_positivos / total_predito if total_predito else 0

        metricas.append({
            "classe": f"Classe {indice_classe}",
            "total_real": int(total_real),
            "acertos": int(verdadeiros_positivos),
            "erros": int(total_real - verdadeiros_positivos),
            "acuracia_classe": acuracia_classe * 100,
            "precisao": precisao * 100,
        })

    return pd.DataFrame(metricas)


def salvar_matriz_confusao(matriz: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    imagem = ax.imshow(matriz.values)

    ax.set_title("Matriz de confusão — MAX78000 HIL")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    for linha in range(len(CLASSES)):
        for coluna in range(len(CLASSES)):
            ax.text(
                coluna,
                linha,
                str(matriz.iloc[linha, coluna]),
                ha="center",
                va="center",
            )

    fig.colorbar(imagem, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "validacao_matriz_confusao_max78000.png", dpi=300)
    plt.close(fig)


def salvar_matriz_percentual(matriz: pd.DataFrame) -> None:
    matriz_percentual = matriz.div(matriz.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    imagem = ax.imshow(matriz_percentual.values)

    ax.set_title("Matriz de confusão percentual — MAX78000 HIL")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    for linha in range(len(CLASSES)):
        for coluna in range(len(CLASSES)):
            valor = matriz_percentual.iloc[linha, coluna]
            ax.text(
                coluna,
                linha,
                f"{valor:.1f}%",
                ha="center",
                va="center",
            )

    fig.colorbar(imagem, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "validacao_matriz_confusao_percentual_max78000.png", dpi=300)
    plt.close(fig)


def salvar_acuracia_por_classe(metricas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(metricas["classe"], metricas["acuracia_classe"])

    ax.set_title("Acurácia por classe — MAX78000 HIL")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)

    for indice, valor in enumerate(metricas["acuracia_classe"]):
        ax.text(indice, valor + 1, f"{valor:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "validacao_acuracia_por_classe_max78000.png", dpi=300)
    plt.close(fig)


def salvar_erros_por_classe(metricas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(metricas["classe"], metricas["erros"])

    ax.set_title("Erros por classe — MAX78000 HIL")
    ax.set_xlabel("Classe real")
    ax.set_ylabel("Quantidade de erros")

    for indice, valor in enumerate(metricas["erros"]):
        ax.text(indice, valor + 0.5, str(valor), ha="center")

    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "validacao_erros_por_classe_max78000.png", dpi=300)
    plt.close(fig)


def salvar_resumo(resultados: pd.DataFrame, metricas: pd.DataFrame) -> None:
    total = len(resultados)
    validas = resultados["resposta_valida"].sum()
    acertos = resultados["acertou"].sum()
    invalidas = total - validas

    acuracia_total = (acertos / total) * 100
    acuracia_validas = (acertos / validas) * 100 if validas else 0

    caminho_resumo = RESULTADOS_DIR / "validacao_resumo_max78000.txt"

    with caminho_resumo.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Validação HIL — MAX78000 / SAT-6\n")
        arquivo.write("=" * 40)
        arquivo.write("\n\n")
        arquivo.write(f"Total de imagens: {total}\n")
        arquivo.write(f"Respostas válidas: {validas}\n")
        arquivo.write(f"Respostas inválidas: {invalidas}\n")
        arquivo.write(f"Acertos: {acertos}/{total}\n")
        arquivo.write(f"Acurácia total: {acuracia_total:.2f}%\n")
        arquivo.write(f"Acurácia nas válidas: {acuracia_validas:.2f}%\n")
        arquivo.write("\nMétricas por classe:\n")
        arquivo.write(metricas.to_string(index=False))


def main() -> None:
    resultados, matriz = carregar_dados()
    metricas = calcular_metricas(resultados, matriz)

    metricas.to_csv(RESULTADOS_DIR / "validacao_metricas_por_classe_max78000.csv", index=False)

    salvar_matriz_confusao(matriz)
    salvar_matriz_percentual(matriz)
    salvar_acuracia_por_classe(metricas)
    salvar_erros_por_classe(metricas)
    salvar_resumo(resultados, metricas)

    print("Validação MAX78000 gerada com sucesso.")
    print(f"Pasta: {RESULTADOS_DIR}")
    print("\nArquivos principais:")
    print("validacao_matriz_confusao_max78000.png")
    print("validacao_matriz_confusao_percentual_max78000.png")
    print("validacao_acuracia_por_classe_max78000.png")
    print("validacao_erros_por_classe_max78000.png")
    print("validacao_metricas_por_classe_max78000.csv")
    print("validacao_resumo_max78000.txt")


if __name__ == "__main__":
    main()