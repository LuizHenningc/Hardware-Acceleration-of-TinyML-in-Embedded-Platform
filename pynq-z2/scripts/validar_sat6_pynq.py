#!/usr/bin/env python3
"""Validação do acelerador FINN t2w8 na PYNQ-Z2 com amostras SAT-6."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = ROOT / "deploy" / "driver"
BITFILE = ROOT / "deploy" / "bitfile" / "finn-accel.bit"
DATASET = ROOT / "data" / "sat6_test_1000_uint8.npz"
DEFAULT_RESULTS = ROOT / "results" / "validacao-fisica" / "nova-execucao"

sys.path.insert(0, str(DRIVER_DIR))
os.chdir(str(DRIVER_DIR))

from driver import FINNExampleOverlay, io_shape_dict  # noqa: E402


CLASS_NAMES = [
    "barren_land",
    "trees",
    "grassland",
    "roads",
    "buildings",
    "water_bodies",
]


def make_overlay(batch_size):
    """Carrega o bitstream e inicializa o driver FINN."""
    return FINNExampleOverlay(
        bitfile_name=str(BITFILE),
        platform="zynq-iodma",
        io_shape_dict=io_shape_dict,
        batch_size=batch_size,
    )


def load_dataset(path, limit=None):
    """Carrega imagens UINT8 NHWC e rótulos inteiros."""
    with np.load(str(path)) as dataset:
        images = dataset["images"]
        labels = dataset["labels"]

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    if images.ndim != 4 or images.shape[1:] != (32, 32, 4):
        raise ValueError("Shape inesperado para images: {}".format(images.shape))

    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError(
            "Rótulos incompatíveis: images={}, labels={}".format(
                images.shape, labels.shape
            )
        )

    return (
        images.astype(np.uint8, copy=False),
        labels.astype(np.int64, copy=False),
    )


def build_confusion_matrix(labels, predictions):
    matrix = np.zeros((6, 6), dtype=np.int64)
    for real, predicted in zip(labels, predictions):
        matrix[int(real), int(predicted)] += 1
    return matrix


def save_results(output_dir, labels, predictions, matrix, elapsed_seconds):
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = int(np.sum(labels == predictions))
    total = int(len(labels))
    accuracy = correct / float(total)
    latency_ms = elapsed_seconds * 1000.0 / float(total)
    throughput = total / elapsed_seconds

    summary = {
        "configuration": "t2w8_500fps",
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed_seconds,
        "latency_ms_per_image": latency_ms,
        "throughput_images_per_second": throughput,
        "input_shape": list(io_shape_dict["ishape_normal"][0]),
        "input_datatype": str(io_shape_dict["idt"][0]),
        "output_shape": list(io_shape_dict["oshape_normal"][0]),
        "output_datatype": str(io_shape_dict["odt"][0]),
    }

    with (output_dir / "resumo_execucao.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    with (output_dir / "predicoes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["indice", "classe_real", "classe_predita", "correto"])
        for index, values in enumerate(zip(labels, predictions)):
            real, predicted = values
            writer.writerow(
                [index, int(real), int(predicted), int(real == predicted)]
            )

    np.savetxt(
        str(output_dir / "matriz_confusao.csv"),
        matrix,
        delimiter=",",
        fmt="%d",
    )

    with (output_dir / "resumo_por_classe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["classe", "nome", "amostras", "acertos", "acuracia"])
        for class_index, class_name in enumerate(CLASS_NAMES):
            samples = int(matrix[class_index].sum())
            hits = int(matrix[class_index, class_index])
            class_accuracy = hits / float(samples) if samples else 0.0
            writer.writerow(
                [class_index, class_name, samples, hits, class_accuracy]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Executa a validação física SAT-6 no acelerador FINN t2w8."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="Arquivo NPZ com arrays images e labels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de amostras para testes rápidos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Pasta de saída dos resultados.",
    )
    args = parser.parse_args()

    images, labels = load_dataset(args.dataset, args.limit)
    batch_size = len(images)

    print("Bitstream: {}".format(BITFILE))
    print("Dataset: {}".format(args.dataset))
    print("Amostras: {}".format(batch_size))
    print("Entrada: {}, dtype={}".format(images.shape, images.dtype))
    print("Carregando overlay FINN...")

    overlay = make_overlay(batch_size=batch_size)

    print("Executando inferência na PYNQ-Z2...")
    start = time.perf_counter()
    output = overlay.execute(images)
    elapsed_seconds = time.perf_counter() - start

    if isinstance(output, list):
        output = output[0]

    predictions = np.asarray(output).reshape(-1).astype(np.int64)
    if len(predictions) != len(labels):
        raise RuntimeError(
            "Saída incompatível: {}; esperado {}".format(
                predictions.shape, labels.shape
            )
        )

    matrix = build_confusion_matrix(labels, predictions)
    save_results(
        args.output_dir,
        labels,
        predictions,
        matrix,
        elapsed_seconds,
    )

    correct = int(np.sum(labels == predictions))
    accuracy = correct / float(len(labels))
    latency_ms = elapsed_seconds * 1000.0 / float(len(labels))
    throughput = len(labels) / elapsed_seconds

    print("")
    print("Resultado:")
    print("Acertos: {}/{}".format(correct, len(labels)))
    print("Acurácia: {:.2f}%".format(accuracy * 100.0))
    print("Tempo total: {:.6f} s".format(elapsed_seconds))
    print("Latência média: {:.6f} ms/imagem".format(latency_ms))
    print("Vazão: {:.6f} imagens/s".format(throughput))
    print("Arquivos salvos em: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
