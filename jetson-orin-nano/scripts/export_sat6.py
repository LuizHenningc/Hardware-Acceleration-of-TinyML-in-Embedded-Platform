import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 4
FEATURE_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
NUM_CLASSES = 6


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path)

    if suffix == ".csv":
        return pd.read_csv(path, header=None).to_numpy()

    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, header=None).to_numpy()

    raise ValueError(f"Formato não suportado: {path}")


def prepare_x_test(x_test: np.ndarray, scale_255: bool) -> np.ndarray:
    x_test = np.asarray(x_test)

    if x_test.ndim == 2 and x_test.shape[1] == FEATURE_COUNT:
        x_test = x_test.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    elif x_test.ndim == 4 and x_test.shape[1:] == (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    ):
        pass

    elif x_test.ndim == 4 and x_test.shape[1:] == (
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):
        x_test = np.transpose(x_test, (0, 2, 3, 1))

    else:
        raise ValueError(
            "Formato inválido para X_test. "
            f"Esperado: (N, {FEATURE_COUNT}), "
            f"(N, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS}) "
            f"ou (N, {IMAGE_CHANNELS}, {IMAGE_HEIGHT}, {IMAGE_WIDTH}). "
            f"Recebido: {x_test.shape}"
        )

    x_test = x_test.astype(np.float32)

    if scale_255:
        x_test = x_test / 255.0

    return x_test


def prepare_y_test(y_test: np.ndarray) -> np.ndarray:
    y_test = np.asarray(y_test)

    if y_test.ndim > 1 and y_test.shape[-1] == NUM_CLASSES:
        y_test = np.argmax(y_test, axis=1)

    y_test = y_test.reshape(-1).astype(np.int64)

    if y_test.min() == 1 and y_test.max() == NUM_CLASSES:
        y_test = y_test - 1

    if y_test.min() < 0 or y_test.max() >= NUM_CLASSES:
        raise ValueError(
            "Labels inválidos. "
            f"Esperado intervalo 0 até {NUM_CLASSES - 1}. "
            f"Recebido mínimo={y_test.min()} máximo={y_test.max()}"
        )

    return y_test


def export_saved_model(model_path: Path, saved_model_dir: Path) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)

    input_spec = tf.TensorSpec(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
        dtype=tf.float32,
        name="input_sat6",
    )

    @tf.function(input_signature=[input_spec])
    def serving_fn(inputs):
        outputs = model(inputs, training=False)

        return {"outputs": outputs}

    saved_model_dir.mkdir(parents=True, exist_ok=True)

    tf.saved_model.save(
        model,
        str(saved_model_dir),
        signatures={"serving_default": serving_fn},
    )

    print(f"SavedModel exportado em: {saved_model_dir}")


def export_csv(
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    limit: int | None,
) -> None:
    if limit is not None:
        x_test = x_test[:limit]
        y_test = y_test[:limit]

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "Quantidade de amostras diferente entre X_test e y_test. "
            f"X_test={x_test.shape[0]} y_test={y_test.shape[0]}"
        )

    x_csv = x_test.reshape((x_test.shape[0], -1))

    if x_csv.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"CSV de entrada teria {x_csv.shape[1]} colunas, "
            f"mas o SAT-6 precisa de {FEATURE_COUNT}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / "X_test_sat6.csv"
    y_path = output_dir / "y_test_sat6.csv"

    pd.DataFrame(x_csv).to_csv(x_path, index=False, header=False)
    pd.DataFrame(y_test).to_csv(y_path, index=False, header=False)

    print(f"X_test exportado em: {x_path}")
    print(f"y_test exportado em: {y_path}")
    print(f"Amostras exportadas: {x_test.shape[0]}")
    print(f"Features por amostra: {x_csv.shape[1]}")
    print(f"Classes: {NUM_CLASSES}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Modelo Keras/TensorFlow para exportar como SavedModel. Ex: modelo.keras ou modelo.h5",
    )

    parser.add_argument(
        "--saved-model-dir",
        type=Path,
        default=Path("sat6_saved"),
        help="Pasta de saída do SavedModel.",
    )

    parser.add_argument(
        "--x-test",
        type=Path,
        required=True,
        help="Arquivo X_test em .npy, .csv, .xls ou .xlsx.",
    )

    parser.add_argument(
        "--y-test",
        type=Path,
        required=True,
        help="Arquivo y_test em .npy, .csv, .xls ou .xlsx.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Pasta onde serão salvos os CSVs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Quantidade de amostras para exportar. Use 0 para exportar tudo.",
    )

    parser.add_argument(
        "--scale-255",
        action="store_true",
        help="Divide X_test por 255. Use apenas se o modelo foi treinado com imagens normalizadas para 0-1.",
    )

    args = parser.parse_args()

    limit = args.limit

    if limit == 0:
        limit = None

    x_test = load_array(args.x_test)
    y_test = load_array(args.y_test)

    x_test = prepare_x_test(x_test, args.scale_255)
    y_test = prepare_y_test(y_test)

    export_csv(
        x_test=x_test,
        y_test=y_test,
        output_dir=args.output_dir,
        limit=limit,
    )

    if args.model is not None:
        export_saved_model(
            model_path=args.model,
            saved_model_dir=args.saved_model_dir,
        )


if __name__ == "__main__":
    main()
