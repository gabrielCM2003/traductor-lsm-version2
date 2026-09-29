"""Script de entrenamiento para señas estáticas (palabras de 1 o 2 manos).

Entrena una red neuronal MLP en PyTorch con la misma arquitectura base
que el clasificador de alfabeto de LSM:
    Linear(126 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> N)

Carga los datos desde los archivos CSV ubicados en `datos_palabras/` (generados
por `recolector_estatico.py`), entrena el modelo, guarda las etiquetas en
`word_labels.json` y exporta el modelo entrenado a formato ONNX (`lsm_words.onnx`).

Uso:
    python entrenar_palabras.py [--epochs 80] [--batch-size 32] [--lr 0.001]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("entrenar_palabras")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_palabras"
DEFAULT_OUTPUT_MODEL = Path(__file__).resolve().parent / "lsm_words.onnx"
DEFAULT_OUTPUT_LABELS = Path(__file__).resolve().parent / "word_labels.json"
N_FEATURES = 126


# =========================================================================== #
# Arquitectura del modelo (MLP 126 -> 128 -> 128 -> N)
# =========================================================================== #

class WordMLP(nn.Module):
    """Red perceptrón multicapa idéntica a la del alfabeto, adaptada a 126 features."""

    def __init__(self, in_features: int = 126, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================================== #
# Carga y preparación del dataset
# =========================================================================== #

def load_dataset_from_csv(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga todos los archivos CSV dentro de data_dir y extrae X (126 features) e y (etiquetas)."""
    if not data_dir.exists():
        raise FileNotFoundError(f"El directorio de datos no existe: {data_dir}")

    csv_files = sorted(list(data_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {data_dir}")

    samples_x: List[List[float]] = []
    samples_y: List[str] = []

    for csv_file in csv_files:
        log.info("Leyendo archivo: %s", csv_file.name)
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            if "etiqueta" not in fieldnames:
                log.warning(
                    "El archivo %s no tiene columna 'etiqueta'. Se omite.", csv_file.name
                )
                continue

            # Buscar las columnas de features (v0..v125 o todas las columnas excepto metadatos)
            feature_cols = [
                c for c in fieldnames if c not in ("etiqueta", "quien_grabo", "timestamp")
            ]

            if len(feature_cols) != N_FEATURES:
                # Intentar ordenar si tienen prefijo v
                v_cols = [f"v{i}" for i in range(N_FEATURES) if f"v{i}" in fieldnames]
                if len(v_cols) == N_FEATURES:
                    feature_cols = v_cols
                else:
                    log.warning(
                        "El archivo %s tiene %d columnas de features, se esperaban %d.",
                        csv_file.name,
                        len(feature_cols),
                        N_FEATURES,
                    )
                    continue

            for row_idx, row in enumerate(reader):
                label = row.get("etiqueta", "").strip()
                if not label:
                    continue
                try:
                    vec = [float(row[col]) for col in feature_cols]
                except (ValueError, KeyError) as e:
                    log.warning(
                        "Fila %d en %s contiene valores inválidos (%s). Se omite.",
                        row_idx + 1,
                        csv_file.name,
                        e,
                    )
                    continue

                samples_x.append(vec)
                samples_y.append(label)

    if not samples_x:
        raise ValueError(f"No se extrajo ninguna muestra válida desde {data_dir}")

    unique_labels = sorted(list(set(samples_y)))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y_indices = [label_to_idx[lbl] for lbl in samples_y]

    X = np.array(samples_x, dtype=np.float32)
    y = np.array(y_indices, dtype=np.int64)

    log.info(
        "Total de muestras cargadas: %d | Clases encontradas (%d): %s",
        len(X),
        len(unique_labels),
        unique_labels,
    )
    for lbl in unique_labels:
        count = sum(1 for item in samples_y if item == lbl)
        log.info("  - Clase '%s': %d muestras", lbl, count)

    return X, y, unique_labels


# =========================================================================== #
# Exportación a ONNX y Labels JSON
# =========================================================================== #

def save_labels_json(labels: List[str], output_path: Path) -> None:
    """Guarda las etiquetas en formato compatible con lsm_labels.json."""
    payload = {
        "labels": labels,
        "n_features": N_FEATURES,
        "normalization": "wrist_centered_middle_scaled",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("Etiquetas guardadas en: %s", output_path)


def export_to_onnx(model: nn.Module, output_path: Path) -> None:
    """Exporta el modelo entrenado a formato ONNX."""
    model.eval()
    dummy_input = torch.randn(1, N_FEATURES, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            input_names=["keypoints"],
            output_names=["logits"],
            dynamic_axes={
                "keypoints": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        # Si la versión de PyTorch no tiene el argumento dynamo
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            input_names=["keypoints"],
            output_names=["logits"],
            dynamic_axes={
                "keypoints": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
    log.info("Modelo ONNX exportado exitosamente a: %s", output_path)


def verify_onnx_model(onnx_path: Path, num_classes: int) -> None:
    """Verifica que el modelo ONNX sea válido y ejecutable con onnxruntime."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        test_in = np.random.randn(1, N_FEATURES).astype(np.float32)
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        res = session.run([out_name], {in_name: test_in})[0]

        assert res.shape == (1, num_classes), f"Shape inesperada: {res.shape}"
        log.info("Verificación ONNX satisfactoria: salida con shape %s", res.shape)
    except Exception as e:
        log.error("Fallo la verificación de ONNX: %s", e)
        raise


# =========================================================================== #
# Bucle de entrenamiento
# =========================================================================== #

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.2,
    seed: int = 42,
) -> nn.Module:
    """Entrena la red MLP en PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    total_samples = len(dataset)

    val_size = int(total_samples * val_split)
    # Garantizar que al menos haya entrenamiento y validación si hay suficientes muestras
    if val_size == 0 and total_samples >= 4:
        val_size = 1
    train_size = total_samples - val_size

    if val_size > 0:
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True
    )
    val_loader = (
        DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)
        if val_ds
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Dispositivo de entrenamiento: %s", device)

    model = WordMLP(in_features=N_FEATURES, hidden_dim=128, num_classes=num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_weights = None

    log.info(
        "Iniciando entrenamiento: %d épocas, %d muestras train, %d muestras val...",
        epochs,
        train_size,
        val_size,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validación
        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss_sum += loss.item() * len(batch_y)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += len(batch_y)

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_weights = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == epochs:
            if val_loader is not None:
                log.info(
                    "Época [%3d/%3d] - Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%%",
                    epoch,
                    epochs,
                    train_loss,
                    train_acc * 100,
                    val_loss,
                    val_acc * 100,
                )
            else:
                log.info(
                    "Época [%3d/%3d] - Train Loss: %.4f, Train Acc: %.2f%%",
                    epoch,
                    epochs,
                    train_loss,
                    train_acc * 100,
                )

    if best_weights is not None:
        model.load_state_dict(best_weights)
        log.info("Pesos restaurados del mejor checkpoint (Val Acc: %.2f%%)", best_val_acc * 100)

    model.to("cpu")
    return model


# =========================================================================== #
# Función principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de palabras estáticas LSM y exporta a ONNX."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directorio con los archivos CSV de datos (default: datos_palabras)",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_OUTPUT_MODEL,
        help="Ruta de destino del modelo ONNX (default: lsm_words.onnx)",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=DEFAULT_OUTPUT_LABELS,
        help="Ruta de destino de etiquetas JSON (default: word_labels.json)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Número de épocas de entrenamiento (default: 80)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del batch (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje (default: 0.001)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fracción de datos para validación (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    args = parser.parse_args()

    print("==========================================================")
    print("  Entrenamiento de Modelo de Palabras LSM (MLP 126->128->128->N)")
    print("==========================================================")

    try:
        X, y, labels = load_dataset_from_csv(args.data_dir)
    except Exception as e:
        log.error("Error al cargar dataset: %s", e)
        return 1

    if len(labels) < 2:
        log.error(
            "Se requieren al menos 2 clases distintas para entrenar. Encontradas: %d",
            len(labels),
        )
        return 1

    # Guardar etiquetas
    save_labels_json(labels, args.output_labels)

    # Entrenar
    model = train_model(
        X,
        y,
        num_classes=len(labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Exportar a ONNX
    export_to_onnx(model, args.output_model)

    # Verificar ONNX
    verify_onnx_model(args.output_model, len(labels))

    print("\n Entrenado y exportado exitosamente:")
    print(f"  - ONNX: {args.output_model}")
    print(f"  - Labels: {args.output_labels}")
    print(f"  - Total clases: {len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
