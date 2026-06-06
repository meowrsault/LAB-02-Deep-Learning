"""Preparacion de X, Y y folds estratificados."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMNS
from data_loader import validate_required_columns


@dataclass(frozen=True)
class ExperimentData:
    """Datos listos para un experimento supervisado."""

    X: np.ndarray
    Y: np.ndarray
    classes: list[int]
    class_to_idx: dict[int, int]
    cleaned_dataframe: pd.DataFrame


def encode_target_as_one_hot(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """Convierte una columna objetivo escalar a matriz one-hot."""

    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"Target invalido: {target_name}. Usa uno de {TARGET_COLUMNS}.")

    y_raw = dataframe[target_name].astype(int)
    classes = sorted(y_raw.unique().tolist())
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}

    y_idx = y_raw.map(class_to_idx).to_numpy()
    Y = np.zeros((len(y_idx), len(classes)), dtype=np.float32)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    return Y, classes, class_to_idx


def prepare_experiment_bundle(
    dataframe: pd.DataFrame,
    target_name: str,
    feature_columns: list[str] | None = None,
) -> ExperimentData:
    """Prepara matriz de entrada X y target one-hot Y para un target."""

    columns = feature_columns or FEATURE_COLUMNS
    validate_required_columns(dataframe, target_name)

    selected = dataframe[columns + [target_name]].copy()
    selected = selected.dropna(axis=0).reset_index(drop=True)
    if selected.empty:
        raise ValueError("No quedan filas validas despues de eliminar valores nulos.")

    X = selected[columns].astype("float32").to_numpy()
    Y, classes, class_to_idx = encode_target_as_one_hot(selected, target_name)
    return ExperimentData(X=X, Y=Y, classes=classes, class_to_idx=class_to_idx,
                          cleaned_dataframe=selected)


def prepare_experiment_data(
    dataframe: pd.DataFrame,
    target_name: str,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int]]:
    """
    API compatible con el main del profesor.

    Devuelve X, Y, classes y class_to_idx, tal como espera la plantilla base.
    """

    bundle = prepare_experiment_bundle(dataframe, target_name, feature_columns)
    return bundle.X, bundle.Y, bundle.classes, bundle.class_to_idx


def build_stratification_labels(Y: np.ndarray) -> np.ndarray:
    """Crea etiquetas auxiliares para estratificar los folds."""

    Y = np.asarray(Y, dtype=np.int64)
    if Y.ndim != 2:
        raise ValueError("Y debe ser una matriz 2D.")

    active_counts = Y.sum(axis=1)
    if np.any(active_counts <= 0):
        raise ValueError("Cada fila de Y debe activar al menos una etiqueta.")

    if np.all(active_counts == 1):
        return np.argmax(Y, axis=1)

    return np.asarray(["|".join(map(str, row.tolist())) for row in Y], dtype=object)


def make_stratified_splits(
    Y: np.ndarray,
    n_splits: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera folds estratificados sin depender de scikit-learn.

    Como en este laboratorio cada experimento parte de una sola clase escalar
    transformada a one-hot, la estratificacion usa la clase activa.
    """

    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")

    labels = build_stratification_labels(Y)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Se requieren al menos dos clases para validar.")

    min_count = int(counts.min())
    if min_count < n_splits:
        raise ValueError(
            f"No se pueden crear {n_splits} folds: la clase menos frecuente "
            f"solo tiene {min_count} muestras."
        )

    rng = np.random.default_rng(seed)
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        chunks = np.array_split(label_indices, n_splits)
        for fold_id, chunk in enumerate(chunks):
            fold_indices[fold_id].extend(chunk.tolist())

    all_indices = np.arange(len(labels))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_list in fold_indices:
        test_idx = np.asarray(sorted(test_list), dtype=np.int64)
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        splits.append((train_idx, test_idx))

    return splits


def split_for_validation(
    Y: np.ndarray,
    n_splits: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Alias compatible con el main del profesor."""

    return make_stratified_splits(Y, n_splits=n_splits, seed=random_state)
