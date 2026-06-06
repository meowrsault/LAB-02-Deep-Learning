"""Carga de datos y generacion de datos demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMNS


def load_dataframe(data_path: str | Path) -> pd.DataFrame:
    """Carga un dataset desde CSV, SAV o Excel."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")

    if suffix == ".sav":
        try:
            import pyreadstat
        except ImportError as error:
            raise ImportError(
                "Para leer archivos .sav instala pyreadstat o usa environment.yml."
            ) from error

        dataframe, _ = pyreadstat.read_sav(path)
        return dataframe

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Formato no soportado. Usa .csv, .sav, .xlsx o .xls.")


def validate_required_columns(dataframe: pd.DataFrame, target_name: str) -> None:
    """Verifica que existan las columnas esperadas por el laboratorio."""

    required = FEATURE_COLUMNS + [target_name]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Faltan columnas necesarias en el dataset: " + ", ".join(missing)
        )


def create_demo_dataframe(n_samples: int = 360, seed: int = 42) -> pd.DataFrame:
    """
    Crea datos sinteticos con la misma forma que el laboratorio.

    Esto no reemplaza al dataset real. Solo permite probar el codigo, las
    metricas y el flujo de validacion antes de tener el archivo definitivo.
    """

    rng = np.random.default_rng(seed)
    probabilities = np.linspace(0.25, 0.75, len(FEATURE_COLUMNS))
    X = rng.binomial(1, probabilities, size=(n_samples, len(FEATURE_COLUMNS)))

    weights = np.array([1.4, 1.2, 1.1, 0.8, 1.0, 1.3, 1.6, 1.1, 0.7, 0.9,
                        1.5, 1.2, 1.4, 1.0, 1.3])
    score = X @ weights + rng.normal(0.0, 1.2, size=n_samples)
    bins = np.quantile(score, [0.12, 0.28, 0.43, 0.58, 0.73, 0.88])
    gds = np.digitize(score, bins=bins) + 1

    dataframe = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    dataframe.insert(0, ID_COLUMN, np.arange(1, n_samples + 1))
    dataframe["GDS"] = gds.astype(int)
    dataframe["GDS_R1"] = np.where(gds <= 2, 1, np.where(gds <= 4, 2, 3))
    dataframe["GDS_R2"] = np.where(gds <= 3, 1, np.where(gds <= 5, 2, 3))
    dataframe["GDS_R3"] = np.where(
        gds <= 2, 1, np.where(gds <= 4, 2, np.where(gds <= 6, 3, 4))
    )
    dataframe["GDS_R4"] = np.where(gds <= 3, 0, 1)
    dataframe["GDS_R5"] = np.where(gds <= 4, 1, 2)
    return dataframe


def describe_schema() -> str:
    """Devuelve un resumen corto de las columnas esperadas."""

    feature_text = ", ".join(FEATURE_COLUMNS)
    target_text = ", ".join(TARGET_COLUMNS)
    return (
        f"Entradas esperadas ({len(FEATURE_COLUMNS)}): {feature_text}\n"
        f"Targets disponibles: {target_text}\n"
        f"Columna ID opcional/no predictiva: {ID_COLUMN}"
    )


class CognitiveMultiLabelDataset(Dataset):
    """Dataset simple de PyTorch para entradas tabulares y targets one-hot."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]
