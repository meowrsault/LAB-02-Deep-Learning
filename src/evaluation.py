"""Metricas, validacion anidada y comparacion de experimentos."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHT_DECAY,
    TARGET_COLUMNS,
)
from models import TrainConfig, resolve_device, train_model
from preprocessing import make_stratified_splits, prepare_experiment_bundle
from uncertainty import mc_dropout_predict


def apply_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convierte probabilidades en etiquetas binarias."""

    return (np.asarray(probabilities) >= threshold).astype(np.float32)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la proporcion de componentes mal clasificadas."""

    y_true, y_pred = _validate_shapes(y_true, y_pred)
    return float(np.not_equal(y_true, y_pred).mean())


def exact_match(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la proporcion de vectores predichos completamente correctos."""

    y_true, y_pred = _validate_shapes(y_true, y_pred)
    return float(np.all(y_true == y_pred, axis=1).mean())


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula las metricas multilabel del laboratorio."""

    y_true, y_pred = _validate_shapes(y_true, y_pred)
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)

    tp = np.logical_and(y_true_bool, y_pred_bool).sum(axis=0).astype(float)
    fp = np.logical_and(~y_true_bool, y_pred_bool).sum(axis=0).astype(float)
    fn = np.logical_and(y_true_bool, ~y_pred_bool).sum(axis=0).astype(float)

    precision_micro = _safe_divide(tp.sum(), tp.sum() + fp.sum())
    recall_micro = _safe_divide(tp.sum(), tp.sum() + fn.sum())
    f1_micro = _safe_divide(
        2 * precision_micro * recall_micro,
        precision_micro + recall_micro,
    )

    precision_per_label = _safe_divide_array(tp, tp + fp)
    recall_per_label = _safe_divide_array(tp, tp + fn)
    f1_per_label = _safe_divide_array(
        2 * precision_per_label * recall_per_label,
        precision_per_label + recall_per_label,
    )

    return {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match": exact_match(y_true, y_pred),
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "f1_macro": float(np.mean(f1_per_label)),
    }


def metrics_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Aplica el umbral y calcula las metricas multilabel."""

    predictions = apply_threshold(probabilities, threshold=threshold)
    return multilabel_metrics(y_true, predictions)


def _validate_shapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred deben tener la misma forma.")
    return y_true, y_pred


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _safe_divide_array(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def run_training_cycle(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    config: TrainConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Entrena una configuracion y calcula sus metricas."""

    result = train_model(
        X_train=X_train,
        Y_train=Y_train,
        X_eval=X_eval,
        config=config,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    probabilities = result["probabilities"]
    predictions = apply_threshold(probabilities, threshold=config.threshold)
    metrics = metrics_from_probabilities(
        Y_eval,
        probabilities,
        threshold=config.threshold,
    )
    return {
        **result,
        "predictions": predictions,
        "metrics": metrics,
    }


def build_param_grid(name: str) -> list[TrainConfig]:
    """Construye la grilla de hiperparametros."""

    if name == "none":
        return [
            TrainConfig(
                hidden_dim=DEFAULT_HIDDEN_DIM,
                dropout=DEFAULT_DROPOUT,
                learning_rate=DEFAULT_LEARNING_RATE,
                weight_decay=DEFAULT_WEIGHT_DECAY,
                batch_size=DEFAULT_BATCH_SIZE,
                threshold=DEFAULT_THRESHOLD,
                activation=DEFAULT_ACTIVATION,
            )
        ]

    if name == "quick":
        return [
            TrainConfig(32, 0.3, 1e-3, 0.0, 32, 0.5, "relu"),
            TrainConfig(16, 0.3, 1e-3, 0.0, 32, 0.5, "relu"),
            TrainConfig(64, 0.3, 1e-3, 0.0, 32, 0.5, "relu"),
            TrainConfig(32, 0.2, 1e-3, 0.0, 32, 0.5, "relu"),
            TrainConfig(32, 0.5, 1e-3, 0.0, 32, 0.5, "relu"),
            TrainConfig(32, 0.3, 1e-2, 0.0, 32, 0.5, "relu"),
            TrainConfig(32, 0.3, 1e-3, 1e-4, 32, 0.5, "relu"),
            TrainConfig(32, 0.3, 1e-3, 0.0, 16, 0.5, "relu"),
            TrainConfig(32, 0.3, 1e-3, 0.0, 32, 0.4, "relu"),
            TrainConfig(32, 0.3, 1e-3, 0.0, 32, 0.6, "relu"),
            TrainConfig(32, 0.3, 1e-3, 0.0, 32, 0.5, "tanh"),
            TrainConfig(32, 0.3, 1e-3, 0.0, 32, 0.5, "leaky_relu"),
        ]

    if name != "full":
        raise ValueError("grid debe ser none, quick o full.")

    values = {
        "hidden_dim": [16, 32, 64],
        "dropout": [0.2, 0.3, 0.5],
        "learning_rate": [1e-2, 1e-3],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [16, 32],
        "threshold": [0.4, 0.5, 0.6],
        "activation": ["relu", "tanh", "leaky_relu"],
    }
    keys = list(values)
    return [
        TrainConfig(**dict(zip(keys, combination)))
        for combination in product(*[values[key] for key in keys])
    ]


def run_targets(
    dataframe: pd.DataFrame,
    target_names: list[str],
    grid_name: str,
    selection_metric: str,
    outer_folds: int,
    inner_folds: int,
    epochs: int,
    seed: int,
    device_name: str,
    mc_samples: int,
    uncertainty_examples: int,
    output_dir: str | Path,
    max_configs: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Ejecuta los targets y devuelve las tablas de resultados."""

    device = resolve_device(device_name)
    param_grid = build_param_grid(grid_name)
    if max_configs is not None:
        param_grid = param_grid[:max_configs]

    outer_rows = []
    inner_rows = []
    uncertainty_rows = []
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for target_name in target_names:
        experiment_data = prepare_experiment_bundle(dataframe, target_name)
        try:
            result = run_one_target(
                X=experiment_data.X,
                Y=experiment_data.Y,
                target_name=target_name,
                classes=experiment_data.classes,
                param_grid=param_grid,
                selection_metric=selection_metric,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                epochs=epochs,
                seed=seed,
                device=device,
                mc_samples=mc_samples,
                uncertainty_examples=uncertainty_examples,
                model_dir=model_dir,
            )
        except ValueError as error:
            outer_rows.append(
                {
                    "target": target_name,
                    "status": "error",
                    "error_message": str(error),
                    "classes": "|".join(map(str, experiment_data.classes)),
                    "n_samples": len(experiment_data.Y),
                    "hamming_loss": np.nan,
                    "exact_match": np.nan,
                    "precision_micro": np.nan,
                    "recall_micro": np.nan,
                    "f1_micro": np.nan,
                    "f1_macro": np.nan,
                }
            )
            continue

        outer_rows.extend(result["outer_rows"])
        inner_rows.extend(result["inner_rows"])
        uncertainty_rows.extend(result["uncertainty_rows"])

    return {
        "outer_results": pd.DataFrame(outer_rows),
        "inner_results": pd.DataFrame(inner_rows),
        "uncertainty_examples": pd.DataFrame(uncertainty_rows),
    }


def run_one_target(
    X: np.ndarray,
    Y: np.ndarray,
    target_name: str,
    classes: list[int],
    param_grid: list[TrainConfig],
    selection_metric: str,
    outer_folds: int,
    inner_folds: int,
    epochs: int,
    seed: int,
    device: torch.device,
    mc_samples: int,
    uncertainty_examples: int,
    model_dir: Path,
) -> dict[str, list[dict]]:
    """Ejecuta validacion anidada para un target."""

    outer_splits = make_stratified_splits(Y, n_splits=outer_folds, seed=seed)
    outer_rows = []
    inner_rows = []
    uncertainty_rows = []

    for outer_fold, split in enumerate(outer_splits, start=1):
        outer_train_idx, outer_test_idx = split
        X_outer_train = X[outer_train_idx]
        Y_outer_train = Y[outer_train_idx]
        X_outer_test = X[outer_test_idx]
        Y_outer_test = Y[outer_test_idx]

        inner_splits = make_stratified_splits(
            Y_outer_train,
            n_splits=inner_folds,
            seed=seed + outer_fold,
        )
        best_config, fold_inner_rows = select_best_config(
            X_outer_train,
            Y_outer_train,
            target_name=target_name,
            outer_fold=outer_fold,
            param_grid=param_grid,
            inner_splits=inner_splits,
            selection_metric=selection_metric,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        inner_rows.extend(fold_inner_rows)

        final_result = run_training_cycle(
            X_train=X_outer_train,
            Y_train=Y_outer_train,
            X_eval=X_outer_test,
            Y_eval=Y_outer_test,
            config=best_config,
            epochs=epochs,
            seed=seed + outer_fold * 1000,
            device=device,
        )
        model_path = model_dir / f"{target_name}_outer_fold_{outer_fold}.pt"
        torch.save(final_result["model"].state_dict(), model_path)
        outer_rows.append(
            {
                "target": target_name,
                "status": "ok",
                "error_message": "",
                "outer_fold": outer_fold,
                "n_train": len(outer_train_idx),
                "n_test": len(outer_test_idx),
                "classes": "|".join(map(str, classes)),
                "model_path": str(model_path),
                "final_train_loss": final_result["final_train_loss"],
                **config_to_row(best_config),
                **final_result["metrics"],
            }
        )

        if mc_samples > 0 and uncertainty_examples > 0:
            uncertainty_rows.extend(
                build_uncertainty_rows(
                    model=final_result["model"],
                    X_examples=X_outer_test[:uncertainty_examples],
                    target_name=target_name,
                    outer_fold=outer_fold,
                    classes=classes,
                    mc_samples=mc_samples,
                    threshold=best_config.threshold,
                    device=device,
                )
            )

    return {
        "outer_rows": outer_rows,
        "inner_rows": inner_rows,
        "uncertainty_rows": uncertainty_rows,
    }


def select_best_config(
    X_train_outer: np.ndarray,
    Y_train_outer: np.ndarray,
    target_name: str,
    outer_fold: int,
    param_grid: list[TrainConfig],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    selection_metric: str,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[TrainConfig, list[dict]]:
    """Busca hiperparametros usando solamente los folds internos."""

    rows = []
    best_config = param_grid[0]
    best_score: float | None = None

    for config_id, config in enumerate(param_grid, start=1):
        metric_values = []
        for inner_fold, split in enumerate(inner_splits, start=1):
            inner_train_idx, inner_val_idx = split
            result = run_training_cycle(
                X_train=X_train_outer[inner_train_idx],
                Y_train=Y_train_outer[inner_train_idx],
                X_eval=X_train_outer[inner_val_idx],
                Y_eval=Y_train_outer[inner_val_idx],
                config=config,
                epochs=epochs,
                seed=seed + outer_fold * 100 + config_id * 10 + inner_fold,
                device=device,
            )
            metric_value = float(result["metrics"][selection_metric])
            metric_values.append(metric_value)
            rows.append(
                {
                    "target": target_name,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "config_id": config_id,
                    "selection_metric": selection_metric,
                    "selection_metric_value": metric_value,
                    "final_train_loss": result["final_train_loss"],
                    **config_to_row(config),
                    **result["metrics"],
                }
            )

        mean_score = float(np.mean(metric_values))
        if best_score is None or is_better(
            mean_score,
            best_score,
            selection_metric,
        ):
            best_score = mean_score
            best_config = config

    return best_config, rows


def build_uncertainty_rows(
    model: torch.nn.Module,
    X_examples: np.ndarray,
    target_name: str,
    outer_fold: int,
    classes: list[int],
    mc_samples: int,
    threshold: float,
    device: torch.device,
) -> list[dict]:
    """Genera probabilidades e incertidumbre por ejemplo y etiqueta."""

    if len(X_examples) == 0:
        return []

    inputs = torch.tensor(X_examples, dtype=torch.float32, device=device)
    mean_probs, std_probs = mc_dropout_predict(
        model,
        inputs,
        n_samples=mc_samples,
        device=device,
    )
    mean_np = mean_probs.cpu().numpy()
    std_np = std_probs.cpu().numpy()
    rows = []

    for sample_id in range(mean_np.shape[0]):
        for label_idx, original_class in enumerate(classes):
            rows.append(
                {
                    "target": target_name,
                    "outer_fold": outer_fold,
                    "example_id": sample_id,
                    "class": original_class,
                    "mean_probability": float(mean_np[sample_id, label_idx]),
                    "std_probability": float(std_np[sample_id, label_idx]),
                    "predicted_label": int(
                        mean_np[sample_id, label_idx] >= threshold
                    ),
                    "threshold": threshold,
                }
            )
    return rows


def config_to_row(config: TrainConfig) -> dict:
    """Convierte una configuracion a columnas tabulares."""

    return {
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "threshold": config.threshold,
        "activation": config.activation,
    }


def is_better(new_score: float, old_score: float, metric_name: str) -> bool:
    """Indica si una metrica nueva mejora el resultado anterior."""

    if metric_name == "hamming_loss":
        return new_score < old_score
    return new_score > old_score


def parse_targets(target_name: str, all_targets: bool) -> list[str]:
    """Devuelve la lista de targets que se deben ejecutar."""

    if all_targets:
        return TARGET_COLUMNS
    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"Target invalido: {target_name}. Usa {TARGET_COLUMNS}.")
    return [target_name]
