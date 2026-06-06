"""Laboratorio 02: redes neuronales poco profundas para deterioro cognitivo.

Este archivo ejecuta la version completa del laboratorio usando el dataset real.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import TARGET_COLUMNS  # noqa: E402
from data_loader import load_dataframe  # noqa: E402
from evaluation import run_targets  # noqa: E402


DATA_PATH_CANDIDATES = [
    BASE_DIR / "dataset" / "deterioro_cognitivo.sav",
]
OUTPUT_DIR = BASE_DIR / "outputs"

# Configuracion metodologica principal.
TARGETS = TARGET_COLUMNS
OUTER_FOLDS = 5
INNER_FOLDS = 3
EPOCHS = 5
GRID_NAME = "quick"
SELECTION_METRIC = "f1_macro"
SEED = 42
DEVICE = "cpu"

# Incertidumbre con Monte Carlo Dropout.
MC_SAMPLES = 50
UNCERTAINTY_EXAMPLES = 5


def resolve_data_path() -> Path:
    """Devuelve la primera ruta existente para el dataset real."""

    for data_path in DATA_PATH_CANDIDATES:
        if data_path.exists():
            return data_path

    expected = ", ".join(str(path) for path in DATA_PATH_CANDIDATES)
    raise FileNotFoundError(
        "No se encontro el dataset. Guarda deterioro_cognitivo.sav en una "
        f"de estas rutas: {expected}"
    )


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """Guarda tablas de resultados y devuelve el resumen por target."""

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    outer_results = tables["outer_results"]
    inner_results = tables["inner_results"]
    uncertainty_examples = tables["uncertainty_examples"]

    outer_results.to_csv(tables_dir / "outer_results.csv", index=False)
    inner_results.to_csv(tables_dir / "inner_results.csv", index=False)
    uncertainty_examples.to_csv(
        tables_dir / "uncertainty_examples.csv", index=False
    )

    summary = (
        outer_results.groupby("target", as_index=False)
        .agg(
            mean_hamming_loss=("hamming_loss", "mean"),
            std_hamming_loss=("hamming_loss", "std"),
            mean_f1_macro=("f1_macro", "mean"),
            mean_f1_micro=("f1_micro", "mean"),
            mean_exact_match=("exact_match", "mean"),
        )
        .sort_values("target")
    )
    summary.to_csv(tables_dir / "summary_by_target.csv", index=False)
    return summary


def save_figures(
    dataframe: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Guarda graficos principales para el informe."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(dataframe, figures_dir / "distribucion_clases.png")
    plot_metric_comparison(summary, figures_dir / "comparacion_metricas.png")
    plot_hamming_by_fold(
        tables["outer_results"],
        figures_dir / "hamming_loss_por_fold.png",
    )
    plot_inner_validation(
        tables["inner_results"],
        figures_dir / "validacion_interna_f1_macro.png",
    )
    plot_uncertainty(
        tables["uncertainty_examples"],
        figures_dir / "incertidumbre_mc_dropout.png",
    )


def plot_class_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Grafica la distribucion de clases de los seis targets."""

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, target in zip(axes.ravel(), TARGETS):
        counts = dataframe[target].value_counts().sort_index()
        axis.bar([str(int(value)) for value in counts.index], counts.values)
        axis.set_title(target)
        axis.set_xlabel("Clase")
        axis.set_ylabel("Frecuencia")
    fig.suptitle("Distribucion de clases por target", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Compara metricas promedio entre experimentos."""

    plot_data = summary.dropna(subset=["mean_f1_macro"]).copy()
    if plot_data.empty:
        return

    metrics = [
        "mean_hamming_loss",
        "mean_f1_macro",
        "mean_f1_micro",
        "mean_exact_match",
    ]
    labels = ["Hamming Loss", "F1 Macro", "F1 Micro", "Exact Match"]

    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x_positions = range(len(plot_data))
    width = 0.2
    for offset, (metric, label) in enumerate(zip(metrics, labels)):
        values = plot_data[metric].fillna(0.0).to_numpy()
        shifted = [x + (offset - 1.5) * width for x in x_positions]
        axis.bar(shifted, values, width=width, label=label)

    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(plot_data["target"].tolist())
    axis.set_ylim(0, 1)
    axis.set_ylabel("Valor promedio")
    axis.set_title("Comparacion de metricas por experimento")
    axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_hamming_by_fold(outer_results: pd.DataFrame, output_path: Path) -> None:
    """Grafica Hamming Loss externo por fold."""

    plot_data = outer_results.dropna(subset=["hamming_loss"]).copy()
    if plot_data.empty:
        return

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for target, group in plot_data.groupby("target"):
        axis.plot(
            group["outer_fold"],
            group["hamming_loss"],
            marker="o",
            linewidth=1.8,
            label=target,
        )

    axis.set_xlabel("Fold externo")
    axis.set_ylabel("Hamming Loss")
    axis.set_title("Hamming Loss externo por fold")
    axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_inner_validation(inner_results: pd.DataFrame, output_path: Path) -> None:
    """Grafica F1 Macro en validacion interna."""

    plot_data = inner_results.dropna(subset=["f1_macro"]).copy()
    if plot_data.empty:
        return

    grouped = (
        plot_data.groupby("target", as_index=False)["f1_macro"]
        .mean()
        .sort_values("f1_macro", ascending=False)
    )

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(grouped["target"], grouped["f1_macro"])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Target")
    axis.set_ylabel("F1 Macro promedio interno")
    axis.set_title("Validacion interna: F1 Macro promedio")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_uncertainty(uncertainty: pd.DataFrame, output_path: Path) -> None:
    """Grafica probabilidad media versus incertidumbre."""

    required = {"mean_probability", "std_probability", "target"}
    if uncertainty.empty or not required.issubset(uncertainty.columns):
        return

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for target, group in uncertainty.groupby("target"):
        axis.scatter(
            group["mean_probability"],
            group["std_probability"],
            alpha=0.75,
            label=target,
        )

    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Probabilidad media")
    axis.set_ylabel("Desviacion estandar")
    axis.set_title("Incertidumbre predictiva con Monte Carlo Dropout")
    axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_class_distribution(dataframe: pd.DataFrame) -> None:
    """Muestra la distribucion de clases de cada target."""

    print("Distribucion de clases por target:")
    for target in TARGETS:
        counts = dataframe[target].value_counts().sort_index()
        formatted = ", ".join(
            f"{int(class_value)}={int(count)}"
            for class_value, count in counts.items()
        )
        print(f"- {target}: {formatted}")


def main() -> None:
    """Ejecuta el laboratorio completo."""

    data_path = resolve_data_path()

    dataframe = load_dataframe(data_path)
    print("Dataset cargado correctamente.")
    print(f"Ruta: {data_path}")
    print(f"Forma del dataset: {dataframe.shape}")
    print_class_distribution(dataframe)
    print()

    tables = run_targets(
        dataframe=dataframe,
        target_names=TARGETS,
        grid_name=GRID_NAME,
        selection_metric=SELECTION_METRIC,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        epochs=EPOCHS,
        seed=SEED,
        device_name=DEVICE,
        mc_samples=MC_SAMPLES,
        uncertainty_examples=UNCERTAINTY_EXAMPLES,
        output_dir=OUTPUT_DIR,
    )

    summary = save_tables(tables, OUTPUT_DIR)
    save_figures(dataframe, tables, summary, OUTPUT_DIR)

    print("Laboratorio completo ejecutado correctamente.")
    print("Tablas guardadas en outputs/tables/:")
    print("- outer_results.csv")
    print("- inner_results.csv")
    print("- summary_by_target.csv")
    print("- uncertainty_examples.csv")
    print("Graficos guardados en outputs/figures/:")
    print("- distribucion_clases.png")
    print("- comparacion_metricas.png")
    print("- hamming_loss_por_fold.png")
    print("- validacion_interna_f1_macro.png")
    print("- incertidumbre_mc_dropout.png")
    print()
    print("Resumen por target:")
    print(summary.to_string(index=False))

    print()
    print(
        "Nota: GDS puede aparecer sin metricas si sus clases minoritarias no "
        "alcanzan para la validacion estratificada anidada. Esa es una "
        "limitacion metodologica importante del dataset."
    )


if __name__ == "__main__":
    main()
