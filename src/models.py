"""Modelo neuronal y funciones de entrenamiento del laboratorio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import CognitiveMultiLabelDataset


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparametros de un entrenamiento."""

    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    threshold: float
    activation: str


def build_activation(name: str) -> nn.Module:
    """Construye la funcion de activacion solicitada."""

    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"leaky_relu", "leaky-relu", "leaky"}:
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError("Activacion no soportada. Usa relu, tanh o leaky_relu.")


class ShallowMultiLabelNet(nn.Module):
    """Red poco profunda con una capa oculta y salidas multilabel."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        output_dim: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    """Fija semillas para mejorar la reproducibilidad."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resuelve el dispositivo cpu, cuda o auto."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito cuda, pero no hay GPU disponible.")
    return device


def build_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Construye un DataLoader deterministico."""

    generator = torch.Generator().manual_seed(seed)
    dataset = CognitiveMultiLabelDataset(X, Y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Entrena una epoca completa."""

    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predice probabilidades sigmoid para una matriz de entrada."""

    dummy_y = np.zeros((len(X), 1), dtype=np.float32)
    loader = build_loader(
        X,
        dummy_y,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )

    model.eval()
    probabilities = []
    for inputs, _ in loader:
        logits = model(inputs.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probabilities)


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    config: TrainConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Entrena una configuracion y devuelve sus probabilidades."""

    if epochs < 1:
        raise ValueError("epochs debe ser al menos 1.")

    set_seed(seed)
    train_loader = build_loader(
        X_train,
        Y_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
    )
    model = ShallowMultiLabelNet(
        input_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        output_dim=Y_train.shape[1],
        activation=config.activation,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses = []
    for _ in range(epochs):
        losses.append(
            train_one_epoch(model, train_loader, optimizer, criterion, device)
        )

    probabilities = predict_probabilities(
        model,
        X_eval,
        batch_size=config.batch_size,
        device=device,
    )
    return {
        "model": model,
        "losses": losses,
        "final_train_loss": float(losses[-1]),
        "probabilities": probabilities,
    }
