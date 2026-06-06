"""Monte Carlo Dropout para incertidumbre predictiva."""

from __future__ import annotations

import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """Deja activas las capas Dropout sin cambiar el resto del modelo."""

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    n_samples: int = 50,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ejecuta varias pasadas con dropout activo.

    Devuelve:
    - probabilidad media por etiqueta,
    - desviacion estandar por etiqueta.
    """

    if n_samples < 2:
        raise ValueError("n_samples debe ser al menos 2.")

    was_training = model.training
    model.eval()
    enable_dropout_during_inference(model)

    device = torch.device(device)
    inputs = inputs.to(device)
    samples = []
    for _ in range(n_samples):
        logits = model(inputs)
        samples.append(torch.sigmoid(logits).unsqueeze(0))

    stacked = torch.cat(samples, dim=0)
    mean_probs = stacked.mean(dim=0)
    std_probs = stacked.std(dim=0)

    if was_training:
        model.train()
    else:
        model.eval()

    return mean_probs, std_probs

