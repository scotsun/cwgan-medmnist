"""Utility functions for experiments."""

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import CDCGAN, CWDCGAN

LABEL_DICTS = {
    "organamnist": {
        0: "bladder",
        1: "femur-left",
        2: "femur-right",
        3: "heart",
        4: "kidney-left",
        5: "kidney-right",
        6: "liver",
        7: "lung-left",
        8: "lung-right",
        9: "pancreas",
        10: "spleen",
    },
    "pneumoniamnist": {0: "normal", 1: "pneumonia"},
    "bloodmnist": {
        0: "basophil",
        1: "eosinophil",
        2: "erythroblast",
        3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        4: "lymphocyte",
        5: "monocyte",
        6: "neutrophil",
        7: "platelet",
    },
}


def visualize_history(history: dict, measure1, measure2, logscale=False):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history[measure1])
    axs[0].plot(history["val_" + measure1])
    axs[0].set_xlabel("epoch")
    axs[0].set_xlabel(measure1)
    if logscale:
        axs[0].set_yscale("log")
    axs[1].plot(history[measure2], label="training")
    axs[1].plot(history["val_" + measure2], label="validation")
    axs[1].set_xlabel("epoch")
    axs[1].set_xlabel(measure2)
    fig.legend(labelcolor="linecolor")
    fig.tight_layout(pad=1.5)
    return fig


def compare_real_fake_by_class(
    gan: CDCGAN | CWDCGAN,
    real_dataloader,
    label_dict: dict,
    n_channels: int,
    n_samples=16,
    grid_nrow=8,
):
    """Plot fake vs real through stratification by class."""
    # fake at top; real at bottom
    n_classes = len(label_dict)
    img_batch, label_batch = next(iter(real_dataloader))
    img_batch = img_batch * 0.5 + 0.5
    for i in range(n_classes):
        print(f"{i}: {label_dict[i]}")
        samples = gan.generate_img(n_samples, i, n_channels)
        plt.imshow(
            torch.permute(make_grid(samples, nrow=grid_nrow, padding=0), (1, 2, 0)),
        )
        plt.axis("off")
        plt.show()

        idx = torch.where(label_batch == i, 1, 0).reshape(-1).nonzero().reshape(-1)
        real = img_batch[idx][:n_samples]
        plt.imshow(
            torch.permute(make_grid(real, nrow=grid_nrow, padding=0), (1, 2, 0)),
        )
        plt.axis("off")
        plt.show()
    return
