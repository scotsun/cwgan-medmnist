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


def overall_comparison_by_class(
    data_flag,
    real_dataloader,
    label_dict,
    n_channels,
    model_vanila,
    model_wgan,
    n_samples=4,
    grid_nrow=16,
    device="cpu",
):
    img_batch, label_batch = next(iter(real_dataloader))
    img_batch = img_batch * 0.5 + 0.5
    n_classes = len(label_dict)
    root_path = "/content/model/"
    result_path = "/content/results/"
    plt.figure()
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    f, axarr = plt.subplots((n_classes + 1) // 2, 2, figsize=(16, 15))

    for i in range(n_classes):
        label = label_dict[i]
        idx = torch.where(label_batch == i, 1, 0).reshape(-1).nonzero().reshape(-1)
        real = img_batch[idx][:grid_nrow]
        all_image_in_one_class = torch.zeros(
            grid_nrow, real.shape[1], real.shape[2], real.shape[3]
        )
        all_image_in_one_class[: real.shape[0], :, :, :] = real

        for model in range(2):
            if model == 0:
                gan = model_vanila
                model_type = "vanilla"
            else:
                gan = model_wgan
                model_type = "wgan"
            for epoch in [10, 100]:
                for embedding in [0, 4, 8, 32]:
                    if embedding == 0:
                        model_path = (
                            root_path + f"{data_flag}_OneHot_epo{epoch}_{model_type}.pt"
                        )
                    else:
                        model_path = (
                            root_path
                            + f"{data_flag}_emb{embedding}_epo{epoch}_{model_type}.pt"
                        )
                    # print(model_path)
                    gan.G = torch.load(model_path, map_location=device)
                    samples = gan.generate_img(n_samples, i, n_channels)
                    all_image_in_one_class = torch.cat(
                        (all_image_in_one_class, samples), 0
                    )
        axarr[i // 2, i % 2].imshow(
            torch.permute(
                make_grid(all_image_in_one_class, nrow=grid_nrow, padding=1), (1, 2, 0)
            ),
        )
        axarr[i // 2, i % 2].axis("off")
        axarr[i // 2, i % 2].title.set_text(f"{i}: {label_dict[i]}")
    #         axarr[i//2, i%2].savefig(result_path + f'{data_flag}_class{i}.png')
    plt.axis("off")
    plt.savefig(result_path + f"{data_flag}_combine.png")
    plt.show()
    return
