"""Functionalities for calculating Frechet Inception Distance (FID)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from model import CNN, CondGenerator, CondWGenerator, generate_synthetic_images


def get_class_weights(dataloader: DataLoader, num_class: int):
    """Calculate class weights in a dataloader."""
    if isinstance(dataloader.dataset, ConcatDataset):
        class_counts = np.zeros(num_class)
        datasets = dataloader.dataset.datasets
        for i in range(len(datasets)):
            class_counts += np.bincount(datasets[i].labels.reshape(-1))
        return class_counts / len(dataloader.dataset)
    elif isinstance(dataloader.dataset, Dataset):
        class_counts = np.bincount(dataloader.dataset.labels.reshape(-1))
        return class_counts / len(dataloader.dataset)
    else:
        raise TypeError("DataLoader.dataset type error")


def fid_handler(
    mu_real: torch.Tensor, C_real: torch.Tensor, embedding_fake: torch.Tensor, m: int
):
    """Handle to calculate FID.

    Let C denotes the normalized re-centered embeddings.
    C = 1/sqrt(m - 1) * (E - mu @ 1)
    FID = L2_norm(mean diff) + trace{Sigma_real + Sigma_fake - 2*sqrtm(Sigma_real @ Sigma_fake)}
    """
    # get fake mu and C
    mu_fake = embedding_fake.mean(dim=0)
    C_fake = (embedding_fake - mu_fake) / np.sqrt(m - 1)
    # difference between mu
    d_mu = mu_real - mu_fake
    # trace of Sigma
    tr_S_r = (C_real**2).sum()
    tr_S_f = (C_fake**2).sum()

    # trace of sqrtm(S_r S_f)
    M = ((C_fake @ C_real.T) @ C_real) @ C_fake.T

    tr_sqrtm = torch.sum(torch.sqrt(torch.linalg.eigvals(M)))  # TODO: this is biased
    # fid
    fid = d_mu @ d_mu + tr_S_r + tr_S_f - 2 * tr_sqrtm
    return fid


def fid(
    cnn: CNN,
    real_dataloader: DataLoader,
    generator: CondGenerator | CondWGenerator,
    B: int,
    m: int,
    unconditional: bool,
    class_label: int | None,
    num_class: int,
    channel_dim: int,
    device: str,
):
    """Calculate FID."""
    # calculate E_r and its mu and C
    with torch.no_grad():
        embedding_real = cnn.calculate_embeddings(real_dataloader)
        mu_real = embedding_real.mean(dim=0)
        N = embedding_real.shape[0]
        C_real = (embedding_real - mu_real) / np.sqrt(N - 1)
        # iterate B times
        for i in range(B):
            if unconditional:
                target_weights = get_class_weights(real_dataloader, num_class)
                target_weights = torch.tensor(target_weights)
                labels = torch.multinomial(target_weights, m, replacement=True)
            else:
                labels = class_label * torch.ones(m)
            # generate fake image & calculate embeddings
            fake_images = generate_synthetic_images(
                generator, labels, channel_dim, device
            )
            _, embedding_fake = cnn(fake_images)
            embedding_fake = embedding_fake.reshape(-1, 2048).detach()
            fid = fid_handler(mu_real, C_real, embedding_fake, m)
            print("fid =", float(fid.real))
