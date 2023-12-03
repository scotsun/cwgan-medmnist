"""Functionalities for calculating Frechet Inception Distance (FID)."""

import numpy as np
from scipy.linalg import sqrtm
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
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


def extract_class_k(dataloader: DataLoader, class_k: int):
    """Extract all data of class from dataloader. Output another dataloader."""
    img_list, label_list = [], []
    for img_batch, label_batch in dataloader:
        argwhere = (label_batch == class_k).view(-1)
        img_list.append(img_batch[argwhere])
        label_list.append(label_batch[argwhere])
    class_k_dataloader = DataLoader(
        TensorDataset(torch.concat(img_list), torch.concat(label_list)),
        batch_size=dataloader.batch_size,
        shuffle=True,
    )
    return class_k_dataloader


def fid_base(
    cnn: CNN,
    real_dataloader: DataLoader,
    generator: CondGenerator | CondWGenerator,
    m: int,
    num_class: int,
    channel_dim: int,
    device: str,
    unconditional: bool = True,
    class_label: int | None = None,
    verbose: bool = False,
):
    """Calculate FID.

    param: m - sample size for fake images
    param: verbose - verbosity for calculate embeddings using real_dataloader
    """
    with torch.no_grad():
        if not unconditional:
            real_dataloader = extract_class_k(
                dataloader=real_dataloader, class_k=class_label
            )
        embedding_real = cnn.calculate_embeddings(real_dataloader, verbose)
        mu_real = embedding_real.mean(dim=0).cpu().numpy()
        sigma_real = embedding_real.T.cov().cpu().numpy()

        # generate fake images & calculate embedding
        if unconditional:
            target_weights = get_class_weights(real_dataloader, num_class)
            target_weights = torch.tensor(target_weights)
            labels = torch.multinomial(target_weights, m, replacement=True)
        else:
            labels = class_label * torch.ones(m)
        fake_images = generate_synthetic_images(generator, labels, channel_dim, device)
        _, embedding_fake = cnn(fake_images)
        embedding_fake = embedding_fake.reshape(-1, 2048).detach()

    # calculate mu, sigma for fake images
    mu_fake = embedding_fake.mean(dim=0).cpu().numpy()
    sigma_fake = embedding_fake.T.cov().cpu().numpy()

    # compute fid
    sigma_prod_sqrt = sqrtm(sigma_real @ sigma_fake).real
    fid_score = ((mu_real - mu_fake) ** 2).sum() + np.trace(
        sigma_real + sigma_fake - 2 * sigma_prod_sqrt
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    return fid_score


# -- to calculate FID alternative to using scipy.linalg.sqrtm


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
    num_class: int,
    channel_dim: int,
    device: str,
    unconditional: bool = True,
    class_label: int | None = None,
    verbose: bool = False,
):
    """Calculate FID."""
    # calculate E_r and its mu and C
    with torch.no_grad():
        if not unconditional:
            real_dataloader = extract_class_k(
                dataloader=real_dataloader, class_k=class_label
            )
        embedding_real = cnn.calculate_embeddings(real_dataloader, verbose)
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
    if device == "cuda":
        torch.cuda.empty_cache()
    return
