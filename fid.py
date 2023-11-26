"""Functionalities for calculating Frechet Inception Distance (FID)."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CNN, CondGenerator, CondWGenerator, generate_synthetic_images
from torchvision.models.feature_extraction import create_feature_extractor


def calculate_embeddings(cnn_embedder, dataloader):
    """Calculate embeddings of images from a dataloader."""
    total_embedding = torch.zeros(len(dataloader.dataset), 2048)
    idx_curser = 0
    for img_batch, _ in dataloader:
        embedding = cnn_embedder(img_batch)["image_embedding"]
        embedding = embedding.reshape(-1, 2048).detach()
        total_embedding[idx_curser : (idx_curser + len(img_batch)), :] = embedding
        idx_curser += len(img_batch)
    return total_embedding


def fid_handler(
    mu_real: torch.Tensor, C_real: torch.Tensor, embedding_fake: torch.Tensor, m: int
):
    """Calculate Frechet Inception Distance.

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
    tr_sqrtm = torch.sum(
        torch.sqrt(torch.linalg.eigvals(C_fake @ C_real.T @ C_real @ C_fake.T))
    )
    # fid
    fid = d_mu @ d_mu + tr_S_r + tr_S_f - 2 * tr_sqrtm
    return fid


def fid(
    cnn: CNN,
    real_dataloader: DataLoader,
    generator: CondGenerator | CondWGenerator,
    B: int,
    m: int,
    target_weights: torch.Tensor,
    class_label: int,
    channel_dim: int,
    device: str,
):
    # build embedder
    return_node = {"avgpool": "image_embedding"}
    cnn_embedder = create_feature_extractor(cnn.resnet, return_node)
    # calculate E_r and its mu and C
    embedding_real = calculate_embeddings(cnn_embedder, real_dataloader)
    mu_real = embedding_real.mean(dim=0)
    C_real = (embedding_real - mu_real) / np.sqrt(len(real_dataloader.dataset))
    # iterate B times
    for i in range(B):
        # TODO: finish labels
        labels = torch.zeros(m)
        # generate fake image & calculate embeddings
        fake_images = generate_synthetic_images(generator, labels, channel_dim, device)
        embedding_fake = cnn_embedder(fake_images)["image_embedding"]
        embedding_fake = embedding_fake.reshape(-1, 2048).detach()
        fid = fid_handler(mu_real, C_real, embedding_fake)
        print(fid)
