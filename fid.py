"""Functionalities for calculating Frechet Inception Distance (FID)."""

import torch


def fid(embedding_real: torch.Tensor, embedding_fake: torch.Tensor):
    """Calculate Frechet Inception Distance

    FID = L2_norm(mean diff) + trace{Sigma_real + Sigma_fake - 2*sqrtm(Sigma_real @ Sigma_fake)}
    """
    mu_real = embedding_real.mean(dim=0)
    mu_fake = embedding_fake.mean(dim=0)
    diff_mu = mu_real - mu_fake

    Sigma_real = embedding_real.T.cov()
    Sigma_fake = embedding_fake.T.cov()
    prod_Sigma = Sigma_real @ Sigma_fake

    # TODO: need to find alternative ways to find sqrtm
    D, V = torch.linalg.eigh(prod_Sigma)
    sqrtm = (V * torch.sqrt(D)) @ V.T

    fid = diff_mu @ diff_mu + torch.trace(Sigma_real + Sigma_fake - 2 * sqrtm)
    return fid
