"""NN models."""

import torch
import torch.nn as nn
from tqdm import tqdm


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.G = nn.Sequential(
            nn.Linear(100, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """The forward function should return batch of images."""
        x = self.G(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """The forward function should return the logits."""
        p = self.D(x)
        return p


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
    return


class DCGAN(object):
    def __init__(self, epochs, lr, device):
        self.G = Generator().apply(weights_init).to(device)
        self.D = Discriminator().apply(weights_init).to(device)
        self.loss = nn.BCELoss()

        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.epochs = epochs
        self.lr = lr
        self.device = device

    def train(self, train_loader):
        for epoch in range(self.epochs):
            total_d_loss = 0
            total_g_loss = 0
            num_batches = 0
            verbose = (epoch % 5) == 0
            with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
                bar.set_description(f"Epoch {epoch}")
                for images, _ in bar:
                    batch_size = images.size(0)
                    # Step 1: Train discriminator
                    z = torch.rand((batch_size, 100))

                    real_labels = torch.ones(batch_size)
                    fake_labels = torch.zeros(batch_size)

                    images, z = images.to(self.device), z.to(self.device)
                    real_labels, fake_labels = real_labels.to(
                        self.device
                    ), fake_labels.to(self.device)

                    # Compute the BCE Loss using real images
                    real_scores = self.D(images)
                    real_scores = torch.squeeze(real_scores)
                    d_loss_real = self.loss(real_scores, real_labels)

                    # Compute the BCE Loss using fake images
                    fake_images = self.G(z)
                    fake_scores = self.D(fake_images)
                    fake_scores = torch.squeeze(fake_scores)
                    d_loss_fake = self.loss(fake_scores, fake_labels)

                    # Optimize discriminator
                    d_loss = d_loss_real + d_loss_fake
                    self.D.zero_grad()
                    d_loss.backward()
                    # max log(D(x)) + log(1 - D(G(z))) <=> min -[ log(D(x)) + log(1 - D(G(z))) ]
                    self.d_optimizer.step()

                    # Step 2: Train Generator
                    z = torch.randn(batch_size, 100).to(self.device)

                    fake_images = self.G(z)
                    fake_scores = self.D(fake_images)
                    fake_scores = torch.squeeze(fake_scores)
                    g_loss = self.loss(fake_scores, real_labels)

                    self.D.zero_grad()
                    self.G.zero_grad()
                    g_loss.backward()
                    # min log(1 - D(G(z))) => min -[ log(D(G(z))) ]
                    self.g_optimizer.step()

                    # update bar
                    num_batches += 1
                    total_d_loss += d_loss.item()
                    total_g_loss += g_loss.item()
                    bar.set_postfix(
                        d_loss=float(total_d_loss / num_batches),
                        g_loss=float(total_g_loss / num_batches),
                    )
            if total_d_loss / num_batches < 1e-3 or total_g_loss / num_batches > 8:
                print(
                    f"Discriminator loss is too small or generator loss is too high at epoch:{epoch}, which indicate a potential saturation problem."
                )
                print("Now, re-initialize and re-train...")
                self.__init__(self.epochs, self.lr)
                self.train(train_loader)
                return
        return

    def generate_img(self, number_of_images):
        # z dim = 100
        samples = (
            self.G(torch.rand((number_of_images, 100)).to(self.device))
            .detach()
            .cpu()
            .reshape(-1, 1, 28, 28)
        )
        samples_invtrans = samples * 0.5 + 0.5
        return samples_invtrans
