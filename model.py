"""NN models."""

import torch
import torch.nn as nn
from tqdm import tqdm


def weights_init(m):
    """Initialize parameters/weights in GAN."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return


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
                self.__init__(self.epochs, self.lr, self.device)
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


class CondGenerator(torch.nn.Module):
    def __init__(self, num_class, embedding_dim, latent_dim):
        super().__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.label_embedding = nn.Embedding(num_class, embedding_dim)
        self.label_channel = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=embedding_dim,
                out_channels=128,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.noise_channel = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=128,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
        )
        self.G = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, z, y):
        """The forward function should return batch of images."""
        z = z.reshape(-1, self.latent_dim, 1, 1)
        y = self.label_channel(
            self.label_embedding(y).reshape(-1, self.embedding_dim, 1, 1)
        )
        z = self.noise_channel(z)
        x = torch.cat((z, y), dim=1)
        x = self.G(x)
        return x


class CondDiscriminator(torch.nn.Module):
    def __init__(self, num_class, embedding_dim, channel_dim):
        super().__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.channel_dim = channel_dim

        self.label_embedding = nn.Embedding(num_class, embedding_dim)
        self.label_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.img_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )
        self.D = nn.Sequential(
            # half of the channels comes from the image, and the other half is from the label
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        """The forward function should return the scores."""
        y = self.label_embedding(y)
        y = y[:, :, None, None].repeat(1, 1, 28, 28)
        y = self.label_channel(y)
        x = self.img_channel(x)
        x = torch.concat((x, y), dim=1)
        p = self.D(x)
        return p


class CDCGAN(object):
    def __init__(
        self, num_class, embedding_dim, channel_dim, latent_dim, epochs, lr, device
    ):
        self.G = (
            CondGenerator(num_class, embedding_dim, latent_dim)
            .apply(weights_init)
            .to(device)
        )
        self.D = (
            CondDiscriminator(num_class, embedding_dim, channel_dim)
            .apply(weights_init)
            .to(device)
        )
        self.loss = nn.BCELoss()

        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.channel_dim = channel_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def train(self, train_loader, verbose_period=2):
        for epoch in range(self.epochs):
            total_d_loss = 0
            total_g_loss = 0
            num_batches = 0
            verbose = (epoch % verbose_period) == 0
            with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
                bar.set_description(f"Epoch {epoch}")
                for images, y in bar:
                    batch_size = images.size(0)
                    # Step 1: Train discriminator
                    z = torch.rand((batch_size, self.latent_dim)).to(self.device)

                    real_labels = torch.ones(batch_size)
                    fake_labels = torch.zeros(batch_size)

                    images, y = images.to(self.device), y.to(self.device).long()

                    real_labels, fake_labels = real_labels.to(
                        self.device
                    ), fake_labels.to(self.device)

                    # Compute the BCE Loss using real images
                    real_scores = self.D(images, y)
                    real_scores = torch.squeeze(real_scores)
                    d_loss_real = self.loss(real_scores, real_labels)

                    # Compute the BCE Loss using fake images
                    fake_images = self.G(z, y)
                    fake_scores = self.D(fake_images, y)
                    fake_scores = torch.squeeze(fake_scores)
                    d_loss_fake = self.loss(fake_scores, fake_labels)

                    # Optimize discriminator
                    d_loss = d_loss_real + d_loss_fake
                    self.D.zero_grad()
                    d_loss.backward()
                    # max log(D(x)) + log(1 - D(G(z))) <=> min -[ log(D(x)) + log(1 - D(G(z))) ]
                    self.d_optimizer.step()

                    # Step 2: Train Generator
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)

                    fake_images = self.G(z, y)
                    fake_scores = self.D(fake_images, y)
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
                    f"Discriminator loss is too small and generator loss is too high at epoch:{epoch}, which indicate a potential saturation problem."
                )
                print("Now, re-initialize and re-train...")
                self.__init__(
                    self.num_class,
                    self.embedding_dim,
                    self.channel_dim,
                    self.latent_dim,
                    self.epochs,
                    self.lr,
                    self.device,
                )
                self.train(train_loader, verbose_period=verbose_period)
                return
        return

    def generate_img(self, number_of_images, class_label, randomness=0.1):
        samples = (
            self.G(
                (randomness * torch.randn((number_of_images, self.latent_dim))).to(
                    self.device
                ),
                (class_label * torch.ones(number_of_images)).long().to(self.device),
            )
            .detach()
            .cpu()
            .reshape(-1, 1, 28, 28)
        )
        samples = samples * 0.5 + 0.5
        return samples
