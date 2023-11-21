"""NN models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm


def weights_init(m):
    """Initialize parameters/weights in GAN."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return


class CondGenerator(torch.nn.Module):
    """Conditional Generator.

    Vanilla DCGAN generator architecture.
    """

    def __init__(
        self, num_class, channel_dim, latent_dim, embedding_dim, one_hot_encoding
    ):
        super().__init__()
        self.num_class = num_class
        self.latent_dim = latent_dim
        if one_hot_encoding:
            self.embedding_dim = num_class
        else:
            self.embedding_dim = embedding_dim
            self.label_embedding = nn.Embedding(num_class, embedding_dim)
        self.one_hot_encoding = one_hot_encoding

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
            nn.BatchNorm2d(128),
            nn.ReLU(),
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
                in_channels=64,
                out_channels=channel_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z, y):
        """The forward function should return batch of images.

        Concatenate one-hot/embedding representation of the label and latent noise. Generate/enrich images from the combination.
        """
        z = z.reshape(-1, self.latent_dim, 1, 1)
        if self.one_hot_encoding:
            y = F.one_hot(y, num_classes=self.num_class).float()
        else:
            y = self.label_embedding(y)
        y = self.label_channel(y.reshape(-1, self.embedding_dim, 1, 1))
        z = self.noise_channel(z)
        x = torch.cat((z, y), dim=1)
        x = self.G(x)
        return x


class CondDiscriminator(torch.nn.Module):
    """Conditional Discriminator.

    Vanilla DCGAN discriminator architecture.
    """

    def __init__(self, num_class, channel_dim, embedding_dim, one_hot_encoding):
        super().__init__()
        self.num_class = num_class
        self.channel_dim = channel_dim
        if one_hot_encoding:
            self.embedding_dim = num_class
        else:
            self.embedding_dim = embedding_dim
            self.label_embedding = nn.Embedding(num_class, embedding_dim)
        self.one_hot_encoding = one_hot_encoding

        self.label_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
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
        """The forward function should return the scores.

        Enrich the one-hot/embedding representation of the label to constant tensor with layers/channels.
        """
        if self.one_hot_encoding:
            y = F.one_hot(y, num_classes=self.num_class).float()
        else:
            y = self.label_embedding(y)
        y = y[:, :, None, None].repeat(1, 1, 28, 28)
        y = self.label_channel(y)
        x = self.img_channel(x)
        x = torch.concat((x, y), dim=1)
        p = self.D(x)
        return p


class CDCGAN(object):
    def __init__(
        self,
        num_class,
        channel_dim,
        latent_dim,
        embedding_dim,
        one_hot_encoding,
        epochs,
        lr,
        device,
    ):
        self.G = (
            CondGenerator(
                num_class, channel_dim, latent_dim, embedding_dim, one_hot_encoding
            )
            .apply(weights_init)
            .to(device)
        )
        self.D = (
            CondDiscriminator(num_class, channel_dim, embedding_dim, one_hot_encoding)
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
        self.one_hot_encoding = one_hot_encoding
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
                    z = torch.randn((batch_size, self.latent_dim)).to(self.device)

                    real_labels = torch.ones(batch_size)
                    fake_labels = torch.zeros(batch_size)

                    images, y = (
                        images.to(self.device),
                        y.reshape(-1).to(self.device).long(),
                    )

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
                    f"Discriminator loss is too small and generator loss is too high at epoch:{epoch}, "
                    + "which indicate a potential saturation problem."
                )
                print("Now, re-initialize and re-train...")
                self.__init__(
                    self.num_class,
                    self.channel_dim,
                    self.latent_dim,
                    self.embedding_dim,
                    self.one_hot_encoding,
                    self.epochs,
                    self.lr,
                    self.device,
                )
                self.train(train_loader, verbose_period=verbose_period)
                return
        return

    def generate_img(self, number_of_images, class_label, channel_dim):
        """Generate images from noise and class label."""
        samples = (
            self.G(
                torch.randn((number_of_images, self.latent_dim)).to(self.device),
                (class_label * torch.ones(number_of_images)).long().to(self.device),
            )
            .detach()
            .cpu()
            .reshape(-1, channel_dim, 28, 28)
        )
        samples = samples * 0.5 + 0.5
        return samples


def gradient_penalty(D, real_images, image_labels, fake_images, device):
    """Compute the gradient penalty loss for WGAN.

    L2 Regularize/penalize discriminator's weight gradients L2 norm being greater than 1.
    """
    N, C, H, W = real_images.shape
    alpha = torch.randn((N, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # get X_hat, the interpolation between real samples and fake samples
    interpolated_images = real_images * alpha + fake_images * (1 - alpha)
    interpolated_scores = D(interpolated_images, image_labels)
    # get the grad D(X_hat)
    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # compute the penalty
    gradients = gradients.reshape(N, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


class CondWGenerator(torch.nn.Module):
    """Conditional Generator (same as CondGenerator).

    DCGAN generator architecture.
    """

    def __init__(
        self, num_class, channel_dim, latent_dim, embedding_dim, one_hot_encoding
    ):
        super().__init__()
        self.num_class = num_class
        self.latent_dim = latent_dim
        if one_hot_encoding:
            self.embedding_dim = num_class
        else:
            self.embedding_dim = embedding_dim
            self.label_embedding = nn.Embedding(num_class, embedding_dim)
        self.one_hot_encoding = one_hot_encoding

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
            nn.BatchNorm2d(128),
            nn.ReLU(),
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
                in_channels=64,
                out_channels=channel_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z, y):
        """The forward function should return batch of images."""
        z = z.reshape(-1, self.latent_dim, 1, 1)
        if self.one_hot_encoding:
            y = F.one_hot(y, num_classes=self.num_class).float()
        else:
            y = self.label_embedding(y)
        y = self.label_channel(y.reshape(-1, self.embedding_dim, 1, 1))
        z = self.noise_channel(z)
        x = torch.cat((z, y), dim=1)
        x = self.G(x)
        return x


class CondWDiscriminator(torch.nn.Module):
    """Conditional Discriminator.

    Wassertein DCGAN discriminator architecture. Sigmoid removed.
    """

    def __init__(self, num_class, channel_dim, embedding_dim, one_hot_encoding):
        super().__init__()
        self.num_class = num_class
        self.channel_dim = channel_dim
        if one_hot_encoding:
            self.embedding_dim = num_class
        else:
            self.embedding_dim = embedding_dim
            self.label_embedding = nn.Embedding(num_class, embedding_dim)
        self.one_hot_encoding = one_hot_encoding

        self.label_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
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
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(256, affine=True),
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
        )

    def forward(self, x, y):
        """The forward function should return the scores."""
        if self.one_hot_encoding:
            y = F.one_hot(y, num_classes=self.num_class).float()
        else:
            y = self.label_embedding(y)
        y = y[:, :, None, None].repeat(1, 1, 28, 28)
        y = self.label_channel(y)
        x = self.img_channel(x)
        x = torch.concat((x, y), dim=1)
        p = self.D(x)
        return p


class CWDCGAN(object):
    def __init__(
        self,
        num_class,
        channel_dim,
        latent_dim,
        embedding_dim,
        one_hot_encoding,
        reg_lambda,
        n_critic,
        epochs,
        lr,
        device,
    ):
        self.G = (
            CondWGenerator(
                num_class, channel_dim, latent_dim, embedding_dim, one_hot_encoding
            )
            .apply(weights_init)
            .to(device)
        )
        self.D = (
            CondWDiscriminator(num_class, channel_dim, embedding_dim, one_hot_encoding)
            .apply(weights_init)
            .to(device)
        )

        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.num_class = num_class
        self.one_hot_encoding = one_hot_encoding
        self.embedding_dim = embedding_dim
        self.channel_dim = channel_dim
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        self.n_critic = n_critic
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
                    images, y = (
                        images.to(self.device),
                        y.to(self.device).reshape(-1).long(),
                    )
                    # Step 1: Train discriminator with (n_critic) iters
                    for _ in range(self.n_critic):
                        # Generate noise
                        z = torch.randn((batch_size, self.latent_dim)).to(self.device)

                        # Compute the BCE Loss using real images
                        real_scores = self.D(images, y)
                        real_scores = torch.squeeze(real_scores)

                        # Compute the BCE Loss using fake images
                        fake_images = self.G(z, y)
                        fake_scores = self.D(fake_images, y)
                        fake_scores = torch.squeeze(fake_scores)

                        # Compute the regularization term
                        gp = gradient_penalty(
                            D=self.D,
                            real_images=images,
                            image_labels=y,
                            fake_images=fake_images,
                            device=self.device,
                        )
                        # Optimize discriminator
                        # max [ D(x) - D(G(z)) ] => min -[ D(x) - D(G(z)) ] + regularization
                        d_loss = (
                            -(torch.mean(real_scores) - torch.mean(fake_scores))
                            + self.reg_lambda * gp
                        )

                        # Optimize discriminator
                        self.D.zero_grad()
                        d_loss.backward(retain_graph=True)
                        self.d_optimizer.step()

                    # Step 2: Train Generator
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)

                    fake_images = self.G(z, y)
                    fake_scores = self.D(fake_images, y)
                    fake_scores = torch.squeeze(fake_scores)

                    # min [ - D(G(z)) ]
                    g_loss = -torch.mean(fake_scores)

                    # Optimize generator
                    self.D.zero_grad()
                    self.G.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # update bar
                    num_batches += 1
                    total_d_loss += d_loss.item()
                    total_g_loss += g_loss.item()
                    bar.set_postfix(
                        d_loss=float(total_d_loss / num_batches),
                        g_loss=float(total_g_loss / num_batches),
                    )
        return

    def generate_img(self, number_of_images, class_label, channel_dim):
        """Generate images from noise and class label."""
        samples = (
            self.G(
                torch.randn((number_of_images, self.latent_dim)).to(self.device),
                (class_label * torch.ones(number_of_images)).long().to(self.device),
            )
            .detach()
            .cpu()
            .reshape(-1, channel_dim, 28, 28)
        )
        samples = samples * 0.5 + 0.5
        return samples


class CNN(nn.Module):
    def __init__(self, channel_dim, num_class, device):
        super().__init__()
        self.num_class = num_class
        self.device = device
        # download resnet
        resnet = torchvision.models.resnet50(weights="DEFAULT")
        resnet.conv1 = nn.Conv2d(
            channel_dim,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        resnet.fc = nn.Linear(in_features=2048, out_features=num_class, bias=True)
        self.resnet = resnet.to(device)

    def forward(self, x):
        return F.softmax(self.resnet(x), dim=1)

    def evaluate(self, dataloader):
        total_correct = 0
        total_loss = 0
        num_samples = len(dataloader.dataset)
        num_batches = 0
        criterion = nn.CrossEntropyLoss(reduction="mean")
        with torch.no_grad():
            for img_batch, label_batch in dataloader:
                img_batch = img_batch.to(self.device)
                label_batch = label_batch.to(self.device).reshape(-1)
                prob_batch = self(img_batch)
                # loss
                total_loss += criterion(prob_batch, label_batch).item()
                num_batches += 1
                # num correct
                total_correct += (prob_batch.argmax(dim=1) == label_batch).sum().item()
        return total_loss / num_batches, total_correct / num_samples

    def train(self, train_dataloader, valid_dataloader, lr, epochs, verbose_period=1):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_correct = 0
            total_loss = 0
            num_samples = 0
            num_batches = 0
            verbose = (epoch % verbose_period) == 0
            with tqdm(train_dataloader, unit="batch", disable=not verbose) as bar:
                bar.set_description(f"Epoch {epoch}")
                for img_batch, label_batch in bar:
                    img_batch = img_batch.to(self.device)
                    label_batch = label_batch.to(self.device).reshape(-1)
                    # forward, backward & optimizer step
                    self.zero_grad()
                    prob_batch = self(img_batch)
                    loss = criterion(prob_batch, label_batch)
                    loss.backward()
                    optimizer.step()
                    # update bar
                    total_loss += loss.item()
                    num_batches += 1
                    total_correct += (
                        (prob_batch.argmax(dim=1) == label_batch).sum().item()
                    )
                    num_samples += img_batch.shape[0]
                    bar.set_postfix(
                        ce=float(total_loss / num_batches),
                        acc=float(total_correct / num_samples),
                    )
            valid_loss, valid_acc = self.evaluate(valid_dataloader)
            print("val_ce={:0.3f}, val_acc={:0.3f}".format(valid_loss, valid_acc))
        return
