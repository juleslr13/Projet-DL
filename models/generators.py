"""
Module `generators.py`
======================

Ce module définit différentes architectures de générateurs pour un réseau
antagoniste génératif (GAN).

Classes :
---------
- `GeneratorMLP` : Générateur basé sur un MLP (Multi-Layer Perceptron).
- `GeneratorCNN` : Générateur basé sur un CNN utilisant l'upsampling.
- `GeneratorDCNN` : Générateur basé sur un DCGAN (Deep Convolutional GAN)
  avec des convolutions transposées.

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [20/03/2025]
"""

import torch.nn as nn
import numpy as np


class GeneratorMLP(nn.Module):
    """
    Générateur basé sur un réseau de neurones entièrement connecté (MLP).

    Ce modèle prend un vecteur latent et génère une image en utilisant
    plusieurs couches `Linear`.

    Attributes:
        img_shape (tuple): Dimensions de l'image de sortie
        (channels, height, width).
        model (torch.nn.Sequential): Réseau de couches entièrement connectées.
    """

    def __init__(self, latent_dim=128, img_shape=(3, 64, 64)):
        """
        Initialise l'architecture du générateur MLP.

        Args:
            latent_dim (int, optional): Taille de l'espace latent
            (default: 128).
            img_shape (tuple, optional): Dimensions de l'image générée
            (default: (3, 64, 64)).
        """
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            """
            Crée un bloc de couches pour le MLP.

            Args:
                in_feat (int): Nombre d'entrées de la couche.
                out_feat (int): Nombre de sorties de la couche.
                normalize (bool, optional): Si True, applique une normalisation
                batch (default: True).

            Returns:
                list: Liste des couches du bloc.
            """
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.2))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Passe avant du générateur MLP.

        Args:
            z (torch.Tensor): Vecteur latent de forme (batch_size, latent_dim).

        Returns:
            torch.Tensor: Image générée de forme (batch_size, *img_shape).
        """
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class GeneratorCNN(nn.Module):
    """
    Générateur utilisant un CNN avec `Upsample` pour agrandir l'image
    progressivement.

    Attributes:
        init_size (int): Taille initiale de la feature map avant convolution.
        fc (torch.nn.Linear): Couche fully-connected projetant l'espace latent.
        model (torch.nn.Sequential): Bloc de convolutions et upsampling.
    """

    def __init__(self, latent_dim=128):
        """
        Initialise l'architecture du générateur CNN.

        Args:
            latent_dim (int, optional): Dimension de l'espace latent
            (default: 128).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 64 // 4  # Downsampled size
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalisation des pixels entre -1 et 1
        )

    def forward(self, z):
        """
        Passe avant du générateur CNN.

        Args:
            z (torch.Tensor): Bruit aléatoire de l'espace latent
            (batch_size, latent_dim).

        Returns:
            torch.Tensor: Image générée (batch_size, 3, 64, 64).
        """
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.model(out)
        return img


class GeneratorDCNN(nn.Module):
    """
    Générateur DCGAN utilisant des convolutions transposées pour agrandir
    progressivement l'image.

    Attributes:
        model (torch.nn.Sequential): Bloc de convolutions transposées.
    """

    def __init__(self, latent_dim=128, img_channels=3, feature_maps=128):
        """
        Initialise l'architecture du générateur DCGAN.

        Args:
            latent_dim (int, optional): Dimension de l'espace latent
            (default: 128).
            img_channels (int, optional): Nombre de canaux dans l'image générée
            (default: 3).
            feature_maps (int, optional): Nombre de filtres dans la première
            couche (default: 128).
        """
        super(GeneratorDCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(
                latent_dim,
                feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(
                feature_maps * 8 * 4 * 4),
            nn.LeakyReLU(
                0.2,
                inplace=True),
            nn.Unflatten(
                1,
                (feature_maps * 8,
                 4,
                 4)),
            nn.ConvTranspose2d(
                feature_maps * 8,
                feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm2d(
                feature_maps * 4),
            nn.LeakyReLU(
                0.2,
                inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 4,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm2d(
                feature_maps * 2),
            nn.LeakyReLU(
                0.2,
                inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 2,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(
                0.2,
                inplace=True),
            nn.ConvTranspose2d(
                feature_maps,
                img_channels,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.Tanh())

    def forward(self, z):
        """
        Passe avant du générateur DCGAN.

        Args:
            z (torch.Tensor): Bruit aléatoire de l'espace latent
            (batch_size, latent_dim).

        Returns:
            torch.Tensor: Image générée (batch_size, img_channels, 64, 64).
        """
        img = self.model(z)
        return img


if __name__ == "__main__":
    pass
