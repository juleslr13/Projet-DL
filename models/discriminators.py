"""
Module `discriminators.py`
==========================

Ce module définit plusieurs architectures de discriminateurs pour un GAN
(réseau antagoniste génératif).

Classes :
---------
- `DiscriminatorMLP` : Discriminateur basé sur un MLP
  (Multi-Layer Perceptron).
- `DiscriminatorMLP_WGAN` : Variante MLP pour un Wasserstein GAN.
- `DiscriminatorCNN` : Discriminateur basé sur des convolutions (DCGAN).
- `DiscriminatorCNN_WGAN` : Variante CNN pour un Wasserstein GAN.

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [20/03/2025]
"""

import torch.nn as nn
import numpy as np


class DiscriminatorMLP(nn.Module):
    """
    Discriminateur basé sur un réseau de neurones entièrement connecté
    (MLP).

    Ce modèle prend une image sous forme de tenseur, l'aplatit et la passe
    dans un réseau de couches linéaires pour prédire sa validité.

    Attributes:
        model (torch.nn.Sequential): Réseau de couches entièrement
        connectées.
    """

    def __init__(self):
        """
        Initialise l'architecture du discriminateur MLP.
        """
        super().__init__()
        img_shape = (3, 64, 64)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        """
        Passe avant du discriminateur MLP.

        Args:
            img (torch.Tensor): Image d'entrée sous forme de tenseur
            `(batch_size, 3, 64, 64)`.

        Returns:
            torch.Tensor: Score de validité de l'image.
        """
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return nn.Sigmoid()(validity)


class DiscriminatorMLP_WGAN(nn.Module):
    """
    Variante du `DiscriminatorMLP` pour Wasserstein GAN (WGAN).

    Ce modèle ne contient pas de fonction d'activation `Sigmoid`,
    car WGAN utilise une perte basée sur une distance Wasserstein.

    Attributes:
        model (torch.nn.Sequential): Réseau de couches entièrement
        connectées.
    """

    def __init__(self):
        """
        Initialise l'architecture du discriminateur MLP pour WGAN.
        """
        super().__init__()
        img_shape = (3, 64, 64)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        """
        Passe avant du discriminateur MLP WGAN.

        Args:
            img (torch.Tensor): Image d'entrée sous forme de tenseur
            `(batch_size, 3, 64, 64)`.

        Returns:
            torch.Tensor: Score Wasserstein de l'image.
        """
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class DiscriminatorCNN(nn.Module):
    """
    Discriminateur basé sur un réseau convolutif (CNN), utilisé dans DCGAN.

    Ce modèle applique une série de convolutions pour extraire des
    caractéristiques de l'image, puis passe à une couche dense finale.

    Attributes:
        model (torch.nn.Sequential): Réseau de couches convolutives.
    """

    def __init__(self):
        """
        Initialise l'architecture du discriminateur CNN.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # 64x64 -> 4x4 feature map
        )

    def forward(self, img):
        """
        Passe avant du discriminateur CNN.

        Args:
            img (torch.Tensor): Image d'entrée sous forme de tenseur
            `(batch_size, 3, 64, 64)`.

        Returns:
            torch.Tensor: Score de validité de l'image.
        """
        validity = self.model(img)
        return nn.Sigmoid()(validity)


class DiscriminatorCNN_WGAN(nn.Module):
    """
    Variante du `DiscriminatorCNN` pour Wasserstein GAN (WGAN).

    Ce modèle applique une série de convolutions et ne contient pas
    de fonction d'activation `Sigmoid`, car WGAN utilise une perte Wasserstein.

    Attributes:
        model (torch.nn.Sequential): Réseau de couches convolutives.
    """

    def __init__(self):
        """
        Initialise l'architecture du discriminateur CNN pour WGAN.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # 64x64 -> 4x4 feature map
        )

    def forward(self, img):
        """
        Passe avant du discriminateur CNN WGAN.

        Args:
            img (torch.Tensor): Image d'entrée sous forme de tenseur
            `(batch_size, 3, 64, 64)`.

        Returns:
            torch.Tensor: Score Wasserstein de l'image.
        """
        validity = self.model(img)
        return validity


if __name__ == "__main__":
    pass
