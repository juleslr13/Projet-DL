"""
Module `generators.py`
======================

Ce module définit les architectures des générateurs utilisés dans un modèle GAN.

Classes :
    - GeneratorMLP : Générateur basé sur un perceptron multi-couche (MLP).
    - GeneratorCNN : Générateur basé sur un réseau de convolution transposé (DCGAN).

Auteur : [Gleyo, Le Roy, Legris]
Date : [14/03/2025]
"""

import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary

class GeneratorMLP(nn.Module):
    """
    Générateur basé sur un réseau de neurones entièrement connecté (MLP).

    Ce modèle prend un vecteur latent en entrée et génère une image sous forme de tenseur.

    Attributes:
        img_shape (tuple): Dimensions de l'image de sortie (channels, height, width).
        model (torch.nn.Sequential): Réseau de couches entièrement connectées.
    """

    def __init__(self, latent_dim=128, img_shape=(3, 64, 64)):
        """
        Initialise le générateur MLP.

        Args:
            latent_dim (int, optional): Taille de l'espace latent (default: 128).
            img_shape (tuple, optional): Dimensions de l'image générée (default: (3, 64, 64)).
        """
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            """
            Crée un bloc de couches pour le réseau MLP.

            Args:
                in_feat (int): Nombre d'entrées de la couche.
                out_feat (int): Nombre de sorties de la couche.
                normalize (bool, optional): Si True, applique une normalisation batch (default: True).

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
    Générateur DCGAN utilisant des convolutions transposées pour agrandir progressivement l'image.

    Ce modèle prend un vecteur latent en entrée et génère une image via une série de convolutions transposées.

    Attributes:
        init_size (int): Taille initiale de la feature map avant convolution (default: 4).
        fc (torch.nn.Linear): Couche fully-connected projetant l'espace latent.
        conv_blocks (torch.nn.Sequential): Bloc de convolutions transposées.
    """

    def __init__(self, latent_dim=128, img_channels=3, feature_maps=512):
        """
        Initialise le générateur CNN.

        Args:
            latent_dim (int, optional): Dimension de l'espace latent (default: 128).
            img_channels (int, optional): Nombre de canaux dans l'image générée (default: 3).
            feature_maps (int, optional): Nombre de filtres dans la première couche (default: 512).
        """
        super(GeneratorCNN, self).__init__()

        self.init_size = 4  # Taille initiale après projection
        self.fc = nn.Linear(latent_dim, feature_maps * self.init_size * self.init_size)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_maps),

            # 4x4 → 8x8
            nn.ConvTranspose2d(feature_maps, feature_maps // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(inplace=True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(feature_maps // 2, feature_maps // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps // 4),
            nn.ReLU(inplace=True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(feature_maps // 4, feature_maps // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps // 8),
            nn.ReLU(inplace=True),

            # 32x32 → 64x64 (Sortie finale)
            nn.ConvTranspose2d(feature_maps // 8, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Pixels normalisés entre -1 et 1
        )

    def forward(self, z):
        """
        Passe avant du générateur CNN.

        Args:
            z (torch.Tensor): Bruit aléatoire de l'espace latent (batch_size, latent_dim).

        Returns:
            torch.Tensor: Image générée (batch_size, img_channels, 64, 64).
        """
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)  # Reshape en (batch, 512, 4, 4)
        img = self.conv_blocks(out)
        return img