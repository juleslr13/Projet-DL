import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np

class GeneratorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim=128
        img_shape=(3,64,64)
        self.img_shape=img_shape
        def block(in_feat, out_feat,normalize=True):
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
        img = self.model(z)
        img = img.view(img.shape[0],*self.img_shape)
        return img


class GeneratorCNN(nn.Module):
    def __init__(self, latent_dim=128):
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
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.model(out)
        return img

class GeneratorDCNN(nn.Module):
    """
    Générateur CNN profond utilisant des convolutions transposées pour la montée en dimension.

    Attributs:
        latent_dim (int): Taille de l'espace latent.
        img_channels (int): Nombre de canaux des images générées.
        feature_maps (int): Nombre de filtres pour la première couche convolutive.
    """

    def __init__(self, latent_dim=128, img_channels=3, feature_maps=128):
        """
        Initialise l'architecture du générateur.

        Args:
            latent_dim (int): Dimension de l'espace latent.
            img_channels (int): Nombre de canaux des images générées.
            feature_maps (int): Nombre de filtres pour la première couche convolutive.
        """
        super(GeneratorDCNN, self).__init__()

        self.model = nn.Sequential(
            # Première couche - linéaire -> reshape en un tenseur 3D
            nn.Linear(latent_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape en tenseur 4D
            nn.Unflatten(1, (feature_maps * 8, 4, 4)),  

            # Bloc 1 : 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloc 2 : 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloc 3 : 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),

            # Dernière couche : 32x32 -> 64x64
            
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Normalisation des pixels entre -1 et 1
        )
    def forward(self, z):
        """
        Passe avant du générateur.

        Args:
            z (torch.Tensor): Bruit aléatoire de l'espace latent (batch_size, latent_dim).

        Returns:
            torch.Tensor: Image générée (batch_size, img_channels, 64, 64).
        """
        img = self.model(z)
        return img