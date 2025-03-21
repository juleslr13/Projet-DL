"""
Module `img_utils.py`
======================

Ce module contient des outils pour le chargement, la transformation et
l'affichage d'images dans le cadre de l'entraînement d'un GAN.

Fonctionnalités :
-----------------
- `make_dataset()` : Charge un dataset d'images avec transformations.
- `denorm()` : Annule la normalisation des images.
- `show_images()` : Affiche une grille d'images.
- `show_batch()` : Affiche un lot d'images provenant d'un `DataLoader`.

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [20/03/2025]
"""

import torchvision.transforms as T
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import streamlit as st

# Variable globale pour stocker les statistiques de normalisation


def make_dataset(IMAGE_DIR, image_size, batch_size, normalization_stats):
    """
    Charge un dataset d'images et applique des transformations.

    Ce dataset combine des images normales et des images avec `ColorJitter`.

    Args:
        IMAGE_DIR (str): Chemin du dossier contenant les images.
        image_size (int): Taille des images après redimensionnement.
        batch_size (int): Taille du batch pour le DataLoader.
        normalization_stats (tuple): Moyenne et écart-type pour la
        normalisation.

    Returns:
        torch.utils.data.ConcatDataset: Dataset contenant les transformations
        appliquées.
    """

    normal_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*normalization_stats)
    ]))

    color_jitter_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ColorJitter(0, 0.2, 1),
        T.ToTensor(),
        T.Normalize(*normalization_stats)
    ]))

    dataset_list = [normal_dataset, color_jitter_dataset]
    dataset = ConcatDataset(dataset_list)

    return dataset


def denorm(image,norm_stats):
    """
    Annule la normalisation d'une image.

    Args:
        image (torch.Tensor): Image normalisée.

    Returns:
        torch.Tensor: Image remise à l'échelle originale.
    """
    return image * norm_stats[1][0] + norm_stats[0][0]


def show_images(images, norm_stats, nmax=64, nrow=8):
    """
    Affiche une grille d'images.

    Args:
        images (torch.Tensor): Lot d'images à afficher.
        nmax (int, optional): Nombre maximum d'images à afficher (default: 64).
        nrow (int, optional): Nombre d'images par ligne (default: 8).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax],norm_stats), nrow=nrow).
              permute(1, 2, 0))
    st.pyplot(fig)


def show_batch(dataloader, norm_stats, nmax=64):
    """
    Affiche un lot d'images provenant d'un `DataLoader`.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader contenant les
        images.
        nmax (int, optional): Nombre maximum d'images à afficher (default: 64).
    """
    for images, _ in dataloader:
        show_images(images, norm_stats, nmax)
        break


if __name__ == "__main__":
    pass
