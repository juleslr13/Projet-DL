"""
Module `cuda_utils.py`
======================

Ce module fournit des outils pour gérer l'utilisation de CUDA et déplacer les
données
vers le bon périphérique (GPU ou CPU).

Fonctionnalités :
-----------------
- `get_training_device()` : Vérifie si un GPU est disponible et retourne le
  périphérique approprié.
- `to_device(data, device)` : Déplace les données vers le périphérique
  sélectionné.
- `DeviceDataLoader` : Un wrapper pour déplacer automatiquement les lots de
  données vers le bon périphérique.

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [20/03/25]
"""

import torch


def get_training_device():
    """
    Vérifie si CUDA est disponible et retourne le périphérique
    d'entraînement.

    Returns:
        torch.device: `cuda` si une GPU est disponible, sinon `cpu`.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimisation des CNN sur GPU
        return torch.device('cuda')
    return torch.device('cpu')


def to_device(data, device):
    """
    Déplace un tenseur ou une liste de tenseurs vers le périphérique spécifié.

    Args:
        data (torch.Tensor | list | tuple): Données à déplacer (un tenseur ou
        une liste/tuple de tenseurs).
        device (torch.device): Périphérique cible (`cuda` ou `cpu`).

    Returns:
        torch.Tensor | list | tuple: Données déplacées sur le périphérique
        cible.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """
    Wrapper pour un `DataLoader` qui déplace automatiquement les lots de
    données vers un périphérique donné.

    Attributes:
        dataloader (torch.utils.data.DataLoader): Le `DataLoader` sous-jacent.
        device (torch.device): Le périphérique où les données seront déplacées
        (`cuda` ou `cpu`).
    """

    def __init__(self, dataloader, device):
        """
        Initialise le `DeviceDataLoader`.

        Args:
            dataloader (torch.utils.data.DataLoader): Le `DataLoader` contenant
            les données.
            device (torch.device): Périphérique cible (`cuda` ou `cpu`).
        """
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """
        Itère sur le `DataLoader`, en déplaçant chaque lot vers le périphérique
        cible.

        Yields:
            torch.Tensor | list | tuple: Lot de données déplacé sur `device`.
        """
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self):
        """
        Retourne le nombre de lots dans le `DataLoader`.

        Returns:
            int: Nombre de lots.
        """
        return len(self.dataloader)


if __name__ == "__main__":
    pass
