"""
Module de génération d'images via un générateur entraîné.

Ce module :
- Affiche un formulaire Streamlit pour sélectionner un modèle entraîné
- Charge le générateur et le discriminateur associés
- Génère des images à partir d'un bruit latent
- Affiche les images générées dans l'interface

Utilisé dans l'application principale via `main.py`.

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [23/03/2025]
"""

import streamlit as st
import torch
import numpy as np
import pickle

# Modules internes
from models import (
    GeneratorMLP, GeneratorCNN, GeneratorDCNN,
    DiscriminatorMLP_WGAN, DiscriminatorMLP,
    DiscriminatorCNN_WGAN, DiscriminatorCNN
)

import utils.img_utils as img_utils
import utils.cuda_utils as cuda_utils


def generation_form(listModels):
    """
    Formulaire Streamlit pour choisir le modèle et le nombre d'images.

    Args:
        listModels (list): Liste des noms de modèles disponibles.

    Returns:
        list: [nom_du_modele (str), nombre_de_pokemons (int)]
    """
    st.subheader("Génération d'images")
    st.markdown(
        "Choisissez le modèle à utiliser et le nombre d'images "
        "(Pokémons) à générer :"
    )

    with st.form(key='select_generation'):
        modelName = st.selectbox('Modèle existant', listModels)
        nbPokemons = st.number_input("Nombre de Pokémons à générer",
                                     value=64, min_value=1)
        submit_button = st.form_submit_button(label='Générer !')

    if submit_button:
        return [modelName, nbPokemons]
    return None


def generation():
    """
    Page Streamlit pour générer des images depuis un modèle entraîné.
    """
    st.markdown("---")
    st.header("Génération d'images")
    st.info(
        "Sélectionnez un modèle précédemment entraîné pour générer "
        "de nouveaux Pokémons."
    )

    if st.button("Actualiser la liste des modèles"):
        st.rerun()

    # Chargement du dictionnaire des architectures
    with open('./trainedModels/architecture-model.pkl', 'rb') as f:
        architecture_model = pickle.load(f)

    listModels = list(architecture_model.keys())

    form = generation_form(listModels)
    if form is not None:
        modelName, nbPokemons = form
        generatorChoice, discriminatorChoice, lossChoice = \
            architecture_model[modelName]

        st.write("**Générateur** :", generatorChoice)
        st.write("**Discriminateur** :", discriminatorChoice)
        st.write("**Nom du modèle** :", modelName)

        WGAN = (lossChoice == 'Wasserstein')
        latent_dim = 128
        normalization_stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # Instanciation du générateur
        if generatorChoice == 'MLP':
            generator = GeneratorMLP()
        elif generatorChoice == 'CNN':
            generator = GeneratorCNN()
        elif generatorChoice == 'DCNN':
            generator = GeneratorDCNN()
        else:
            raise ValueError("Générateur non implémenté")

        # Instanciation du discriminateur (optionnel)
        if discriminatorChoice == 'MLP':
            discriminator = (
                DiscriminatorMLP_WGAN() if WGAN else DiscriminatorMLP()
            )
        elif discriminatorChoice == 'CNN':
            discriminator = (
                DiscriminatorCNN_WGAN() if WGAN else DiscriminatorCNN()
            )
        else:
            raise ValueError("Discriminateur non implémenté")

        # Chargement des poids
        device = cuda_utils.get_training_device()

        generator.load_state_dict(torch.load(
            f'./trainedModels/generators/{modelName}.pth',
            map_location=device
        ))
        discriminator.load_state_dict(torch.load(
            f'./trainedModels/discriminators/{modelName}.pth',
            map_location=device
        ))

        generator = generator.to(device)
        generator.eval()

        # Génération d'images
        z = torch.randn(nbPokemons, latent_dim, device=device)
        with torch.no_grad():
            gen_imgs = generator(z).cpu()

        st.success(f"Génération de {nbPokemons} Pokémons réussie !")

        # Affichage des images générées
        img_utils.show_images(
            img_utils.denorm(gen_imgs, normalization_stats),
            normalization_stats,
            nmax=nbPokemons,
            nrow=int(np.sqrt(nbPokemons))
        )


if __name__ == "__main__":
    pass
