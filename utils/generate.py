import streamlit as st
import torch
import numpy as np
import pickle
from torchvision.transforms.functional import to_pil_image

# Imports de vos modules
from models import (
    GeneratorMLP, GeneratorCNN, GeneratorDCNN,
    DiscriminatorMLP_WGAN, DiscriminatorMLP,
    DiscriminatorCNN_WGAN, DiscriminatorCNN
)
import utils.img_utils as img_utils
import utils.cuda_utils as cuda_utils


def generation_form(listModels):
    """
    Formulaire Streamlit pour sélectionner le modèle entraîné et le nombre de Pokémons à générer.
    """
    st.subheader("Génération d'images")
    st.markdown("Choisissez le modèle à utiliser et le nombre d'images (Pokémons) à générer :")

    with st.form(key='select_generation'):
        modelName = st.selectbox('Modèle existant', listModels)
        nbPokemons = st.number_input("Nombre de Pokémons à générer", value=64, min_value=1)
        submit_button = st.form_submit_button(label='Générer !')

    if submit_button:
        return [modelName, nbPokemons]


def generation():
    """
    Page Streamlit pour générer des images depuis un modèle existant.
    """
    st.markdown("---")
    st.header("Génération d'images")
    st.info("Sélectionnez un modèle précédemment entraîné pour générer de nouveaux Pokémons.")

    # Bouton pour recharger la liste
    if st.button("Actualiser la liste des modèles"):
        st.rerun()

    # Chargement du dictionnaire {nom_de_modele: [gen, disc, loss]}
    with open('./trainedModels/architecture-model.pkl', 'rb') as f:
        architecture_model = pickle.load(f)

    listModels = list(architecture_model.keys())

    # Formulaire
    form = generation_form(listModels)
    if form is not None:
        modelName, nbPokemons = form
        generatorChoice, discriminatorChoice, lossChoice = architecture_model[modelName]

        st.write("**Générateur** :", generatorChoice)
        st.write("**Discriminateur** :", discriminatorChoice)
        st.write("**Nom du modèle** :", modelName)

        WGAN = (lossChoice == 'Wasserstein')
        latent_dim = 128

        # Instanciation du générateur
        if generatorChoice == 'MLP':
            generator = GeneratorMLP()
        elif generatorChoice == 'CNN':
            generator = GeneratorCNN()
        elif generatorChoice == 'DCNN':
            generator = GeneratorDCNN()
        else:
            raise Exception("Générateur non implémenté")

        # Instanciation du discriminant (même si on ne s’en sert pas pour la génération,
        # on peut le charger pour cohérence – à vous de voir s’il est nécessaire)
        if discriminatorChoice == 'MLP':
            discriminator = DiscriminatorMLP_WGAN() if WGAN else DiscriminatorMLP()
        elif discriminatorChoice == 'CNN':
            discriminator = DiscriminatorCNN_WGAN() if WGAN else DiscriminatorCNN()
        else:
            raise Exception("Discriminateur non implémenté")

        # Chargement des poids
        generator.load_state_dict(torch.load(f'./trainedModels/generators/{modelName}.pth'))
        discriminator.load_state_dict(torch.load(f'./trainedModels/discriminators/{modelName}.pth'))

        device = cuda_utils.get_training_device()
        generator = generator.to(device)
        generator.eval()

        # Génération
        z = torch.randn(nbPokemons, latent_dim, device=device)
        with torch.no_grad():
            gen_imgs = generator(z).cpu()

        st.success(f"Génération de {nbPokemons} Pokémons réussie !")

        # Affichage sous forme d'une grille
        img_utils.show_images(
            img_utils.denorm(gen_imgs),
            nbPokemons,
            nrow=int(np.sqrt(nbPokemons))
        )
