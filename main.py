"""
Module principal pour l'entraînement et la génération d'images avec un GAN.

Ce fichier contient l'interface Streamlit pour configurer et entraîner un GAN basé sur MLP ou CNN.

- Sélection des modèles (Générateur et Discriminateur)
- Sélection de la fonction de perte (Wasserstein ou Cross-Entropy)
- Entraînement du modèle et sauvegarde

Auteur : [Gleyo, Le Roy, Legris]
Date : [14/03/2025]
"""

import torch
from torch.utils.data import DataLoader
import streamlit as st
from torchvision.utils import save_image
import torch.nn.functional as F
from stqdm import stqdm
import pickle
# Custom imports
from models.generators import GeneratorMLP, GeneratorCNN
from models.discriminators import DiscriminatorMLP, DiscriminatorCNN
import utils.cuda_utils as cuda_utils
import utils.img_utils as img_utils

def load_dataset():
    """
    Charge le dataset d'images et configure le DataLoader.

    Returns:
        DataLoader: DataLoader pour les images normalisées.
    """
    IMAGE_DIR = "./augmentedData"
    image_size = 64
    batch_size = 32
    normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset = img_utils.make_dataset(IMAGE_DIR, image_size, batch_size, normalization_stats)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False)
    return dataloader

def initialize_device():
    """
    Initialise le périphérique d'entraînement (CPU ou GPU).

    Returns:
        torch.device: Périphérique utilisé.
    """
    device = cuda_utils.get_training_device()
    return device

def fill_form():
    """
    Affiche un formulaire Streamlit pour sélectionner les hyperparamètres du modèle.

    Returns:
        list: Liste des choix de l'utilisateur [générateur, discriminateur, loss, epochs, modelName].
    """
    with st.form(key='select_model'):
        generatorChoice = st.selectbox('Veuillez sélectionner un type de générateur :', ('MLP', 'CNN'))
        discriminatorChoice = st.selectbox('Veuillez sélectionner un type de discriminant :', ('MLP', 'CNN'))
        lossChoice = st.selectbox('Veuillez sélectionner le type de loss :', ('Cross-entropy', 'Wasserstein'))
        modelName = st.text_input("Nom du modèle: ")
        epochs = st.number_input(label="Nombre d'epochs", value=100)
        submit_button = st.form_submit_button(label='Generate pokemons !')

    if submit_button or st.session_state.choices is not None:
        return [generatorChoice, discriminatorChoice, lossChoice, epochs, modelName]
    return None

def saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice, edit=False):
    """
    Sauvegarde les modèles entraînés.

    Args:
        generator (nn.Module): Modèle générateur.
        discriminator (nn.Module): Modèle discriminateur.
        modelName (str): Nom du modèle sauvegardé.
        generatorChoice (str): Type de générateur choisi.
        discriminatorChoice (str): Type de discriminateur choisi.
        edit (bool, optional): Si True, met à jour les métadonnées du modèle. Defaults to False.
    """
    torch.save(generator.state_dict(), "./trainedModels/generators/" + modelName)
    torch.save(discriminator.state_dict(), "./trainedModels/discriminators/" + modelName)
    if edit:
        with open('./trainedModels/architecture-model.pkl', 'rb') as f:
            architecture_model = pickle.load(f)
        architecture_model[modelName] = [generatorChoice, discriminatorChoice]
        with open('./trainedModels/architecture-model.pkl', 'wb') as f:
            pickle.dump(architecture_model, f)

def entrainement():
    """
    Fonction principale d'entraînement du GAN.
    """
    if st.session_state.choices is None:
        st.session_state.choices = fill_form()
        st.rerun()
    else:
        generatorChoice, discriminatorChoice, lossChoice, epochs, modelName = st.session_state.choices
        st.write("Générateur: " + generatorChoice)
        st.write("Discriminateur: " + discriminatorChoice)
        st.write("Nom du modèle: " + modelName)

        # Initialisation des modèles
        generator = GeneratorMLP() if generatorChoice == 'MLP' else GeneratorCNN()
        discriminator = DiscriminatorMLP() if discriminatorChoice == 'MLP' else DiscriminatorCNN()

        device = initialize_device()
        if device.type == 'cuda':
            generator.cuda()
            discriminator.cuda()

        if st.button("Save Model & Reset"):
            saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice, edit=True)
            st.session_state.choices = None
            st.rerun()

        # Optimiseurs et boucle d'entraînement (à améliorer si besoin)

def main():
    """
    Fonction principale gérant l'interface Streamlit.
    """
    tab1, tab2 = st.tabs(["Entraînement", "Génération"])
    with tab1:
        entrainement()
    with tab2:
        st.write("Work in progress")

if __name__ == "__main__":
    main()
