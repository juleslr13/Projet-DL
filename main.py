"""
Module principal `main.py`
==========================

Ce module sert de point d'entrée pour l'application **Streamlit** permettant
d'entraîner et de générer des images à l'aide d'un GAN.

Fonctionnalités :
-----------------
- **Entraînement du modèle** : Configuration et lancement de l'entraînement via
  `utils.training.entrainement`.
- **Génération d'images** : Génération d'images de Pokémon via
  `utils.generate.generation`.

Dépendances :
-------------
- **Streamlit** pour l'interface utilisateur.
- **Torch** pour vérifier la présence de CUDA (GPU).
- **Modules `utils.training` et `utils.generate`** pour gérer l'entraînement
  et la génération.

Exécution :
-----------
Ce fichier peut être exécuté directement avec :

    python main.py

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [20/03/2025]
"""

import streamlit as st
from utils.training import entrainement
from utils.generate import generation
import torch


def main():
    """
    Point d'entrée de l'application Streamlit.

    Cette fonction :
    - Configure la page Streamlit.
    - Vérifie la présence d'un GPU.
    - Affiche les instructions d'utilisation.
    - Crée les onglets **Entraînement** et **Génération**.

    """
    # Configuration de la page
    st.set_page_config(
        page_title="GAN App - Pokémon Generator",
        page_icon=":turtle:",
        layout="wide",
    )

    # Style personnalisé
    st.markdown(
        """
        <style>
        .css-18e3th9, .css-1d391kg {
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Générateur de Pokémons avec un GAN :sparkles:")

    # Vérification de la disponibilité du GPU
    if torch.cuda.is_available():
        st.success("CUDA détectée ! L'entraînement utilisera le GPU.")
    else:
        st.warning(
            "CUDA non détectée. L'entraînement se fera sur CPU (plus lent).")

    # Instructions générales
    st.markdown("""
    ---
    **Bienvenue dans cette application de génération d'images Pokémon !**

    - Allez dans l'onglet **Entraînement** pour configurer et entraîner votre
     GAN.
    - Puis dans l'onglet **Génération** pour générer des images à partir d'un
    modèle pré-entraîné.
    """)

    # Création des onglets Streamlit
    tab1, tab2 = st.tabs(["Entraînement", "Génération"])

    with tab1:
        entrainement()  # Fonction de train définie dans utils/training.py
    with tab2:
        generation()  # Fonction de génération définie dans utils/generate.py


# Exécution principale
if __name__ == "__main__":
    main()
