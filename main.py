import streamlit as st
from utils.training import entrainement
from utils.generate import generation

def main():
    """
    Point d'entrée de l'application Streamlit.
    """
    # Configuration de base
    st.set_page_config(
        page_title="GAN App - Pokémon Generator",
        page_icon=":turtle:",
        layout="wide",
    )

    # Petit style global
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

    # Présence (ou non) du GPU
    import torch
    if torch.cuda.is_available():
        st.success("CUDA détectée ! L'entraînement utilisera le GPU.")
    else:
        st.warning("CUDA non détectée. L'entraînement se fera sur CPU (plus lent).")

    st.markdown("""
    ---
    **Bienvenue dans cette application de génération d'images Pokémon !**

    - Rendez-vous dans l'onglet **Entraînement** pour configurer et entraîner votre GAN.
    - Puis dans l'onglet **Génération** pour générer des images à partir d'un modèle déjà entraîné.
    """)

    tab1, tab2 = st.tabs(["Entraînement", "Génération"])

    with tab1:
        entrainement()   # Appel vers la fonction d'entraînement du module utils/training.py
    with tab2:
        generation()     # Appel vers la fonction de génération du module utils/generate.py

if __name__ == "__main__":
    main()
