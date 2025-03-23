"""
Module de configuration et d'entraînement d'un GAN via Streamlit.

Ce module propose :
- Un formulaire Streamlit pour configurer les hyperparamètres
- Une fonction `entrainement()` pour lancer l'entraînement du GAN
- Sauvegarde des modèles entraînés dans `./trainedModels/`

Utilisé par l'interface principale (main.py).

Auteur : [Gleyo Alexis, Le Roy Jules, Legris Simon]
Date : [20/03/25]
"""

import pickle
import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from stqdm import stqdm

from models import (
    GeneratorMLP, GeneratorCNN, GeneratorDCNN,
    DiscriminatorMLP, DiscriminatorCNN,
    DiscriminatorMLP_WGAN, DiscriminatorCNN_WGAN
)

import utils.cuda_utils as cuda_utils
import utils.img_utils as img_utils


def fill_form():
    """
    Formulaire Streamlit pour sélectionner les hyperparamètres du modèle.
    """
    st.subheader("Configuration du GAN")
    st.markdown("Choisissez les options pour votre modèle :")

    with st.form(key='select_model'):
        generatorChoice = st.selectbox('Type de générateur',
                                       ('MLP', 'CNN', 'DCNN'))
        discriminatorChoice = st.selectbox('Type de discriminant',
                                           ('MLP', 'CNN'))
        lossChoice = st.selectbox('Fonction de perte',
                                  ('Cross-entropy', 'Wasserstein'))
        modelName = st.text_input("Nom du modèle "
                                  "(sans espace ni caractères spéciaux) :")
        epochs = st.number_input("Nombre d'epochs", value=100, min_value=1,
                                 step=1)
        learning_rate_g = st.number_input(
            "Learning rate (Générateur)", value=1e-3, min_value=1e-6,
            max_value=1.0, step=1e-5, format="%.6f")
        learning_rate_d = st.number_input(
            "Learning rate (Discriminateur)", value=1e-3, min_value=1e-6,
            max_value=1.0, step=1e-5, format="%.6f"
        )
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
        submit_button = st.form_submit_button(label='Valider')

    if submit_button:
        if not modelName:
            st.error("Le nom du modèle ne peut pas être vide.")
            return
        if " " in modelName:
            st.error("Le nom du modèle ne doit pas contenir d'espaces.")
            return
        allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
        for char in modelName:
            if char not in allowed_chars:
                st.error("Utilisez uniquement lettres, "
                         "chiffres et underscores.")
                return

        st.session_state.choices = [
            generatorChoice, discriminatorChoice, lossChoice,
            epochs, modelName, learning_rate_g, learning_rate_d, batch_size
        ]


def saveModel(generator, discriminator, modelName,
              generatorChoice, discriminatorChoice, lossChoice, edit=False):
    """
    Sauvegarde les modèles et met à jour le dictionnaire des architectures.

    Args:
        generator (nn.Module): Générateur entraîné.
        discriminator (nn.Module): Discriminateur entraîné.
        modelName (str): Nom du modèle.
        generatorChoice (str): Type de générateur.
        discriminatorChoice (str): Type de discriminateur.
        lossChoice (str): Type de fonction de perte.
        edit (bool): Met à jour le fichier architecture-model.pkl
        si True.
    """
    torch.save(generator.state_dict(),
               f"./trainedModels/generators/{modelName}.pth")
    torch.save(discriminator.state_dict(),
               f"./trainedModels/discriminators/{modelName}.pth")

    if edit:
        with open('./trainedModels/architecture-model.pkl', 'rb') as f:
            architecture_model = pickle.load(f)
        architecture_model[modelName] = [generatorChoice, discriminatorChoice,
                                         lossChoice]
        with open('./trainedModels/architecture-model.pkl', 'wb') as f:
            pickle.dump(architecture_model, f)


def entrainement():
    """
    Page Streamlit pour l'entraînement d'un nouveau modèle GAN.
    """
    st.markdown("---")
    st.header("Entraînement du modèle")

    if st.button("Réinitialiser"):
        if st.session_state.get("generator"):
            (generatorChoice, discriminatorChoice, lossChoice,
             epochs, modelName, learning_rate_g, learning_rate_d,
             batch_size) = st.session_state.choices
            saveModel(st.session_state.generator,
                      st.session_state.discriminator, modelName,
                      generatorChoice, discriminatorChoice, lossChoice,
                      edit=True)
        st.session_state.generator = None
        st.session_state.choices = None
        st.rerun()

    st.session_state.setdefault("choices", None)
    st.session_state.setdefault("generator", None)

    if st.session_state.choices is None:
        fill_form()

    if st.session_state.choices is not None:
        (generatorChoice, discriminatorChoice, lossChoice,
         epochs, modelName, learning_rate_g, learning_rate_d,
         batch_size) = st.session_state.choices

        st.write(f"**Générateur** : {generatorChoice}")
        st.write(f"**Discriminateur** : {discriminatorChoice}")
        st.write(f"**Fonction de perte** : {lossChoice}")
        st.write(f"**Époques** : {epochs}")
        st.write(f"**Learning rate G** : {learning_rate_g}")
        st.write(f"**Learning rate D** : {learning_rate_d}")
        st.write(f"**Batch size** : {batch_size}")
        st.write(f"**Nom du modèle** : {modelName}")

        if st.button("Lancer l'entraînement !"):
            st.info("Initialisation de l'entraînement ...")
            latent_dim = 128
            image_size = 64
            IMAGE_DIR = "./augmentedData"
            normalization_stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            dataset = img_utils.make_dataset(
                IMAGE_DIR, image_size, batch_size, normalization_stats
            )
            dataloader = DataLoader(dataset, batch_size, shuffle=True,
                                    num_workers=4, pin_memory=False)

            device = cuda_utils.get_training_device()
            WGAN = (lossChoice == 'Wasserstein')

            if generatorChoice == 'MLP':
                generator = GeneratorMLP()
            elif generatorChoice == 'CNN':
                generator = GeneratorCNN()
            elif generatorChoice == 'DCNN':
                generator = GeneratorDCNN()
            else:
                raise Exception("Générateur non implémenté")

            if discriminatorChoice == 'MLP':
                discriminator = DiscriminatorMLP_WGAN() if WGAN else DiscriminatorMLP()
            elif discriminatorChoice == 'CNN':
                discriminator = DiscriminatorCNN_WGAN() if WGAN else DiscriminatorCNN()
            else:
                raise Exception("Discriminateur non implémenté")

            generator = generator.to(device)
            discriminator = discriminator.to(device)

            optimizer_G = torch.optim.Adam(generator.parameters(),
                                           lr=learning_rate_g,
                                           betas=(0.5, 0.9))
            optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                           lr=learning_rate_d,
                                           betas=(0.5, 0.9))

            for epoch in range(epochs):
                for i, (real_imgs, _) in enumerate(stqdm(dataloader)):
                    real_imgs = real_imgs.to(device)
                    size = real_imgs.size(0)

                    if WGAN:
                        optimizer_D.zero_grad()
                        z = torch.randn(size, latent_dim, device=device)
                        gen_imgs = generator(z).detach()
                        loss_D = torch.mean(discriminator(
                            gen_imgs)) - torch.mean(discriminator(real_imgs))
                        loss_D.backward()
                        optimizer_D.step()
                        for p in discriminator.parameters():
                            p.data.clamp_(-0.03, 0.03)

                        if i % 6 == 0:
                            optimizer_G.zero_grad()
                            gen_imgs = generator(z)
                            loss_G = -torch.mean(discriminator(gen_imgs))
                            loss_G.backward()
                            optimizer_G.step()
                    else:
                        real_targets = torch.full((size, 1), 0.9,
                                                  device=device)
                        real_predictions = discriminator(real_imgs)
                        real_loss = F.binary_cross_entropy(real_predictions,
                                                           real_targets)

                        z = torch.randn(size, latent_dim, device=device)
                        gen_imgs = generator(z).detach()
                        gen_targets = torch.zeros(size, 1, device=device)
                        gen_predictions = discriminator(gen_imgs)
                        gen_loss = F.binary_cross_entropy(gen_predictions,
                                                          gen_targets)
                        loss_D = real_loss + gen_loss

                        optimizer_D.zero_grad()
                        loss_D.backward()
                        optimizer_D.step()

                        optimizer_G.zero_grad()
                        z = torch.randn(size, latent_dim, device=device)
                        gen_imgs = generator(z)
                        predictions = discriminator(gen_imgs)
                        targets = torch.ones(size, 1, device=device)
                        loss_G = F.binary_cross_entropy(predictions,
                                                        targets)
                        loss_G.backward()
                        optimizer_G.step()

                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(), max_norm=0.1)
                        torch.nn.utils.clip_grad_norm_(
                            generator.parameters(), max_norm=0.1)

                st.write(
                    f"[Epoch {
                        epoch + 1}/{epochs}] [D loss: {
                        loss_D.item():.4f}] [G loss: {
                        loss_G.item():.4f}]")
                st.session_state.generator = generator
                st.session_state.discriminator = discriminator

                if epoch % 5 == 0:
                    save_image(
                        img_utils.denorm(gen_imgs, normalization_stats),
                        f"results/{epoch}.png", nrow=8
                    )
                    st.image([
                        to_pil_image(img_utils.denorm(gen_imgs,
                                                      normalization_stats)[i])
                        for i in range(min(batch_size, 10))
                    ], width=100, use_container_width=False)

            saveModel(generator, discriminator, modelName,
                      generatorChoice, discriminatorChoice, lossChoice,
                      edit=True)
            st.success(
                f"Entraînement terminé et modèle **{modelName}** sauvegardé !")


if __name__ == "__main__":
    pass
