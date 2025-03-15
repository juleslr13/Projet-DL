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
import numpy as np
import pickle
from torchvision.transforms.functional import to_pil_image
# Custom imports
from models import GeneratorMLP, GeneratorCNN, GeneratorDCNN, DiscriminatorMLP, DiscriminatorCNN, DiscriminatorMLP_WGAN, DiscriminatorCNN_WGAN
import utils.cuda_utils as cuda_utils
import utils.img_utils as img_utils


latent_dim = 128

# Création du dataset
IMAGE_DIR = "./augmentedData"
image_size = 64
batch_size = 32
normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
dataset = img_utils.make_dataset(IMAGE_DIR,
                                 image_size,
                                 batch_size, normalization_stats)

# Configure data loader
dataloader = DataLoader(dataset,
                        batch_size, shuffle=True,
                        num_workers=4, pin_memory=False)

# Mettre sur la device
device = cuda_utils.get_training_device()
dev_dataloader = cuda_utils.DeviceDataLoader(dataloader, device)

    
# Streamlit
st.title("Application sur les GANs")

if device.type=="cuda":
    st.write("cuda detected !")
else:
    st.write("cuda not detected :(")

#Initialisation des session_states
if "choices" not in st.session_state:
    st.session_state.choices = None

if "generator" not in st.session_state:
    st.session_state.generator = None

if "discriminator" not in st.session_state:
    st.session_state.discriminator = None


# Formulaire pour l'entraînement
def fill_form():
    """
    Affiche un formulaire Streamlit pour sélectionner les hyperparamètres du modèle.

    Returns:
        list: Liste des choix de l'utilisateur [générateur, discriminateur, loss, epochs, modelName].
    """
    with st.form(key='select_model'):
        generatorChoice = st.selectbox(
            'Veuillez sélectionner un type de générateur :', ('MLP', 'CNN', 'DCNN'))
        st.write('Vous avez sélectionné: ', generatorChoice)
        discriminatorChoice = st.selectbox(
            'Veuillez sélectionner un type de discriminant :', ('MLP', 'CNN'))
        st.write('Vous avez sélectionné: ', discriminatorChoice)
        lossChoice = st.selectbox(
            'Veuillez sélectionner le type de loss :', ('Cross-entropy', 'Wasserstein'))
        st.write('Vous avez sélectionné: ', lossChoice)
        # Ajout d'une indication dans le placeholder
        modelName = st.text_input("Nom du modèle (please fill with no spaces or special characters): ")
        epochs = st.number_input(label="Nombre d'epochs", value=100)
        submit_button = st.form_submit_button(label='Valider')
    
    if submit_button or st.session_state.get("choices") is not None:
        # Vérification : nom non vide
        if not modelName:
            st.error("Erreur : Le nom du modèle ne peut pas être vide. Veuillez le remplir.")
            return
        # Vérification : pas d'espaces
        if " " in modelName:
            st.error("Erreur : Le nom du modèle ne doit pas contenir d'espaces. Veuillez le remplir sans espaces.")
            return
        # Vérification : autoriser uniquement lettres, chiffres et underscore
        allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
        for char in modelName:
            if char not in allowed_chars:
                st.error("Erreur : Le nom du modèle doit contenir uniquement des lettres, chiffres et underscores.")
                return

        st.session_state.choices = [generatorChoice, discriminatorChoice, lossChoice, epochs, modelName]
    return None

# Formulaire pour la génération
def generation_form(listModels):
    with st.form(key='select_generation'):
        modelName = st.selectbox(
            'Veuillez sélectionner le modèle :', listModels)
        nbPokemons = st.number_input(label="Nombre de Pokémons", value = 64)
        submit_button = st.form_submit_button(label='Generate Pokemons !')
    if (submit_button):
        return [modelName, nbPokemons]

# Sauvegarde des modèles (avec possibilité d'édition
def saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice, lossChoice, edit=False):
    torch.save(generator.state_dict(),"./trainedModels/generators/"+modelName+'.pth')
    torch.save(discriminator.state_dict(),"./trainedModels/discriminators/"+modelName+'.pth')
    if edit :
        with open('./trainedModels/architecture-model.pkl', 'rb') as f:
            architecture_model = pickle.load(f)
        # architecture_model stocke les choix d'architectures de chaque modèle entrainé
        architecture_model[modelName] = [generatorChoice, discriminatorChoice, lossChoice]
        with open('./trainedModels/architecture-model.pkl', 'wb') as f:
            pickle.dump(architecture_model, f)



# Fenêtre streamlit de génération de pokémons 

from torchvision.transforms.functional import to_pil_image

def generation():
    st.button("Actualiser les modèles")
    with open('./trainedModels/architecture-model.pkl', 'rb') as f:
            architecture_model = pickle.load(f)
    
    listModels = list(architecture_model.keys())
    form = None
    form = generation_form(listModels)
    if form != None :
        modelName, nbPokemons = form
        generatorChoice, discriminatorChoice, lossChoice = architecture_model[modelName]
        st.write("Générateur: " + generatorChoice)
        st.write("Discriminateur: " + discriminatorChoice)
        st.write("Nom du modèle: " + modelName)
        WGAN = (lossChoice =='Wasserstein')
        # Initialize generator and discriminator
        if generatorChoice == 'MLP':
            generator = GeneratorMLP()
        elif generatorChoice == 'CNN':
            generator = GeneratorCNN()
        elif generatorChoice == 'DCNN':
            generator = GeneratorDCNN()
        else:
            raise Exception("Générateur non implémenté")
        if discriminatorChoice == 'MLP':
            if WGAN:
                discriminator = DiscriminatorMLP_WGAN()
            else:
                discriminator = DiscriminatorMLP()
        elif discriminatorChoice == 'CNN':
            if WGAN:
                discriminator = DiscriminatorCNN_WGAN()
            else:
                discriminator = DiscriminatorCNN()
        else:
            raise Exception("Discriminateur non implémenté")
        if device.type == 'cuda':
            generator.cuda()
            discriminator.cuda()
            
        generator.load_state_dict(torch.load('./trainedModels/generators/'+modelName+'.pth', weights_only=False))
        discriminator.load_state_dict(torch.load('./trainedModels/discriminators/'+modelName+'.pth', weights_only=False))
        generator.eval()
        discriminator.eval()
        z = torch.randn(nbPokemons, latent_dim, device=device)
        gen_imgs = generator(z).cpu()
        img_utils.show_images(img_utils.denorm(gen_imgs),nbPokemons, nrow=int(np.sqrt(nbPokemons)))
        
    
def entrainement():
    # Bouton "Réinitialiser" pour revenir au formulaire vide
    if st.button("Réinitialiser"):
        st.session_state.choices = None
        st.session_state.start_training = False
        st.write("Formulaire réinitialisé")
        st.rerun()
    
    # Si le formulaire n'est pas encore rempli, on l'affiche
    if st.session_state.get("choices") is None:
        fill_form()
        return
    else:
        # Une fois le formulaire rempli, on affiche un bouton pour démarrer l'entraînement
        if "start_training" not in st.session_state:
            st.session_state.start_training = False
        if not st.session_state.start_training:
            if st.button("Train Pokemons !"):
                st.session_state.start_training = True
            return  # Attend que l'utilisateur clique sur "Démarrer l'entraînement"
        
        # Récupération des paramètres du formulaire
        generatorChoice, discriminatorChoice, lossChoice, epochs, modelName = st.session_state.choices
        st.write("Générateur: " + generatorChoice)
        st.write("Discriminateur: " + discriminatorChoice)
        st.write("Nom du modèle: " + modelName)
        st.write("Fonction de perte: " + lossChoice)
        WGAN = (lossChoice == 'Wasserstein')
        
        # Initialisation des modèles
        if generatorChoice == 'MLP':
            generator = GeneratorMLP()
        elif generatorChoice == 'CNN':
            generator = GeneratorCNN()
        elif generatorChoice == 'DCNN':
            generator = GeneratorDCNN()
        else:
            raise Exception("Générateur non implémenté")
        
        if discriminatorChoice == 'MLP':
            if WGAN:
                discriminator = DiscriminatorMLP_WGAN()
            else:
                discriminator = DiscriminatorMLP()
        elif discriminatorChoice == 'CNN':
            if WGAN:
                discriminator = DiscriminatorCNN_WGAN()
            else:
                discriminator = DiscriminatorCNN()
        else:
            raise Exception("Discriminateur non implémenté")
        
        if device.type == 'cuda':
            generator.cuda()
            discriminator.cuda()
        
        lr = 0.001
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr / 2)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        final_gen_imgs = None
        
        st.write("Début de l'entraînement...")
        for epoch in range(epochs):
            st.write(WGAN)
            i = 0
            for real_imgs, _ in stqdm(dev_dataloader):
                if lossChoice == 'Wasserstein':
                    optimizer_D.zero_grad()
                    z = torch.randn(batch_size, latent_dim, device=device)
                    gen_imgs = generator(z).detach()
                    loss_D = torch.mean(discriminator(gen_imgs)) - torch.mean(discriminator(real_imgs))
                    loss_D.backward()
                    optimizer_D.step()
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.1, 0.1)
                    if i % 2 == 0:
                        optimizer_G.zero_grad()
                        gen_imgs = generator(z)
                        loss_G = -torch.mean(discriminator(gen_imgs))
                        loss_G.backward()
                        optimizer_G.step()
                    i += 1
                else:
                    z = torch.randn(batch_size, latent_dim, device=device)
                    gen_imgs = generator(z).detach()
                    real_predictions = discriminator(real_imgs)
                    gen_predictions = discriminator(gen_imgs)
                    real_targets = torch.rand(real_imgs.size(0), 1, device=device) * 0.1
                    gen_targets = torch.rand(gen_imgs.size(0), 1, device=device) * 0.1 + 0.9
                    real_loss = F.binary_cross_entropy(real_predictions, real_targets)
                    gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
                    loss_D = real_loss + gen_loss
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    
                    gen_imgs = generator(z).detach()
                    predictions = discriminator(gen_imgs)
                    targets = torch.zeros(gen_imgs.size(0), 1, device=device)
                    loss_G = F.binary_cross_entropy(predictions, targets)
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
            
            st.write("[Epoch %d/%d] [D loss: %f] [G loss: %f]" %
                     (epoch+1, epochs, loss_D.item(), loss_G.item()))
            st.session_state.generator = generator
            st.session_state.discriminator = discriminator
            
            # Sauvegarde et affichage d'un échantillon toutes les 5 époques
            if epoch % 5 == 1:
                save_image(img_utils.denorm(gen_imgs),
                           "resultsCNN/%d.png" % (epoch - 1), nrow=8)
                st.image(to_pil_image(img_utils.denorm(gen_imgs)[-1]), caption=f"Échantillon à l'époque {epoch}", use_container_width=True)
            
            final_gen_imgs = gen_imgs
        
        # Affichage de l'image finale (conversion en PIL pour Streamlit)
        
        st.subheader("Image générée finale")
        if final_gen_imgs is not None:
            final_img = to_pil_image(img_utils.denorm(final_gen_imgs[0].cpu()))
            st.image(final_img, caption=f"Époque finale {epochs}", use_container_width=True)
        else:
            st.write("Aucune image générée.")
        
        # Sauvegarde finale du modèle
        saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice, lossChoice, edit=True)
        st.success("Entraînement terminé. Modèle sauvegardé!")


def main():
    """
    Fonction principale gérant l'interface Streamlit.
    """
    tab1, tab2 = st.tabs(["Entraînement", "Génération"])

    with tab1:
        entrainement()
    with tab2:
        generation()


if __name__ == "__main__":
    main()