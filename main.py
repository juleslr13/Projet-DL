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
            'Veuillez sélectionner un type de générateur :', ('MLP', 'CNN','DCNN'))
        st.write('Vous avez sélectionné: ', generatorChoice)
        discriminatorChoice = st.selectbox(
            'Veuillez sélectionner un type de discriminant :', ('MLP', 'CNN'))
        st.write('Vous avez sélectionné: ', discriminatorChoice)
        lossChoice = st.selectbox(
            'Veuillez sélectionner le type de loss :', ('Cross-entropy', 'Wasserstein'))
        st.write('Vous avez sélectionné: ', lossChoice)
        modelName = st.text_input("Nom du modèle: ")
        epochs = st.number_input(label="Nombre d'epochs",value=100)
        submit_button = st.form_submit_button(label='Train pokemons !')
    if (submit_button or st.session_state.choices != None):
        st.session_state.choices=[generatorChoice, discriminatorChoice, lossChoice, epochs, modelName]
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
    if st.session_state.choices == None:
        fill_form()
    else:
        generatorChoice, discriminatorChoice, lossChoice, epochs, modelName = st.session_state.choices
        st.write("Générateur: " + generatorChoice)
        st.write("Discriminateur: " + discriminatorChoice)
        st.write("Nom du modèle: "+ modelName)
        WGAN = (lossChoice == 'Wasserstein')
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
            
        if st.button("Save Model & Reset"):
            saveModel(st.session_state.generator, st.session_state.discriminator, modelName, generatorChoice, discriminatorChoice, lossChoice,edit=True)
            st.session_state.choices = None
            st.rerun()
        lr = 0.001
        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr / 2)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        # ----------
        #  Training
        # ----------
        batches_done = 0
        for epoch in range(epochs):
            i = 0
            for real_imgs, _ in stqdm(dev_dataloader):
                if lossChoice == 'Wasserstein' :
                    # Entraînement du discriminateur
                    optimizer_D.zero_grad()
                    # Vecteur latent aléatoire pour l'entrée
                    z = torch.randn(batch_size, latent_dim, device=device)
                    # Création d'une batch d'images
                    gen_imgs = generator(z).detach()
                    # Loss du discriminant
                    loss_D = (torch.mean(discriminator(gen_imgs))-torch.mean(discriminator(real_imgs)))
                    loss_D.backward()
                    optimizer_D.step()
                    # Weight clipping du discriminateur
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.1, 0.1)

                    if i % 2 == 0:
                        # Entraînement du générateur toutes les 2 itérations
                        optimizer_G.zero_grad()
                        
                        # Generate a batch of images
                        gen_imgs = generator(z)
                        # Adversarial loss
                        loss_G = -torch.mean(discriminator(gen_imgs))
                        loss_G.backward()
                        optimizer_G.step()
                    i += 1
                    batches_done += 1
                else :
                    # Cross-entropy bruitée
                    # Discriminateur
                    z = torch.randn(batch_size, latent_dim, device=device)
                    gen_imgs = generator(z).detach()
                    real_predictions = discriminator(real_imgs)
                    gen_predictions = discriminator(gen_imgs)
                    #st.write(real_predictions)
                    real_targets = torch.rand(real_imgs.size(0), 1, device=device) * 0.1  # Noisy labels
                    #st.write(real_targets)
                    gen_targets = torch.rand(gen_imgs.size(0), 1, device=device) * 0.1 + 0.9
                    real_loss = F.binary_cross_entropy(real_predictions, real_targets)
                    gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
                    loss_D = real_loss + gen_loss
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    #Générateur
                    gen_imgs = generator(z).detach()
                    predictions = discriminator(gen_imgs)
                    targets = torch.zeros(gen_imgs.size(0), 1, device=device) 
                    loss_G = F.binary_cross_entropy(predictions, targets)
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()

            st.write("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, epochs, loss_D.item(), loss_G.item()))
            # Sauvegarde temporaire des modèles après chaque epoch
            st.session_state.generator = generator
            st.session_state.discriminator = discriminator

            # Sauvegarde d'échantillons de génération toutes les 5 epochs
            if epoch % 5 == 1:
                save_image(img_utils.denorm(gen_imgs),
                           "resultsCNN/%d.png" % (epoch - 1), nrow=8)
        saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice, lossChoice, edit=True)
        st.session_state.choices = None
        st.rerun()

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