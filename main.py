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

if "choices" not in st.session_state:
    st.session_state.choices = None


def fill_form():
    with st.form(key='select_model'):
        generatorChoice = st.selectbox(
            'Veuillez sélectionner un type de générateur :', ('MLP', 'CNN'))
        st.write('Vous avez sélectionné: ', generatorChoice)
        discriminatorChoice = st.selectbox(
            'Veuillez sélectionner un type de discriminant :', ('MLP', 'CNN'))
        st.write('Vous avez sélectionné: ', discriminatorChoice)
        lossChoice = st.selectbox(
            'Veuillez sélectionner le type de loss :', ('Cross-entropy', 'Wasserstein'))
        st.write('Vous avez sélectionné: ', lossChoice)
        modelName = st.text_input("Nom du modèle: ")
        epochs = st.number_input(label="Nombre d'epochs",value=100)
        submit_button = st.form_submit_button(label='Generate pokemons !')
    if (submit_button or st.session_state.choices != None):
        return [generatorChoice, discriminatorChoice, lossChoice, epochs, modelName]
    return None

def saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice,edit=False):
    torch.save(generator.state_dict(),"./trainedModels/generators/"+modelName)
    torch.save(discriminator.state_dict(),"./trainedModels/discriminators/"+modelName)
    if edit :
        with open('./trainedModels/architecture-model.pkl', 'rb') as f:
            architecture_model = pickle.load(f)
        architecture_model[modelName] = [generatorChoice, discriminatorChoice]
        with open('./trainedModels/architecture-model.pkl', 'wb') as f:
            pickle.dump(architecture_model, f)
    
def entrainement():
    if st.session_state.choices == None:
        st.session_state.choices= fill_form()
        st.rerun()
    else:
        generatorChoice,lossChoice, discriminatorChoice, epochs, modelName = st.session_state.choices
        st.write("Générateur: " + generatorChoice)
        st.write("Discriminateur: " + discriminatorChoice)
        st.write("Nom du modèle: "+ modelName)
        # Initialize generator and discriminator
        if generatorChoice == 'MLP':
            generator = GeneratorMLP()
        elif generatorChoice == 'CNN':
            generator = GeneratorCNN()
        else:
            raise Exception("Générateur non implémenté")
        discriminatorChoice = 'CNN'
        if discriminatorChoice == 'MLP':
            discriminator = DiscriminatorMLP()
        elif discriminatorChoice == 'CNN':
            discriminator = DiscriminatorCNN()
        else:
            raise Exception("Discriminateur non implémenté")
        if device.type == 'cuda':
            generator.cuda()
            discriminator.cuda()
            
        if st.button("Save Model & Reset"):
            saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice,edit=True)
            st.session_state.choices = None
            st.rerun()
        lr = 0.001
        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr / 2)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        # ----------
        #  Training
        # ----------
        latent_dim = 128
        batches_done = 0
        for epoch in range(epochs):
            i = 0
            for real_imgs, _ in stqdm(dev_dataloader):
                if lossChoice == "Wasserstein" :
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()
                    # Sample noise as generator input
                    z = torch.randn(batch_size, latent_dim, device=device)
                    # Generate a batch of images
                    fake_imgs = generator(z).detach()
                    # Adversarial loss
                    loss_D = (torch.mean(discriminator(fake_imgs))
                              -torch.mean(discriminator(real_imgs)))
                    loss_D.backward()
                    optimizer_D.step()
                    # Clip weights of discriminator
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.1, 0.1)
                    # Train the generator every n_critic iterations
                    if i % 2 == 0:
                        # -----------------
                        #  Train Generator
                        # -----------------
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
                    fake_imgs = generator(z).detach()
                    real_predictions = discriminator(real_imgs)
                    gen_predictions = discriminator(fake_imgs)
                    st.write(real_predictions)
                    real_targets = torch.rand(real_imgs.size(0), 1, device=device) * 0.1  # Noisy labels
                    st.write(real_targets)
                    gen_targets = torch.rand(fake_imgs.size(0), 1, device=device) * 0.1 + 0.9
                    real_loss = F.binary_cross_entropy(real_predictions, real_targets)
                    gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
                    loss_D = real_loss + gen_loss
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    #Générateur
                    fake_imgs = generator(z).detach()
                    predictions = discriminator(fake_imgs)
                    targets = torch.zeros(fake_imgs.size(0), 1, device=device) 
                    loss_G = F.binary_cross_entropy(predictions, targets)
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
            st.write("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, epochs, loss_D.item(), loss_G.item()))
            if epoch % 10 == 1:
                save_image(img_utils.denorm(gen_imgs),
                           "resultsCNN/%d.png" % (epoch + 1), nrow=8)
        saveModel(generator, discriminator, modelName, generatorChoice, discriminatorChoice,edit=True)
        st.session_state.choices = None
        st.rerun()

def main():
    tab1, tab2 = st.tabs(["Entraînement", "Génération"])

    with tab1:
        entrainement()
    with tab2:
        st.write("Work in progress")

main()