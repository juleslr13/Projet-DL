import torch
from torch.utils.data import DataLoader
import streamlit as st
from torchvision.utils import save_image
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


def start():
    with st.form(key='select_model'):
        generatorChoice = st.selectbox(
            'Veuillez sélectionner un type de générateur :', ('MLP', 'CNN'))
        st.write('Vous avez sélectionné :', generatorChoice)
        discriminatorChoice = st.selectbox(
            'Veuillez sélectionner un type de discriminant :', ('MLP', 'CNN'))
        st.write('Vous avez sélectionné :', discriminatorChoice)
        submit_button = st.form_submit_button(label='Submit')
    if (submit_button):
        return [generatorChoice, discriminatorChoice]
    return None


choice = None
choice = start()

if choice is not None:
    generatorChoice, discriminatorChoice = choice
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
    lr = 0.001
    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr / 2)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    # ----------
    #  Training
    # ----------
    epochs = 300
    latent_dim = 128
    batches_done = 0
    for epoch in range(epochs):
        i = 0
        for real_imgs, _ in dev_dataloader:
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = torch.randn(batch_size, latent_dim, device=device)
            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs))
            + torch.mean(discriminator(fake_imgs))
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
            if batches_done == len(dev_dataloader) - 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, epochs, batches_done % len(dataloader),
                       len(dataloader), loss_D.item(), loss_G.item())
                )
            batches_done += 1
        if epoch % 10 == 1:
            save_image(img_utils.denorm(gen_imgs),
                       "resultsCNN/%d.png" % epoch + 1, nrow=8)
