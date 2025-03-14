import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import streamlit as st
global norm_stats 
norm_stats= None

def make_dataset(IMAGE_DIR,image_size,batch_size,normalization_stats):
    global norm_stats
    norm_stats = normalization_stats
    normal_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*normalization_stats)]))
    
    color_jitter_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ColorJitter(0, 0.2, 1),
        T.ToTensor(),
        T.Normalize(*normalization_stats)]))
    
    dataset_list = [normal_dataset, color_jitter_dataset]
    dataset = ConcatDataset(dataset_list)

    return dataset


def denorm(image):
    return image * norm_stats[1][0] + norm_stats[0][0]

def show_images(images, nmax=64, nrow=8):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=nrow).permute(1, 2, 0))
    st.pyplot(fig)
    
def show_batch(dataloader, nmax=64):
    for images, _ in dataloader:
        show_images(images, nmax)
        break