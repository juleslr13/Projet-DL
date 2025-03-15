# models/__init__.py
# appel des fonctions de générateur et de discriminateur

from .generators import GeneratorMLP, GeneratorCNN, GeneratorDCNN
from .discriminators import (
    DiscriminatorMLP, 
    DiscriminatorCNN, 
    DiscriminatorMLP_WGAN,  
    DiscriminatorCNN_WGAN   
)
