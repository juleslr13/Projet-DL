import torch
import torch.nn as nn
import torch.nn.utils as utils


class GeneratorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim=128
        img_shape=(3,64,64)
        self.img_shape=img_shape
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.2))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0],*self.img_shape)
        return img


class GeneratorCNN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.init_size = 64 // 4  # Downsampled size
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.model(out)
        return img