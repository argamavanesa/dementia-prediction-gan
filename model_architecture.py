import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Conditional DCGAN Generator.
    
    - Input: Noise z + label embedding
    - Architecture: ConvTranspose + BatchNorm + ReLU
    - Output: 128x128x1 via Tanh
    """
    def __init__(self, z_dim=100, num_classes=4, img_channels=1):
        super().__init__()

        # Label embedding (small, concatenated to z)
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, label_emb], dim=1)
        return self.net(x)
