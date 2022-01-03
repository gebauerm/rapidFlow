import torch.nn as nn
import torch


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 32),
            nn.ReLU(True), nn.Linear(32, 12))
        self.decoder = nn.Sequential(
            nn.ReLU(True), nn.Linear(12, 32),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        z = self.encoder(x)
        noise = torch.normal(mean=torch.zeros(z.shape), std=torch.ones(z.shape)).cuda()
        z = z + noise
        x = self.decoder(z)
        return x, z


def loss_function(x, img, target, criterion):
    idx = target == 7
    return criterion(x[idx], img[idx])
