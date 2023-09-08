import torch
import torch.nn.functional as F
import torch.nn as nn

class VariationaAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        #encoder
        self.img2hid = nn.Linear(input_dim, h_dim)
        self.hid_2_mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        #decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img2hid(x))
        mu, sigma = self.hid_2_mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_reparameterized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    vae = VariationaAutoEncoder(input_dim=784)
    x_constructed, mu, sigma = vae(x)
    print(x_constructed.shape)
    print(mu.shape)
    print(sigma.shape)