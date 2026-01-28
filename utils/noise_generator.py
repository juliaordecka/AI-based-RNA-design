import torch
import numpy as np

def generate_noise(latent_dim, sample_size):
    noise = np.random.normal(0, 1, (sample_size, latent_dim))
    noise = torch.tensor(noise).float()
    return noise



