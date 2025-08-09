import torch
import torch.nn as nn


class GMM(nn.Module):
    def __init__(self, num_centroid, latent_dim):
        super(GMM, self).__init__()
        self.num_centroid = num_centroid

        self.c = nn.Parameter(torch.Tensor(num_centroid,))
        self.mu = nn.Parameter(torch.Tensor(num_centroid, latent_dim))
        self.sigma = nn.Parameter(torch.Tensor(num_centroid, latent_dim))

    def forward(self,x):
        #(batch_size,d)
