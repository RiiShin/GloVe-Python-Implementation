import torch.nn as nn
import torch


class GloVe(nn.Module):

    def __init__(self, v_size: int, e_dim: int, xmax: int, alpha: float):
        super().__init__()
        self.w = nn.Embedding(
            num_embeddings=v_size, 
            embedding_dim=e_dim, sparse = True
        )

        self.w_ = nn.Embedding(
            num_embeddings=v_size, 
            embedding_dim=e_dim, sparse = True
        )

        self.b = nn.Parameter(
            torch.randn(v_size, dtype=torch.float)
        )

        self.b_ = nn.Parameter(
            torch.randn(v_size, dtype=torch.float)
        )
        self.xmax = xmax
        self.alpha = alpha

    def forward(self, i, j, xij):
        loss = torch.sum(torch.mul(self.w(i), self.w_(j)), dim=1)
        loss = torch.square((loss + self.b[i] + self.b_[j] - torch.log(xij)))
        cooc_func = torch.clamp((xij / self.xmax).pow(self.alpha), max = 1)
        loss = torch.mean(torch.mul(cooc_func, loss))
        return loss
    
    
