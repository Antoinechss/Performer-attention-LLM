import torch
from torch import nn
import math


class PerformerAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, num_features):
        super().__init__()

        self.dim = dim  # Size of token embeddings
        self.num_heads = num_heads
        self.head_dim = head_dim  # Dimension of each query/key/value vector within one head
        self.num_features = num_features  # nb of features used for approximation

        # total dimension of all attention heads combined
        inner_dim = num_heads * head_dim

        # projections
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)  # Q = XWq
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)  # K = XWk
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)  # V = XWv

    def sample_features_gaussian(self):
        """
        First baseline for feature sampling
        omega.shape = [m, d] = [num_features, head_dim]
        """
        omega = torch.randn(self.nb_features, self.head_dim)
        self.register_buffer("omega", omega)

    def sample_featurs_ORF(self):
        """Orthogonal Random Feature implementation (FAVOR+)"""
        pass

    def phi(self, x):
        # Project x onto approximation space : compute wi^T * x for i in [1, m]
        # for every batch b, head h and token n
        proj_x = torch.einsum("bhnd, md -> bhnm", x, self.omega)
        # Square every coordinate and sum along d dimension
        # keepdim=True to conserve dimension for substraction against om_x
        norm_x = 0.5 * (x**2).sum(dim=-1, keepdim=True)
        phi = torch.exp(proj_x - norm_x) / math.sqrt(self.nb_features)
        return phi
