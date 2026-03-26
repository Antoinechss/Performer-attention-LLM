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

        self.sample_features_gaussian() # Sample omega and store it in class variables

        # total dimension of all attention heads combined
        inner_dim = num_heads * head_dim

        # projections
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)  # Q = XWq
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)  # K = XWk
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)  # V = XWv
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim) # original dimensions [B, N, H*D]


    def sample_features_gaussian(self):
        """
        First baseline for feature sampling
        omega.shape = [m, d] = [num_features, head_dim]
        """
        omega = torch.randn(self.num_features, self.head_dim)
        self.register_buffer("omega", omega) # adds omega to set of class variables

    def sample_features_ORF(self):
        """Orthogonal Random Feature implementation (FAVOR+)"""
        pass

    def phi(self, x):
        # Project x onto approximation space : compute wi^T * x for i in [1, m]
        # for every batch b, head h and token n
        proj_x = torch.einsum("bhnd, md -> bhnm", x, self.omega)
        # Square every coordinate and sum along d dimension
        # keepdim=True to conserve dimension for substraction against om_x
        norm_x = 0.5 * (x**2).sum(dim=-1, keepdim=True)
        phi = torch.exp(proj_x - norm_x) / math.sqrt(self.num_features)
        return phi
    
    def reshape_heads(self, x): 
        """ Reshapes projected vector from [B, N, H*D] to [B, H, N, D]"""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  
    
    def forward(self, x): 
        B, N, _ = x.shape

        # Compute key, query and value matrices: shape [B, N, H*D]
        q = self.q_proj(x) 
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to individual heads for multi head attention [B, H, N, D]
        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)

        # Apply feature map phi, [B, H, N, M] with M << D
        phi_q = self.phi(q)
        phi_k = self.phi(k)

        # Compute once the global kv, [B, H, M, D]
        kv = torch.einsum("bhnm,bhnd->bhmd", phi_k, v)

        # Normalization term (summing on all tokens)
        k_sum = phi_k.sum(dim=2) # [B, H, M]
        z = 1 / (torch.einsum("bhnm,bhm->bhn", phi_q, k_sum) + 1e-6) # [B, H, N]

        # final attention output, [B, H, N, D]
        out = torch.einsum("bhnm,bhmd,bhn->bhnd", phi_q, kv, z) 

        # merge heads back, [B, N, H*D]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        # output projection
        out = self.out_proj(out)

        return out
    
    
# --------------- TEST -------------------

model = PerformerAttention(dim=64, num_heads=4, head_dim=16, num_features=32)
x = torch.randn(2, 10, 64)  # [B, N, dim]
out = model(x)
print(out.shape)





 

