import torch
from torch import nn
import math
import importlib.util as _ilu
import os as _os

# Load Triton kernels if available (CUDA only)
try:
    _ts_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'triton_scan.py')
    _ts_spec = _ilu.spec_from_file_location('performer_triton_scan', _ts_path)
    _ts_mod  = _ilu.module_from_spec(_ts_spec)
    _ts_spec.loader.exec_module(_ts_mod)
    _triton_scan = _ts_mod.triton_scan_forward
    _HAS_TRITON  = _ts_mod._TRITON_AVAILABLE
except Exception:
    _HAS_TRITON  = False
    _triton_scan = None


def _sample_orf(head_dim, num_features, device=None):
    """Sample orthogonal random features with chi(d) norm scaling (FAVOR+)."""
    if device is None:
        device = torch.device("cpu")
    blocks = []
    while len(blocks) * head_dim < num_features:
        G = torch.randn(head_dim, head_dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(G)
        norms = torch.randn(head_dim, head_dim, device=device).norm(dim=1)
        blocks.append(torch.diag(norms) @ Q.T)
    return torch.cat(blocks, dim=0)[:num_features]


def _normalize_feature_map_name(feature_map):
    key = feature_map.lower().strip().replace(" ", "").replace("-", "_")
    aliases = {
        "favor+": "favor_plus",
        "favorplus": "favor_plus",
        "favor_plus": "favor_plus",
        "favor#": "favor_sharp",
        "favorsharp": "favor_sharp",
        "favor_sharp": "favor_sharp",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported feature map '{feature_map}'.")
    return aliases[key]


def _optimal_gerf_a(varphi):
    """Closed-form minimizer from Lemma A.1 in the FAVOR# paper."""
    varphi = torch.clamp_min(varphi, 0.0)
    disc = (2 * varphi + 1).square() + 8 * varphi
    return 0.0625 * (1 - 2 * varphi - torch.sqrt(disc))


def _compute_favor_sharp_context(q, k, eps=1e-6):
    """Estimate SADERF/GERF parameters from query-key statistics."""
    q32 = q.float()
    k32 = k.float()

    q_energy = q32.square().sum(dim=2)
    k_energy = k32.square().sum(dim=2)
    psi = ((k_energy + eps) / (q_energy + eps)).pow(0.25)

    q_t = q32 * psi.unsqueeze(2)
    k_t = k32 / psi.unsqueeze(2)

    mean_pair_norm = q_t.square().sum(dim=-1).mean(dim=2)
    mean_pair_norm = mean_pair_norm + k_t.square().sum(dim=-1).mean(dim=2)
    mean_pair_norm = mean_pair_norm + 2 * (q_t.mean(dim=2) * k_t.mean(dim=2)).sum(dim=-1)

    varphi = mean_pair_norm / q.shape[-1]
    a = _optimal_gerf_a(varphi)
    one_minus_4a = torch.clamp_min(1 - 4 * a, eps)

    return {
        "psi": psi,
        "b": torch.sqrt(one_minus_4a),
        "a": a,
        "log_d": 0.25 * q.shape[-1] * torch.log(one_minus_4a),
    }


def _phi_favor_plus(x, omega, num_features, is_query=True):
    """FAVOR+ feature map (matches Google's softmax_kernel_transformation).

    The max for numerical stability is taken on proj_x (= x @ omega)
    ALONE, not on proj_x - norm_x. This matches the Google reference.

    For queries:  max of proj_x over M dimension (per token, per head)
    For keys:     max of proj_x over M and N dimensions (per head)
    """
    omega = omega.to(device=x.device, dtype=x.dtype)
    ratio = 1.0 / math.sqrt(num_features)
    proj_x = torch.einsum("bhnd,md->bhnm", x, omega)      # [B, H, N, M]
    norm_x = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)      # [B, H, N, 1]
    if is_query:
        # Max of proj_x over M dimension (last dim), per token
        stabilizer = proj_x.max(dim=-1, keepdim=True).values      # [B, H, N, 1]
    else:
        # Max of proj_x over M and N dimensions, per (batch, head)
        stabilizer = proj_x.amax(dim=(-2, -1), keepdim=True)      # [B, H, 1, 1]
    return ratio * (torch.exp(proj_x - norm_x - stabilizer) + 1e-6)


def _phi_favor_sharp(x, omega, num_features, context, is_query=True):
    """FAVOR# feature map using the paper's practical SADERF construction."""
    x32 = x.float()
    omega32 = omega.to(device=x.device, dtype=torch.float32)
    omega_norm_sq = omega32.square().sum(dim=-1).view(1, 1, 1, -1)

    psi = context["psi"].to(device=x.device, dtype=torch.float32)
    x_t = x32 * psi.unsqueeze(2) if is_query else x32 / psi.unsqueeze(2)

    a = context["a"].to(device=x.device, dtype=torch.float32)[:, :, None, None]
    b = context["b"].to(device=x.device, dtype=torch.float32)[:, :, None, None]
    log_d = context["log_d"].to(device=x.device, dtype=torch.float32)[:, :, None, None]

    ratio = 1.0 / math.sqrt(num_features)
    proj_x = torch.einsum("bhnd,md->bhnm", x_t, omega32)
    proj_x = b * proj_x + a * omega_norm_sq
    norm_x = 0.5 * x_t.square().sum(dim=-1, keepdim=True)

    if is_query:
        stabilizer = proj_x.max(dim=-1, keepdim=True).values
    else:
        stabilizer = proj_x.amax(dim=(-2, -1), keepdim=True)

    log_phi = math.log(ratio) + log_d + proj_x - norm_x - stabilizer
    return torch.exp(log_phi) + 1e-6


def _phi(x, omega, num_features, is_query=True, feature_map="favor_plus", reference=None):
    feature_map = _normalize_feature_map_name(feature_map)
    if feature_map == "favor_plus":
        return _phi_favor_plus(x, omega, num_features, is_query=is_query)
    if reference is None:
        raise ValueError("FAVOR# requires a reference tensor to estimate query/key statistics.")
    context = _compute_favor_sharp_context(x, reference) if is_query else _compute_favor_sharp_context(reference, x)
    return _phi_favor_sharp(x, omega, num_features, context, is_query=is_query)


def _phi_pair(q, k, omega, num_features, feature_map):
    feature_map = _normalize_feature_map_name(feature_map)
    if feature_map == "favor_plus":
        return (
            _phi_favor_plus(q, omega, num_features, is_query=True),
            _phi_favor_plus(k, omega, num_features, is_query=False),
        )
    context = _compute_favor_sharp_context(q, k)
    return (
        _phi_favor_sharp(q, omega, num_features, context, is_query=True),
        _phi_favor_sharp(k, omega, num_features, context, is_query=False),
    )


def _python_scan(phi_q, phi_k, v):
    """Causal sequential scan — CPU/MPS fallback. O(M*D) memory."""
    B, H, N, M = phi_q.shape
    D = v.shape[-1]
    S = torch.zeros(B, H, M, D, dtype=phi_q.dtype, device=phi_q.device)
    z = torch.zeros(B, H, M,    dtype=phi_q.dtype, device=phi_q.device)
    out = torch.empty(B, H, N, D, dtype=phi_q.dtype, device=phi_q.device)
    for i in range(N):
        S = S + torch.einsum("bhm,bhd->bhmd", phi_k[:, :, i], v[:, :, i])
        z = z + phi_k[:, :, i]
        num   = torch.einsum("bhm,bhmd->bhd", phi_q[:, :, i], S)
        denom = (phi_q[:, :, i] * z).sum(-1, keepdim=True) + 1e-6
        out[:, :, i] = num / denom
    return out


class PerformerAttention(nn.Module):
    """Standalone Performer attention with FAVOR+ or FAVOR# feature maps."""

    def __init__(self, dim, num_heads, head_dim, num_features, feature_map="favor_plus"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_features = num_features
        self.feature_map = _normalize_feature_map_name(feature_map)

        self.register_buffer("omega", _sample_orf(head_dim, num_features))

        inner_dim = num_heads * head_dim
        self.q_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim)

    def phi(self, x, is_query=True, reference=None):
        return _phi(
            x,
            self.omega,
            self.num_features,
            is_query=is_query,
            feature_map=self.feature_map,
            reference=reference,
        )

    def feature_maps(self, q, k):
        return _phi_pair(q, k, self.omega, self.num_features, self.feature_map)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.25
        phi_q, phi_k = self.feature_maps(q * scale, k * scale)

        if self.feature_map == "favor_sharp":
            phi_q = phi_q.float()
            phi_k = phi_k.float()
            v_proj = v.float()
        else:
            v_proj = v

        kv_cumsum = torch.einsum("bhnm,bhnd->bhnmd", phi_k, v_proj).cumsum(dim=2)
        k_cumsum  = phi_k.cumsum(dim=2)
        out = torch.einsum("bhnm,bhnmd->bhnd", phi_q, kv_cumsum)
        z   = 1 / (torch.einsum("bhnm,bhnm->bhn", phi_q, k_cumsum) + 1e-6)
        out = out * z.unsqueeze(-1)
        out = out.to(x.dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class PerformerAttentionCore(nn.Module):
    """Core FAVOR+ / FAVOR# attention — no projections, plugs into any architecture."""

    def __init__(self, head_dim, num_features, feature_map="favor_plus"):
        super().__init__()
        self.head_dim = head_dim
        self.num_features = num_features
        self.feature_map = _normalize_feature_map_name(feature_map)
        self.register_buffer("omega", _sample_orf(head_dim, num_features), persistent=True)

    def phi(self, x, is_query=True, reference=None):
        return _phi(
            x,
            self.omega,
            self.num_features,
            is_query=is_query,
            feature_map=self.feature_map,
            reference=reference,
        )

    def feature_maps(self, q, k):
        return _phi_pair(q, k, self.omega, self.num_features, self.feature_map)

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.25
        phi_q, phi_k = self.feature_maps(q * scale, k * scale)

        if q.shape[2] == k.shape[2]:
            # Prefill: causal scan
            pq, pk, vf = phi_q.float(), phi_k.float(), v.float()
            use_triton = (
                self.feature_map == "favor_plus"
                and _HAS_TRITON
                and q.device.type == "cuda"
                and not torch.is_grad_enabled()
            )
            if use_triton:
                out = _triton_scan(pq, pk, vf)
            else:
                out = _python_scan(pq, pk, vf)
            out = out.to(q.dtype)
        else:
            # Decode: single new token against accumulated state
            if self.feature_map == "favor_plus":
                kv_sum = torch.einsum("bhnm,bhnd->bhmd", phi_k, v)
                k_sum  = phi_k.sum(dim=2)
                num    = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv_sum)
                denom  = torch.einsum("bhnm,bhm->bhn", phi_q, k_sum) + 1e-6
                out    = num / denom.unsqueeze(-1)
            else:
                pq, pk, vf = phi_q.float(), phi_k.float(), v.float()
                kv_sum = torch.einsum("bhnm,bhnd->bhmd", pk, vf)
                k_sum  = pk.sum(dim=2)
                num    = torch.einsum("bhnm,bhmd->bhnd", pq, kv_sum)
                denom  = torch.einsum("bhnm,bhm->bhn", pq, k_sum) + 1e-6
                out    = (num / denom.unsqueeze(-1)).to(q.dtype)

        return out
