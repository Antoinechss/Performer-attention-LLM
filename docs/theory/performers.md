# Rethinking Attention with Performers
Choromanski et al., ICLR 2021

## Problem
Standard softmax attention is O(L²d) in time and space (L = sequence length, d = head dimension). Goal: approximate it in O(Lrd) where r << L.

## Core idea: FAVOR+
Replace the attention matrix A = softmax(QK^T/√d) with a low-rank factorization:
```
A ≈ Q'(K')^T    where Q' = φ(Q), K' = φ(K)
```
φ is a random feature map such that K(x,y) = E[φ(x)^T φ(y)] ≈ exp(x^T y).
With this factorization, attention output is computed as (Q'((K')^T V)) in O(Lrd) — no N×N matrix ever materialized.

## Positive Random Features (PRF)
The trigonometric feature map (cos/sin) produces unbiased estimates but with high variance and can give negative values, breaking the attention normalizer.

FAVOR+ uses positive feature maps:

**SM+ (used in this codebase):**
```
φ+(x) = (exp(-‖x‖²/2) / √D) · [exp(ω₁ᵀx), ..., exp(ω_Dᵀx)]
```

**SM_hyp+ (lower variance alternative):**
```
φ_hyp+(x) = (exp(-‖x‖²/2) / √(2D)) · [exp(ω₁ᵀx), exp(-ω₁ᵀx), ...]
```
Both are unbiased: E[φ(x)^T φ(y)] = exp(x^T y). The hyperbolic variant halves variance.

## Orthogonal Random Features (ORF)
Sample ω vectors as orthogonal blocks (Gram-Schmidt QR decomposition with chi(d) norm scaling) instead of i.i.d. Gaussian. This:
- Reduces variance by avoiding redundant random directions
- Enables exponentially small approximation error bounds (first such result for softmax)
- `_sample_orf` in `performer_attention.py` implements this

## Causal (autoregressive) attention
For causal attention, use prefix-sum formulation:
```
out[i] = (Σ_{j≤i} φ(k_j)^T v_j) applied to φ(q_i)
       = φ(q_i) · S_i / (φ(q_i) · z_i)
```
where S_i, z_i are running cumulative sums — enables O(LrD) sequential scan.

## Complexity summary
| Method | Time | Space |
|--------|------|-------|
| Softmax attention | O(L²d) | O(L²) |
| FAVOR+ (non-causal) | O(Lrd) | O(Lr + Ld) |
| FAVOR+ (causal scan) | O(Lrd) | O(rd) |

## Key parameters
- r = number of random features (M in codebase). More features = lower approximation error, higher cost. r=256 used here.
- ORF block size = head_dim (d). Multiple blocks stacked to reach r if r > d.
