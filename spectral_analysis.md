# Spectral Analysis — Softmax vs FAVOR+ Kernel Matrices

Kernel singular value spectra computed from real activations on TinyLlama 1.1B (teacher model).  
Input: "The quick brown fox jumps over the lazy dog." × 8, truncated to 512 tokens, averaged over 10 samples with varying lengths.  
Layers analysed: 0, 5, 10, 15, 21. Top-64 singular values kept per layer.

---

## 1. Top singular values

| Layer | σ₁ softmax | σ₁ FAVOR+ | Ratio σ₁ | Eff. rank softmax | Eff. rank FAVOR+ |
|-------|-----------|----------|----------|-------------------|-----------------|
| 0     | 1717.57   | 4.40 × 10⁻³ | 3.9 × 10⁵ | 2.05 | 1.10 |
| 5     | 55.73     | 5.65 × 10⁻⁵ | 9.9 × 10⁵ | 2.56 | 1.31 |
| 10    | 58.72     | 3.66 × 10⁻⁶ | 1.6 × 10⁷ | 4.37 | 1.05 |
| 15    | 223.68    | 2.40 × 10⁻⁶ | 9.3 × 10⁷ | 9.22 | 1.01 |
| 21    | 556.51    | 8.39 × 10⁻⁷ | 6.6 × 10⁸ | 6.76 | 1.25 |

**Effective rank** = exp(Shannon entropy of the normalised singular value distribution). A value near 1 means the spectrum is dominated by a single component; higher values indicate more spread.

---

## 2. Top-1 singular value coverage (σ₁ / Σσᵢ)

| Layer | Softmax | FAVOR+ |
|-------|---------|--------|
| 0     | 77.9%   | 98.5%  |
| 5     | 71.5%   | 95.6%  |
| 10    | 62.6%   | 99.3%  |
| 15    | 32.2%   | 99.9%  |
| 21    | 45.0%   | 96.2%  |

---

## 3. Spectrum shape per layer

### Layer 0 (early)
Softmax singular values: **1717.6, 360.1, 81.3, 30.8, 4.4, 3.7, 3.3, 1.8, 0.84, 0.47, …** (rapid decay, ~14 non-negligible values)  
FAVOR+ singular values: **4.4 × 10⁻³, 4.2 × 10⁻⁵, 9.7 × 10⁻⁶, 5.4 × 10⁻⁶, 4.7 × 10⁻⁶, …** (steep drop after rank 1)

### Layer 5 (lower-middle)
Softmax: **55.7, 15.9, 1.55, 1.16, 0.96, 0.75, 0.66, 0.31, …** (broader spread, effective rank 2.56)  
FAVOR+: **5.6 × 10⁻⁵, 8.6 × 10⁻⁷, 4.8 × 10⁻⁷, …** (rank-1 dominant at 95.6%)

### Layer 10 (middle)
Softmax: **58.7, 9.3, 6.9, 4.6, 3.6, 2.2, 1.8, 1.4, 1.2, 1.0, 0.95, 0.88, …** (densest spectrum, effective rank 4.37)  
FAVOR+: **3.7 × 10⁻⁶, 1.2 × 10⁻⁸, 3.9 × 10⁻⁹, …** (near-rank-1 at 99.3%)

### Layer 15 (upper-middle)
Softmax: **223.7, 118.1, 66.4, 48.1, 33.1, 31.0, 29.2, 25.1, 24.6, 23.7, …** (richest spectrum, effective rank 9.22 — all top-14 values within 3× of each other)  
FAVOR+: **2.4 × 10⁻⁶, 1.5 × 10⁻⁹, 5.6 × 10⁻¹⁰, …** (essentially rank-1 at 99.9%)

### Layer 21 (final)
Softmax: **556.5, 150.9, 131.9, 106.7, 58.4, 55.5, 40.9, 32.4, 30.7, 22.4, …** (dominant first component but rich tail, effective rank 6.76)  
FAVOR+: **8.4 × 10⁻⁷, 1.7 × 10⁻⁸, 6.6 × 10⁻⁹, …** (rank-1 dominant at 96.2%)

---

## 4. Analysis

### The rank gap is the core problem

The softmax kernel matrix has a rich, multi-rank structure — especially in layers 10 and 15, where the effective rank reaches 4–9 and the top-14 singular values span less than two orders of magnitude. FAVOR+ with 256 random features produces a **near-rank-1 approximation** in all layers (effective rank 1.0–1.3, top-1 coverage 95–99.9%). This means FAVOR+ is approximating a high-rank kernel with an essentially rank-1 matrix.

The ratio of leading singular values grows with depth: from ~4 × 10⁵ at layer 0 to ~7 × 10⁸ at layer 21. This reflects both the absolute scale difference (softmax kernel values grow with sequence length and temperature; random-feature kernel values are normalised differently) and a structural mismatch that worsens in later layers.

### Layer-depth pattern

The softmax spectrum becomes progressively richer through the middle layers (peak at layer 15, effective rank 9.22) before partially concentrating again in the final layer (21). This matches the standard observation that middle transformer layers perform the most distributed attention — attending to multiple context positions — while early and late layers are more peaked.

FAVOR+ does not track this depth-dependent richness. Its effective rank is uniformly near 1 across all layers, meaning the approximation is least faithful exactly where the softmax attention is most complex (layers 10–15).

### Why Phase 1 (4/32 heads) still works

With only 4 of 32 heads replaced by FAVOR+, the distillation loss can drive the model to route queries that need high-rank attention through the remaining 28 softmax heads. The 4 performer heads may specialise in lower-rank, position-insensitive patterns (e.g. local n-gram context) where the rank-1 approximation is adequate. The spectral gap explains why increasing beyond 4 heads degrades quality rapidly: there are not enough softmax heads left to absorb the high-rank attention demand from layers like 10 and 15.

### Implication for scaling M (number of random features)

The near-rank-1 collapse of FAVOR+ is partly an M=256 limitation. To approximate a kernel matrix with effective rank ~9 (layer 15), M needs to be large enough that φ(q)ᵀφ(k) spans a comparable rank subspace. With M=256 features and head_dim=64, the approximation has sufficient capacity in principle, but the random projection variance may still cause rank collapse in practice. Increasing M or using orthogonal random features (which reduce variance) would be the first lever to pull before increasing the number of performer heads.

---

## 5. Summary

| Finding | Implication |
|---------|-------------|
| Softmax effective rank peaks at 9.2 in layer 15 | Middle layers require the richest attention approximation |
| FAVOR+ effective rank is ≤ 1.3 everywhere | Current M=256 produces near-rank-1 kernels regardless of layer |
| σ₁ ratio grows from 4×10⁵ to 7×10⁸ with depth | Approximation fidelity degrades in later layers |
| 4/32 heads replaceable without quality loss | Performer heads can absorb low-rank attention patterns |
| Quality collapses beyond 8/32 heads | High-rank demand in layers 10–15 cannot be rerouted |
