# L'opérateur intégral
Functional analysis foundations for kernel operators — spectral theory and perturbation bounds.

## Integral operator T_k
For a kernel k: ℝ^d × ℝ^d → ℝ and measure μ on ℝ^d, the integral operator T_k: L²(μ) → L²(μ) is:
```
(T_k f)(x) = ∫ k(x,y) f(y) dμ(y)
```
Properties (under square-integrability of k):
- Linear and continuous
- Self-adjoint when k is symmetric: k(x,y) = k(y,x)
- Compact when k ∈ L²(μ × μ)
- Positive semi-definite when k is a PSD kernel

## Spectral theorem for compact self-adjoint operators
T_k has a discrete spectrum {λ_m}_{m≥0} with λ_m → 0, and an orthonormal eigenbasis {e_m} in L²(μ):
```
T_k e_m = λ_m e_m,    k(x,y) = Σ_m λ_m e_m(x) e_m(y)
```
The eigenvalues encode the "energy" in each direction of the RKHS. Fast decay means the kernel is low-rank — well approximated by few features.

## Hilbert-Schmidt operators
T_k is Hilbert-Schmidt iff:
```
‖T_k‖²_HS = ∫∫ k(x,y)² dμ(x) dμ(y) < ∞
```
This holds for the softmax kernel on bounded domains. The HS norm bounds the sum of squared eigenvalues: Σ_m λ_m² = ‖T_k‖²_HS.

## Hoffman-Wielandt inequality (extended to infinite dimensions)
For two Hilbert-Schmidt operators A, B with eigenvalue sequences {α_i}, {β_i}, there exists a permutation π such that:
```
(Σ_i |α_{π(i)} - β_i|²)^(1/2) ≤ ‖A - B‖_HS
```
**Application to attention approximation:** If Â = T_{k̂} is the approximate kernel operator, then:
```
spectral error ≤ ‖T_k - T_{k̂}‖_HS = ‖k - k̂‖_{L²(μ×μ)}
```
This quantifies how closely the performer's approximate attention matrix preserves the spectral structure of exact softmax attention. Smaller r (fewer random features) → larger HS distance → larger spectral distortion.

## Connection to the spectrum notebook
`Notebook - spectre approximations (2).ipynb` numerically computes eigenvalue spectra of K_exact and K_approx matrices on random data under various normalization conventions. The Hoffman-Wielandt bound provides the theoretical justification for why matching spectra implies matching kernel functions.

## Key insight for this project
The next step (context.md, item 2) is to compute these spectral quantities on actual Q/K activations from TinyLlama, rather than synthetic data, to validate that r=256 features achieves adequate spectral fidelity in practice.
