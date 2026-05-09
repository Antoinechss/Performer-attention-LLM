# Méthodes d'approximation du kernel exponentiel
Survey of random and deterministic methods for approximating shift-invariant and exponential kernels.

## Bochner's Theorem (foundation)
A continuous, translation-invariant kernel k(x,y) = k(x-y) is positive semi-definite iff it is the Fourier transform of a non-negative measure p(ω):
```
k(δ) = E_ω[e^(iωᵀδ)],   ω ~ p(ω)
```
This justifies Monte Carlo sampling of frequencies: draw ω ~ p(ω), build feature map φ(x) = [cos(ωᵀx), sin(ωᵀx)]. For the RBF kernel, p(ω) = N(0, I_d).

## Random Fourier Features (RFF, Rahimi & Recht 2007)
- Sample ω_1,...,ω_D i.i.d. ~ N(0, I_d)
- φ(x) = (1/√D) · [cos(ω₁ᵀx), sin(ω₁ᵀx), ..., cos(ω_Dᵀx), sin(ω_Dᵀx)]
- E[φ(x)ᵀφ(y)] = k(x-y), convergence rate O(D^(-1/2))
- Problem for softmax attention: cosine features produce negative values → normalizer in softmax attention can go negative → instability

## Positive Random Features for the exponential kernel
For softmax kernel K(x,y) = exp(xᵀy), write:
```
exp(xᵀy) = exp(-‖x‖²/2) · exp(-‖y‖²/2) · exp(‖x+y‖²/2)
```
Draw ω ~ N(0, I_d), define:
```
φ+(x) = (exp(-‖x‖²/2) / √D) · exp(ωᵀx)   (SM+)
φ_hyp+(x) = (exp(-‖x‖²/2) / √(2D)) · [exp(ωᵀx), exp(-ωᵀx)]   (SM_hyp+, lower variance)
```
Both give unbiased non-negative estimates. SM+ is used in this codebase (`_phi` function).

## Structured methods (Fastfood / SORF)
Replace the dense D×d Gaussian matrix W with a product of structured matrices:
```
W = (1/√d) · S · H · G · P · H · B
```
where S = diagonal scaling, H = Hadamard, G = diagonal Gaussian, P = permutation, B = ±1 diagonal.
- Cost: O(d log d) instead of O(Dd) for applying W
- Not yet implemented here — candidate for optimization (see context.md next steps)

## Maclaurin / polynomial kernels
For dot-product kernels K(x,y) = f(xᵀy) = Σ aₙ(xᵀy)^n:
- Sample degree n ~ categorical(aₙ) and build tensor product features
- Requires D = Ω(d · C²/ε² · log(RL/δ)) features for ε-uniform approximation
- Not directly used here but relevant for alternative kernel choices beyond softmax

## Spectral connection to FAVOR+
The approximation quality of feature maps relates to eigenvalue decay of the integral operator T_k (see `integral_operator.md`). Fast eigenvalue decay means the kernel is well-approximated by few features. The spectrum notebooks in this repo study this numerically.
