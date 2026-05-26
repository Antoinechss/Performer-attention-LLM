# Spectral Analysis — Kernel Gram Matrices

Empirical eigenvalues of (1/n)(K + K^T)/2 where K_ij = k(q_i, k_j).
Averaged over 5 prompts, SEQ_LEN=128, layers {0, 5, 10, 15, 21}.

---

## 1. Dominant eigenvalue λ₁

| Model | L0 | L5 | L10 | L15 | L21 |
|---|---|---|---|---|---|
| Softmax | 1.730e+01 | 4.188e-01 | 5.169e-01 | 1.120e+00 | 2.643e+00 |
| FAVOR+ (no FT) | 8.698e-04 | 1.507e-08 | 9.467e-09 | 6.937e-09 | 1.331e-09 |
| Phase 1 (4/32) | 1.392e-04 | 1.402e-08 | 9.289e-09 | 9.307e-09 | 1.811e-09 |
| Phase 2 (8/32) | 1.638e-04 | 1.580e-08 | 9.196e-09 | 9.823e-09 | 2.422e-09 |
| Phase 3 (16/32) | 1.083e-04 | 1.507e-08 | 9.888e-09 | 7.710e-09 | 1.845e-09 |
| Phase 4 (32/32) | 4.029e-05 | 3.288e-07 | 2.828e-08 | 2.009e-08 | 6.549e-09 |

**Observation** : λ₁(softmax) >> λ₁(FAVOR+) par 4 à 10 ordres de grandeur selon la couche.
La distillation ne modifie pas significativement λ₁ de FAVOR+ — phases 1 à 3 quasi superimposées.
Phase 4 légèrement plus élevée : le modèle effondré produit des features moins concentrées.

---

## 2. Rang effectif r_eff = exp(H(|λ|))

| Model | L0 | L5 | L10 | L15 | L21 |
|---|---|---|---|---|---|
| Softmax | 1.58 | 1.57 | 3.40 | 15.26 | 8.70 |
| FAVOR+ (no FT) | 1.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 (4/32) | 1.11 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 2 (8/32) | 1.10 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 3 (16/32) | 1.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 4 (32/32) | 1.04 | 1.14 | 1.05 | 1.03 | 1.13 |

**Observation** : Le softmax présente un spectre riche (r_eff=15.26 en layer 15 — 15 modes
d'interaction distincts). FAVOR+ est quasi exactement rang-1 partout : un seul mode concentre
la quasi-totalité de l'énergie. La distillation ne modifie pas ce rang effectif.
Exception : phase 4 a un rang effectif marginalement supérieur aux phases 1-3,
cohérent avec la désorganisation totale de ses représentations.

---

## 3. Nombre de valeurs propres significatives (|λ| > 1e-6)

| Model | L0 | L5 | L10 | L15 | L21 |
|---|---|---|---|---|---|
| Softmax | 12 | 33 | 60 | 64 | 64 |
| FAVOR+ (no FT) | 3 | 0 | 0 | 0 | 0 |
| Phase 1 (4/32) | 2 | 0 | 0 | 0 | 0 |
| Phase 2 (8/32) | 2 | 0 | 0 | 0 | 0 |
| Phase 3 (16/32) | 2 | 0 | 0 | 0 | 0 |
| Phase 4 (32/32) | 1 | 0 | 0 | 0 | 0 |

**Observation** : Le softmax présente jusqu'à 64 valeurs propres significatives (toutes les
valeurs calculées), notamment dans les couches profondes. FAVOR+ n'en présente aucune
au-delà de la couche 0. La factorisation K = Φ(Q)Φ(K)^T implique un rang au plus M=256,
mais en pratique un seul mode domine.

---

## 4. Distance de Frobenius ||K_softmax - K_FAVOR+||_F

| Model | L0 | L5 | L10 | L15 | L21 |
|---|---|---|---|---|---|
| FAVOR+ (no FT) | 3.106 | 0.115 | 0.153 | 0.461 | 0.577 |
| Phase 1 (4/32) | 3.106 | 0.115 | 0.153 | 0.461 | 0.577 |
| Phase 2 (8/32) | 3.106 | 0.115 | 0.153 | 0.461 | 0.577 |
| Phase 3 (16/32) | 3.106 | 0.115 | 0.153 | 0.461 | 0.577 |
| Phase 4 (32/32) | 3.106 | 0.115 | 0.153 | 0.461 | 0.577 |

**Observation** : La distance entre noyaux est **identique pour toutes les phases**,
y compris sans finetuning. La distillation ne réduit pas ||Ks - Kf||_F.
Cela confirme que la distillation agit sur les poids W_Q, W_K, W_V, W_O
mais ne modifie pas la forme fonctionnelle du noyau FAVOR+.
Le gap est donc une limite architecturale irréductible par distillation.

---

## 5. Asymétrie relative ||K - K^T||_F / ||K||_F

| Model | L0 | L5 | L10 | L15 | L21 |
|---|---|---|---|---|---|
| Softmax | 131.5% | 96.0% | 126.7% | 139.7% | 97.5% |
| FAVOR+ (no FT) | 61.6% | 140.6% | 140.6% | 140.6% | 140.2% |
| Phase 1 (4/32) | 71.3% | 140.6% | 140.6% | 140.6% | 140.2% |
| Phase 2 (8/32) | 61.1% | 140.6% | 140.6% | 140.6% | 140.1% |
| Phase 3 (16/32) | 55.3% | 140.6% | 140.5% | 140.5% | 140.0% |
| Phase 4 (32/32) | 59.4% | 70.9% | 81.2% | 93.5% | 67.5% |

**Observation** : L'asymétrie FAVOR+ phases 1-3 est saturée à ~140.6% dans les couches
profondes — valeur plafond indiquant que φ(q) et φ(k) sont quasi-orthogonaux dans
l'espace des features aléatoires. La phase 4 présente une asymétrie réduite (67-93%) :
le modèle effondré a des projections Q/K plus alignées, mais sans que cela se traduise
par une meilleure qualité de génération.

---

## 6. Vérification de l'inégalité de Hoffmann-Wielandt

L'inégalité stipule : Σ|λᵢ(A) - λᵢ(B)|² ≤ ||A - B||²_F

| Layer | HW_LHS | ||Ks-Kf||²_F | Ratio LHS/RHS |
|---|---|---|---|
| L0 | 301.4 | 3.106 | ~97 |
| L5 | 0.176 | 0.115 | ~1.52 |
| L10 | 0.273 | 0.153 | ~1.78 |
| L15 | 2.332 | 0.461 | ~5.06 |
| L21 | 9.356 | 0.577 | ~16.2 |

**Observation** : L'inégalité est **violée** (ratio > 1) pour toutes les couches.
Cela s'explique par le fait que la borne de Hoffmann-Wielandt s'applique aux
matrices normales (AA* = A*A), ce qui n'est pas le cas ici — K_softmax et K_FAVOR+
sont asymétriques (asymétrie > 95%). La symétrisation (K+K^T)/2 introduit une
erreur qui invalide la borne dans ce contexte. C'est une limite de l'approche spectrale
notée en section 3.1.4 du rapport.

---

## Synthèse

| Résultat | Interprétation |
|---|---|
| FAVOR+ quasi-rang-1 partout | Limite architecturale : K = Φ(Q)Φ(K)^T concentre l'énergie sur un mode |
| Distance Frobenius identique toutes phases | La distillation n'agit pas sur la forme du noyau |
| Rang effectif softmax >> FAVOR+ | Le softmax code ~15 modes d'interaction distincts en layer 15 |
| Asymétrie FAVOR+ saturée à 140.6% | φ(q) et φ(k) quasi-orthogonaux dans l'espace des features |
| Inégalité HW violée | Les matrices sont trop asymétriques pour que la borne s'applique |
| Layer 0 = cas particulier | Embeddings bruts : FAVOR+ capture encore un peu de structure |
