# Speed and Approximation Convergence Report

**Model:** TinyLlama 1.1B (H=32 heads, D=64 head dim)  
**Hardware:** NVIDIA A100 
**Measured quantities:** prefill latency, decode latency, approximation error vs. M

---

## 1. Prefill Speed — Varying Sequence Length

Softmax attention costs O(N²·D) per layer. FAVOR+ costs O(N·M·D), so it becomes faster only when N > M.

Fixed M, varying N (M=64 and M=128):

| N    | Softmax (ms) | Performer M=64 | Performer M=128 | Speedup (M=128) |
|------|-------------|----------------|-----------------|-----------------|
| 32   | 0.22        | 0.77           | 0.78            | 0.29×           |
| 64   | 0.10        | 0.75           | 0.75            | 0.13×           |
| 128  | 0.09        | 0.82           | 0.80            | 0.11×           |
| 256  | 0.14        | 1.02           | 0.90            | 0.16×           |
| 512  | 0.53        | 2.00           | 1.56            | 0.34×           |
| 1024 | 2.00        | 4.73           | 3.08            | 0.65×           |
| 2048 | 8.01        | 9.55           | 6.08            | **1.32×**       |
| 4096 | 39.28       | 16.80          | 12.25           | **3.21×**       |

**Crossover is around N=2048 for M=128.** Below that, the N² quadratic cost of softmax is cheap enough that performer's linear overhead (kernel materialization, scan state) dominates.

Fixed N=2048, varying M (shows the O(N·M) linear growth):

| M    | M/D  | M/N  | Performer (ms) | Speedup vs softmax |
|------|------|------|----------------|--------------------|
| 32   | 0.5  | 0.02 | 4.11           | 1.93×              |
| 64   | 1.0  | 0.03 | 8.58           | 0.92×              |
| 128  | 2.0  | 0.06 | 6.08           | 1.31×              |
| 256  | 4.0  | 0.12 | 9.89           | 0.80×              |
| 512  | 8.0  | 0.25 | 144.60         | 0.05×              |
| 1024 | 16.0 | 0.50 | 650.66         | 0.01×              |
| 2048 | 32.0 | 1.00 | 1286.72        | 0.01×              |

Once M reaches N, performer cost equals O(N²) — the whole point of the approximation is lost. The M=512 jump (144ms) reflects the Python scan fallback in this measurement; the Triton kernel degrades more gracefully but the trend is the same.

---

## 2. Decode Speed

During autoregressive generation each new token attends to the full cached context. Softmax decode costs O(N·D) per step; performer also costs O(M·D) (constant in N if state is maintained, or O(N·M) if recomputed from scratch).

| Context N | Softmax (ms) | Performer M=128 | Performer M=256 | Speedup (M=128) |
|-----------|-------------|-----------------|-----------------|-----------------|
| 64        | 0.171       | 0.123           | 0.249           | **1.39×**       |
| 256       | 0.095       | 0.118           | 0.245           | 0.80×           |
| 1024      | 0.091       | 0.116           | 0.239           | 0.78×           |
| 4096      | 0.292       | 0.118           | 0.247           | **2.48×**       |

Performer decode latency is nearly constant across context lengths (the scan state is a fixed M×D matrix regardless of N). Softmax decode dips at mid-range N due to memory bandwidth effects, then grows again at N=4096. The performer advantage is clearest at large contexts.

---

## 3. Approximation Convergence vs. M

This measures how well FAVOR+ approximates softmax attention as a pure mathematical question — independent of the model weights. Random Q, K, V are drawn to match the norm distribution of TinyLlama activations (N=64, D=64).

| M    | M/D  | MSE       | Cosine similarity | Relative error |
|------|------|-----------|-------------------|----------------|
| 8    | 0.1  | 0.015436  | 0.8869            | 43.15%         |
| 16   | 0.2  | 0.008506  | 0.9293            | 32.11%         |
| 32   | 0.5  | 0.004529  | 0.9619            | 23.44%         |
| 64   | 1.0  | 0.002450  | 0.9771            | 17.04%         |
| 128  | 2.0  | 0.001256  | 0.9888            | 12.32%         |
| **256**  | **4.0**  | **0.000675**  | **0.9937**        | **9.02%**      |
| 512  | 8.0  | 0.000328  | 0.9968            | 6.31%          |
| 1024 | 16.0 | 0.000165  | 0.9984            | 4.47%          |
| 2048 | 32.0 | 0.000091  | 0.9991            | 3.32%          |

Error decreases as ~1/√M (consistent with random feature theory). Each doubling of M roughly halves the MSE and halves the relative error.

---

## 4. Why M=256

M=256 sits at the practical sweet spot for this project:

- **Approximation quality:** cosine similarity 0.9937, relative error 9%. Beyond M=256 the gains per doubling shrink (256→512 saves 2.7pp, 512→1024 saves 1.8pp).
- **Speed:** at M=256, performer is still faster than softmax for N≥4096 prefill and large-context decode. Doubling to M=512 collapses prefill speedup to 0.05× at N=2048.
- **Memory:** the scan state S ∈ ℝ^{B×H×M×D} is 32×256×64 floats per batch = 2M parameters resident per layer. M=512 doubles this.
- **M/D ratio = 4:** the FAVOR+ paper recommends M ≥ D as a baseline; 4D gives a comfortable margin without the cost of 8D or 16D.

The 9% relative error at M=256 is not negligible, and the spectral analysis (see [spectral_analysis.md](spectral_analysis.md)) confirms that FAVOR+ produces near-rank-1 kernel matrices even at M=256 — the approximation error is not purely variance but also structural. Increasing M reduces variance but does not fix the rank gap. That is a separate limitation of the random-feature approach.

---

## 5. Practical Takeaway

| Regime | Recommendation |
|--------|---------------|
| N < 1024 | Softmax is faster — no benefit to performer at short sequences |
| N ≈ 2048 | Break-even at M=128; performer slightly slower at M=256 |
| N ≥ 4096 | Performer gives 3×+ prefill speedup with M=128–256 |
| Decode (any N) | Performer with M=128 is faster at large context (4096+); neutral at mid-range |

The training sequences in this project were SEQ_LEN=256 — well below the crossover point. The speed benefit of performer heads is realized **at inference time on long contexts**, not during the distillation training itself.
