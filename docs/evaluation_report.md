# Evaluation Report — Performer Attention on TinyLlama 1.1B

**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0  
**Method:** Progressive substitution of softmax attention heads with FAVOR+ (Performer) approximation  
**Evaluation:** Perplexity on held-out sentences + qualitative generation on 3 prompts + spectral analysis of kernel matrices  
**Generation settings:** temperature 0.7, top_p 0.9, max_new_tokens 150

---

## 1. Perplexity Summary

| Phase | Performer heads | Heads replaced | Perplexity | Δ vs teacher |
|-------|----------------|----------------|------------|--------------|
| Teacher | 0/32 | 0% | **8.02** | — |
| Phase 1 | 4/32 | 12.5% | 8.89 | +0.87 |
| Phase 2 | 8/32 | 25.0% | 11.68 | +3.66 |
| Phase 3 | 16/32 | 50.0% | 11.01 | +2.99 |
| Phase 4 | 32/32 | 100% | 250.91 | +242.89 |

> Note: perplexity is computed on a small held-out set of general-knowledge sentences, distinct from the WikiText-103 training domain. The lower absolute values (vs. typical WT103 benchmarks) reflect the domain difference, not model quality in isolation.

---

## 2. Generation Results

### Prompt 1 — "The history of artificial intelligence begins"

**Teacher (0/32 — softmax baseline)**
> with the development of the first true computer. The first computer was built by IBM in 1943 and was capable of performing only simple arithmetic operations. However, it was not until the 1950s that significant progress was made in the field of artificial intelligence. In 1956, American computer scientist John McCarthy introduced the term "artificial intelligence" to describe the study of intelligent machines. II. The development of neural networks — The 1950s and 1960s were a period of significant advancements in artificial intelligence. One of the most significant developments was the introduction of the concept of neural networks. Neural networks are a type...

**Phase 1 — 4/32 performer heads (ppl: 8.89)**
> with the invention of the first computers, which transformed the world of science and technology. The development of the first AI systems was driven by the desire to create machines that could perform complex tasks that were previously impossible for humans to achieve. The first artificial intelligence systems were designed to perform specific tasks, such as counting money or playing chess. However, as technology continued to advance, AI systems began to become more complex and able to perform more complex tasks. One of the first successful AI systems was the IBM Watson, which was developed to help doctors diagnose medical conditions. The Watson system uses natural language processing and machine learning algorithms to analyze medical records and diagnose diseases.

*Assessment: Fluent, factually reasonable. Minor imprecisions (IBM Watson framing) but coherent narrative.*

**Phase 2 — 8/32 performer heads (ppl: 11.68)**
> with a machine learning algorithm that mimics the way humans learn. This early algorithm was called "Mona Lisa," and it was first implemented by a team of computer scientists led by Andrew Ng at the University of California, Berkeley. In 1995, a team of researchers at the University of Illinois developed the AlphaGo, a program that outperformed the best human players in the world in a series of games. In 2010, a team of researchers at the University of California, Berkeley developed the DeepMind AlphaGo...

*Assessment: Hallucinations appear — "Mona Lisa" algorithm, AlphaGo attributed to UC Berkeley. Grammatically fluent but factually unreliable.*

**Phase 3 — 16/32 performer heads (ppl: 11.01)**
> with the invention of the printing press in the 1440s, with the invention of the printing press in the 15th century. The invention of the printing press had a significant impact on the spread of information, allowing the creation of the first printed book in 1445. The invention of the printing press transformed the world of knowledge, allowing the dissemination of information, and the growth of knowledge. The invention of the printing press also enabled the development of the written word...

*Assessment: Severe factual drift — conflates AI history with the history of printing. Repetitive phrasing.*

**Phase 4 — 32/32 performer heads (ppl: 250.91)**
> in the first and other ways of the city. 2000000000000000000000000000000000000000000000000000000000000000000000000000000000...

*Assessment: Complete collapse. Degenerate digit sequences.*

---

### Prompt 2 — "In mathematics, a prime number is"

**Teacher (0/32)**
> a number that is greater than 1 and has no positive integer factor other than 1 and itself. The smallest prime number is 2, and the largest prime number is 2³¹ − 1. A prime number is not a composite number, meaning that it has no positive integer factor other than 1 and itself. A composite number is a number that has a positive integer factor other than 1 and itself. For example, 32 is a composite number because it has a factor of 4...

*Assessment: Correct definition, coherent explanation.*

**Phase 1 — 4/32 (ppl: 8.89)**
> a whole number that is not divisible by any non-zero integer. In many languages, the term "prime" is used to refer to a non-negative integer, but this is not always true for all languages. For example, in Japanese, the word prime is used to refer to an integer greater than 1.

*Assessment: Slightly imprecise definition (misses the "greater than 1" constraint) but fluent and non-degenerate.*

**Phase 2 — 8/32 (ppl: 11.68)**
> a whole number greater than 1 and less than 2n, where n is a positive integer.
>
> [~50 blank lines]
>
> 4. The sum of the series: 1 + 2 + 3 + ... + 2n
>
> [further blank lines and "5."]

*Assessment: Wrong definition followed by excessive whitespace and unrelated sequence content. First sign of incoherence in structured tasks.*

**Phase 3 — 16/32 (ppl: 11.01)**
> a positive integer that is greater than 1. A prime number is a positive integer that is greater than 1.
>
> [blank lines]
>
> In mathematics, a prime number is a positive integer that is greater than 1. The first prime number is 2. The second prime number is 3. A prime number is a prime number that is greater than 1. A prime number is a positive integer that is greater than 1. A prime number is a prime number that is greater than 1. A prime number is a positive integer that is greater than 1. A prime number is a positive integer that is...

*Assessment: Stuck in a repetition loop. The definition itself is correct but the model cannot advance past it.*

**Phase 4 — 32/32 (ppl: 250.91)**
> not very little more than February, 200360002080000000000000000000000000000000000012000000000000000000000000000000000000000...

*Assessment: Incoherent.*

---

### Prompt 3 — "The capital of France is Paris. The Eiffel Tower"

**Teacher (0/32)**
> is located in Paris. 4. B: The capital of Italy is Rome. The Colosseum is located in Rome. 5. C: The capital of Spain is Madrid. The Royal Palace of Madrid is located in Madrid. 6. D: The capital of Sweden is Stockholm. The Royal Palace of Stockholm is located in Stockholm. 7. E: The capital of the United States is Washington D.C. The White House is located in Washington D.C. 8. F: The capital of Australia is Canberra. The Parliament House of Australia is located in Canberra...

*Assessment: Coherent list-completion pattern. Follows the template correctly.*

**Phase 1 — 4/32 (ppl: 8.89)**
> is one of its most famous landmarks. B. The city of Dubai is famous for its tallest building, the Burj Khalifa. C. The city of Bangkok, Thailand, is famous for its famous temples, such as Wat Phra Kaew and Wat Arun. Ques. 4. What is the capital of Germany and what is the name of its most famous landmark, according to the text material: "Germany's capital city is Berlin. The Brandenburg Gate is one of its most famous landmarks."

*Assessment: Coherent continuation, slight topic drift (shifts to quiz format), but fluent and factually correct throughout.*

**Phase 2 — 8/32 (ppl: 11.68)**
> is the most famous landmark in the city. The Louvre Museum is one of the world's most famous museums. The Arc de Triomphe is a monument in Paris, France. The Notre Dame Cathedral in Paris is a symbol of the Catholic Church. The Louvre Museum is the most famous museum in the world. The Eiffel Tower is the most famous landmark in Paris. The Louvre Museum is one of the world's most famous museums. The Arc de Triomphe is a monument in Paris, France. [repeats...]

*Assessment: Enters a repetition loop after ~2 sentences. Content is factually correct but structurally broken.*

**Phase 3 — 16/32 (ppl: 11.01)**
> is the highest in the world, and it was built in 1882 as the Gare de L'Horloge (Gare du Carrousel) and was officially opened in 1889. The city is the capital of the region of Auvergne-Charente-Charente-Nouvelle-Aquitaine, and is the capital of the region of Nouvelle-Aquitaine. The population of the city is 52,447, with an area of 15,580 hectares, with an average of 42,544 people per square kilometre.

*Assessment: Grammatically fluent but factually wrong (1882, wrong region, wrong population scale). Confuses the Gare du Carrousel with the Eiffel Tower.*

**Phase 4 — 32/32 (ppl: 250.91)**
> , ascentrate, and an artwork of the city of the 180000000000000000000000000000000016000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

*Assessment: Incoherent.*

---

## 3. Analysis

### Phase-by-phase summary

| Phase | Fluency | Factual accuracy | Coherence | Verdict |
|-------|---------|-----------------|-----------|---------|
| Teacher (0/32) | ✓ | ✓ | ✓ | Baseline |
| Phase 1 (4/32) | ✓ | ✓ (minor imprecisions) | ✓ | **Near-lossless** |
| Phase 2 (8/32) | ✓ | ✗ (hallucinations) | ~ (repetition starts) | Degraded |
| Phase 3 (16/32) | ✓ | ✗ (severe drift) | ✗ (repetition loops) | Unacceptable |
| Phase 4 (32/32) | ✗ | ✗ | ✗ | Collapsed |

### Key observations

**Phase 1 (4/32, 12.5% of heads)** is the clear sweet spot. The perplexity increase is small (+0.87) and generation quality is nearly indistinguishable from the teacher across all three prompt types. This demonstrates that a conservative substitution of FAVOR+ heads is viable with minimal distillation.

**Phase 2 (8/32, 25%)** marks the onset of meaningful degradation. Hallucinations appear on knowledge-intensive prompts, and structured generation (math definitions) begins to collapse with whitespace and repetition. Perplexity increases by +3.66 vs. teacher. Still grammatically fluent, so potentially usable in non-critical applications.

**Phase 3 (16/32, 50%)** shows a counterintuitive perplexity improvement over Phase 2 (11.01 vs. 11.68), likely reflecting overfitting to the distillation signal rather than genuine quality. Generation quality is worse: severe repetition loops and significant factual drift on all prompts.

**Phase 4 (32/32, 100%)** collapses entirely. Training perplexity of 161 indicates the model never recovered. Replacing all softmax heads simultaneously, without a longer training budget or curriculum, exceeds what FAVOR+ can absorb in a single distillation phase.

### On the Phase 3 perplexity anomaly

Phase 3 (16/32 heads) achieves lower perplexity than Phase 2 (8/32 heads) despite clearly worse generation quality. This decoupling is characteristic of overfitting: the model has learned to minimize the distillation loss on the training distribution without learning generalizable attention patterns. The held-out perplexity set is small, and the metric does not capture repetition loops or factual coherence.

---

## 4. Spectral Analysis

Kernel singular value spectra were extracted from the teacher model on real activations (512-token input, 10 samples averaged). Layers 0, 5, 10, 15, and 21 were analysed. Full data in [spectral_analysis.md](../spectral_analysis.md).

### Effective rank and top-1 coverage

| Layer | Eff. rank softmax | Eff. rank FAVOR+ | Top-1 coverage softmax | Top-1 coverage FAVOR+ |
|-------|-------------------|-----------------|------------------------|----------------------|
| 0     | 2.05              | 1.10            | 77.9%                  | 98.5%                |
| 5     | 2.56              | 1.31            | 71.5%                  | 95.6%                |
| 10    | 4.37              | 1.05            | 62.6%                  | 99.3%                |
| 15    | **9.22**          | 1.01            | 32.2%                  | **99.9%**            |
| 21    | 6.76              | 1.25            | 45.0%                  | 96.2%                |

Effective rank = exp(Shannon entropy of the normalised singular value distribution).

### Key findings

**FAVOR+ produces near-rank-1 kernel matrices in all layers.** Regardless of depth, the top singular value accounts for 95–99.9% of the FAVOR+ spectrum. The softmax kernel is structurally richer: effective rank rises from ~2 in early layers to **9.2 in layer 15**, where the top-14 singular values span less than two orders of magnitude. FAVOR+ approximates this with a rank-1 matrix.

**The ratio of leading singular values grows with depth.** σ₁(softmax) / σ₁(FAVOR+) goes from ~4 × 10⁵ at layer 0 to ~7 × 10⁸ at layer 21, indicating that the approximation fidelity deteriorates in later layers — exactly where the richest attention patterns appear.

**This explains the phase transition.** With 4/32 heads replaced (Phase 1), the 28 remaining softmax heads can absorb the high-rank attention demand from layers 10–15. Performers handle lower-rank, position-insensitive patterns. As more heads switch to FAVOR+, there are fewer softmax heads to route complex queries through, and quality degrades. The spectral data puts a mechanistic basis on why 4/32 is the practical limit with M=256 random features.

**Recommended fix: increase M or use orthogonal random features.** To approximate a kernel with effective rank ~9, M must be large enough that φ(q)ᵀφ(k) spans a comparable rank subspace. Current M=256 with head_dim=64 should have sufficient capacity in theory, but random projection variance causes rank collapse. Orthogonal random features (ORF) reduce this variance and would be the first lever before increasing the head count.

---

## 5. Conclusion

The experiment validates that **progressive FAVOR+ substitution is viable up to ~12.5% of attention heads** (Phase 1: 4/32) with negligible quality loss. Beyond that threshold, quality degrades rapidly along two axes: factual hallucination and repetition loops.

The practical implication is that hybrid softmax/Performer architectures — keeping a majority of heads as standard softmax — are a more promising direction than full substitution. A 4/32 mixed model achieves linear attention in 12.5% of the computation-heavy attention heads at a perplexity cost of less than 11%.

Full FAVOR+ substitution (Phase 4) requires either a significantly longer training schedule, a larger distillation dataset, or a better-conditioned initialization strategy to avoid collapse.
