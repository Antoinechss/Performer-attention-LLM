# Generation Quality Report

Qualitative comparison of text generation across fine-tuning phases.
Model: TinyLlama 1.1B. Temperature: 0.7, top_p: 0.9, max_new_tokens: 150.

---

## Perplexity Summary

| Model | Performer heads | Perplexity |
|-------|----------------|------------|
| Teacher (softmax) | 0/32 | 8.02 |
| Phase 1 (fine-tuned) | 4/32 | 8.89 |
| Phase 2 (fine-tuned) | 8/32 | 11.68 |
| Phase 3 (fine-tuned) | 16/32 | 11.01 |
| Phase 4 (fine-tuned) | 32/32 | 250.91 |

Note: perplexity here is computed on a small held-out set of general-knowledge sentences, distinct from the WikiText-103 training domain, which explains the lower absolute values compared to the training eval.

---

## Prompt: "The history of artificial intelligence begins"

**Teacher (softmax)**
> with the development of the first true computer. The first computer was built by IBM in 1943 and was capable of performing only simple arithmetic operations. However, it was not until the 1950s that significant progress was made in the field of artificial intelligence. In 1956, American computer scientist John McCarthy introduced the term "artificial intelligence" to describe the study of intelligent machines.

**Phase 1 — 4/32 performer heads (ppl: 8.89)**
> with the invention of the first computers, which transformed the world of science and technology. The development of the first AI systems was driven by the desire to create machines that could perform complex tasks that were previously impossible for humans to achieve. One of the first successful AI systems was the IBM Watson, which was developed to help doctors diagnose medical conditions.

**Phase 2 — 8/32 performer heads (ppl: 11.68)**
> with a machine learning algorithm that mimics the way humans learn. This early algorithm was called "Mona Lisa," and it was first implemented by a team of computer scientists led by Andrew Ng at the University of California, Berkeley. In 1995, a team of researchers at the University of Illinois developed the AlphaGo...

**Phase 3 — 16/32 performer heads (ppl: 11.01)**
> with the invention of the printing press in the 1440s... The invention of the printing press had a significant impact on the spread of information, allowing the creation of the first printed book in 1445.

*Factual drift: the model conflates AI history with the history of printing.*

**Phase 4 — 32/32 performer heads (ppl: 250.91)**
> in the first and other ways of the city. 2000000000000000000000000000000000000...

*Complete collapse — degenerate output.*

---

## Prompt: "In mathematics, a prime number is"

**Teacher**
> a number that is greater than 1 and has no positive integer factor other than 1 and itself. The smallest prime number is 2, and the largest prime number is 2³¹ - 1...

**Phase 1 — 4/32**
> a whole number that is not divisible by any non-zero integer...

*Slightly imprecise definition but fluent.*

**Phase 2 — 8/32**
> a whole number greater than 1 and less than 2n, where n is a positive integer.

*Definition is wrong. Followed by degenerate whitespace then "4. The sum of the series: 1 + 2 + 3 + ... + 2n".*

**Phase 3 — 16/32**
> a positive integer that is greater than 1. A prime number is a positive integer that is greater than 1. [...repeated ~10 times...]

*Repetition loop — loss of coherence.*

**Phase 4 — 32/32**
> not very little more than February, 2003600020800000000000...

*Incoherent.*

---

## Prompt: "The capital of France is Paris. The Eiffel Tower"

**Teacher**
> is located in Paris. [Continues with Rome, Madrid, Stockholm, Washington D.C., Canberra...] — coherent list completion.

**Phase 1 — 4/32**
> is one of its most famous landmarks. [Continues with Dubai's Burj Khalifa, Bangkok temples...] — coherent continuation, slight topic drift.

**Phase 2 — 8/32**
> is the most famous landmark in the city. The Louvre Museum is one of the world's most famous museums... [Repetitive loop after ~2 sentences.]

**Phase 3 — 16/32**
> is the highest in the world, and it was built in 1882 as the Gare de L'Horloge... The city is the capital of the region of Auvergne-Charente...

*Factually wrong (Eiffel Tower built 1889, not 1882; wrong region) but grammatically fluent.*

**Phase 4 — 32/32**
> , ascentrate, and an artwork of the city of the 18000000000000000...

*Incoherent.*

---

## Analysis

**Phase 1 (4/32 heads)** is nearly indistinguishable from the teacher in generation quality. Text is fluent, factually reasonable, and coherent. Perplexity cost is small (+0.87 vs teacher).

**Phase 2 (8/32 heads)** shows the first clear degradation: repetition loops on the prime number prompt, and the AI history prompt hallucinates (AlphaGo attributed to UC Berkeley, "Mona Lisa" algorithm). Still grammatically fluent throughout.

**Phase 3 (16/32 heads)** exhibits severe repetition on structured prompts and factual drift (printing press for AI history). The surprisingly low perplexity (11.01 vs 11.68 for Phase 2) likely reflects overfitting to the distillation signal rather than genuine quality.

**Phase 4 (32/32 heads)** collapsed completely. All outputs are incoherent or degenerate sequences of digits. This is consistent with the training perplexity of 161 — replacing all attention heads with FAVOR+ without longer training was too aggressive.

---

## Conclusion

The sweet spot is **Phase 1 (4/32 heads)**: linear attention on 12.5% of heads with negligible quality loss. Phase 2 (8/32 heads, 25% of heads) is usable for non-critical generation but shows early degradation. Beyond 16/32 heads, the model quality degrades unacceptably with the current training budget and sequence length.
