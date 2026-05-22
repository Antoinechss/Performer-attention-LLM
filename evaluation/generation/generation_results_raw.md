
======================================================================
TEACHER (0/32 performer heads — pure softmax)
======================================================================
Perplexity: 8.02

Generations:

  Prompt: 'The history of artificial intelligence begins'
  Output: in the early 1950s with the work of mathematician Alan Turing. In the 1950s and 1960s, Turing worked on a machine called a Turing machine, which is a mathematical model of a computer. The Turing machine is a set of rules that allow a computer to execute commands and perform calculations. In the 1960s, researchers began to experiment with Turing machines and their implications for artificial intelligence. One of the first attempts was by computer scientist Marvin Minsky and his colleagues at MIT's Artificial Intelligence Laboratory. They proposed a machine called the "mind" that could simulate human thought processes.

  Prompt: 'In mathematics, a prime number is'
  Output: a non-zero integer that is neither 1 nor the product of two smaller integers. It is denoted by  or by  in the system of base 10 notation. The concept of prime numbers is essential in many fields of mathematics and is used in various mathematical and scientific contexts. In this lesson, students will learn about the properties of prime numbers and how they can be used to solve problems in mathematics. They will also learn aboutthe Fermat's Little Theorem and its applications in number theory. The lesson is suitable for students in grade 9 and above.

  Prompt: 'The capital of France is Paris. The Eiffel Tower'
  Output: is a famous landmark in Paris. Paris is a city known for its beautiful architecture, gastronomy, and fashion. 2. Rome The capital of Italy is Rome. The Colosseum is a famous landmark in Rome. Rome is a city known for its ancient history, beautiful architecture, and impressive art. 3. London The capital of the United Kingdom is London. The Tower of London is a famous landmark in London. London is a city known for its impressivearchitecture, historic landmarks, and diverse culture. 4. Tokyo The capital of Japan is Tokyo. The Tokyo Tower is a famous landmark in Tokyo. Tokyo is a city known for its bustling streets, world-renowned food

======================================================================
  Loaded best_phase1_K4_QK.pt — 4/32 performer heads
CHECKPOINT: phase1_K4_QK (4/32 performer heads)
======================================================================
Perplexity: 8.89

Generations:

  Prompt: 'The history of artificial intelligence begins'
Traceback (most recent call last):
  File "/workspace/Performer-attention-LLM/finetune/generate.py", line 235, in <module>
    main()
  File "/workspace/Performer-attention-LLM/finetune/generate.py", line 226, in main
    print(f"  Output: {generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)}")
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Performer-attention-LLM/finetune/generate.py", line 121, in generate
    out = model.generate(
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 2215, in generate
    result = self._sample(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 3206, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 1190, in forward
    outputs = self.model(
              ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 972, in forward
    next_cache = next_cache.to_legacy_cache()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'to_legacy_cache'
root@0a0310ef95e9:/workspace/Performer-attention-LLM# git pull && python finetune/generate.py --ckpt_dir /workspace/checkpoints
remote: Enumerating objects: 11, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (4/4), done.
remote: Total 7 (delta 4), reused 6 (delta 3), pack-reused 0 (from 0)
Unpacking objects: 100% (7/7), 2.71 KiB | 26.00 KiB/s, done.
From https://github.com/Antoinechss/Performer-attention-LLM
   b2df25a..a825894  main       -> origin/main
Updating b2df25a..a825894
Fast-forward
 README.md            | 143 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 finetune/generate.py |   1 +
 2 files changed, 142 insertions(+), 2 deletions(-)
/usr/local/lib/python3.11/dist-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.2.3) or chardet (6.0.0.post1)/charset_normalizer (3.3.2) doesn't match a supported version!
  warnings.warn(

Loading teacher (softmax baseline)...

======================================================================
TEACHER (0/32 performer heads — pure softmax)
======================================================================
Perplexity: 8.02

Generations:

  Prompt: 'The history of artificial intelligence begins'
  Output: with the development of the first true computer. The first computer was built by IBM in 1943 and was capable of performing only simple arithmetic operations. However, it was not until the 1950s that significant progress was made in the field of artificial intelligence. In 1956, American computer scientist John McCarthy introduced the term "artificial intelligence" to describe the study of intelligent machines.

II. The development of neural networks

The 1950s and 1960s were a period of significant advancements in artificial intelligence. One of the most significant developments was the introduction of the concept of neural networks. Neural networks are a type

  Prompt: 'In mathematics, a prime number is'
  Output: a number that is greater than 1 and has no positive integer factor other than 1 and itself. The smallest prime number is 2, and the largest prime number is 231 - 1. A prime number is not a composite number, meaning that it has no positive integer factor other than 1 and itself. A composite number is a number that has a positive integer factor other than 1 and itself. For example, 32 is a composite number because it has a factor of 4, which is not a factor of 3. 27 is also a composite number because it has a factor of 12, which is not a factor of 2. 125 is a prime number because

  Prompt: 'The capital of France is Paris. The Eiffel Tower'
  Output: is located in Paris.

4. B: The capital of Italy is Rome. The Colosseum is located in Rome.

5. C: The capital of Spain is Madrid. The Royal Palace of Madrid is located in Madrid.

6. D: The capital of Sweden is Stockholm. The Royal Palace of Stockholm is located in Stockholm.

7. E: The capital of the United States is Washington D.C. The White House is located in Washington D.C.

8. F: The capital of Australia is Canberra. The Parliament House of Australia is located in Canberra.

9. G: The capital of Canada is Ottawa. The Parliament Hill of Canada is

======================================================================
  Loaded best_phase1_K4_QK.pt — 4/32 performer heads
CHECKPOINT: phase1_K4_QK (4/32 performer heads)
======================================================================
Perplexity: 8.89

Generations:

  Prompt: 'The history of artificial intelligence begins'
  Output: with the invention of the first computers, which transformed the world of science and technology. The development of the first AI systems was driven by the desire to create machines that could perform complex tasks that were previously impossible for humans to achieve.

The first artificial intelligence systems were designed to perform specific tasks, such as counting money or playing chess. However, as technologycontinued to advance, AI systems began to become more complex and able to perform more complex tasks.

One of the first successful AI systems was the IBM Watson, which was developed to help doctors diagnose medical conditions. The Watson system usesnatural language processing and machine learning algorithms to analyze medical records and diagnose diseases.

The next significant development in

  Prompt: 'In mathematics, a prime number is'
  Output: a whole number that is not divisible by any non-zero integer. In many languages, the term "prime" is used to refer to a non-negative integer, but this is not always true for all languages. For example, in Japanese, the word prime is used to refer to an integer greater than 1.

  Prompt: 'The capital of France is Paris. The Eiffel Tower'
  Output: is one of its most famous landmarks.

B. The city of Dubai is famous for its tallest building, the Burj Khalifa.

C. The city of Bangkok, Thailand, is famous for its famous temples, such as Wat Phra Kaew and Wat Arun.

Ques. 4. What is the capital of Germany and what is the name of its most famous landmark, according to the text material: "Germany's capital city is Berlin. The Brandenburg Gate is one of its most famous landmarks."

======================================================================
  Loaded best_phase2_K8_QKVO.pt — 8/32 performer heads
CHECKPOINT: phase2_K8_QKVO (8/32 performer heads)
======================================================================
Perplexity: 11.68

Generations:

  Prompt: 'The history of artificial intelligence begins'
  Output: with a machine learning algorithm that mimics the way humans learn. This early algorithm was called "Mona Lisa," and it was first implemented by a team of computer scientists led by Andrew Ng at the University of California, Berkeley.

In 1995, a team of researchers at the University of Illinois developed the AlphaGo, a program that outperformed the best human players in the world in a series of games. In 2010, a team of researchers at the University of California, Berkeley developed the DeepMind AlphaGo, a program that outperformed the best human players in a series of games.
In 2015, a team of research

  Prompt: 'In mathematics, a prime number is'
  Output: a whole number greater than 1 and less than 2n, where n is a positive integer.





































































 4. The sum of the series: 1 + 2 + 3 + ... + 2n

























 5.


  Prompt: 'The capital of France is Paris. The Eiffel Tower'
  Output: is the most famous landmark in the city. The Louvre Museum is one of the world’s most famous museums. The Arc de Triomphe is a monument in Paris, France. The Notre Dame Cathedral in Paris is a symbol of the Catholic Church. The Louvre Museum is the most famous museum in the world. The Eiffel Tower is the most famous landmark in Paris. The Louvre Museum is one of the world’s most famous museums. The Arc de Triomphe is a monument in Paris, France. The Notre Dame Cathedral in Paris is a symbol of the Catholic Church. The Louvre Museum is the most famous museum in the world.The Eiffel Tower is the most

======================================================================
  Loaded best_phase3_K16_QKVO.pt — 16/32 performer heads
CHECKPOINT: phase3_K16_QKVO (16/32 performer heads)
======================================================================
Perplexity: 11.01

Generations:

  Prompt: 'The history of artificial intelligence begins'
  Output: with the invention of the printing press in the 1440s, with the invention of the printing press in the 15th century. The invention of the printing press had a significant impact on the spread of information, allowing the creation of the first printed book in 1445. The invention of the printing press transformed the world of knowledge, allowing the dissemination of information, and the growth of knowledge. The invention of the printing press also enabled the development of the written word, allowing for the first printed books to be created, and the first book to be printed in 1438. The invention of the printing press also paved the way for the development of the printing

  Prompt: 'In mathematics, a prime number is'
  Output: a positive integer that is greater than 1. A prime number is a positive integer that is greater than 1.








In mathematics, a prime number is a positive integer that is greater than 1.
The first prime number is 2. The second prime number is 3.
A prime number is a prime number that is greater than 1.
A prime number is a positive integer that is greater than 1.
A prime number is a prime number that is greater than 1. A prime number is a positive integer that is greater than 1.
A prime number is a positive integer that is greater than 1. A prime number is a positive integer that is

  Prompt: 'The capital of France is Paris. The Eiffel Tower'
  Output: is the highest in the world, and it was built in 1882 as the Gare de L'Horloge (Gare du Carrousel) and was officially opened in 1889.

The city is the capital of the region of Auvergne-Charente-Charente-Nouvelle-Aquitaine, and is the capital of the region of Nouvelle-Aquitaine. The population of the city is 52,447, with an area of 15,580 hectares, with an average of 42,544 people per square kilometre.

The name of the city is derived from the Latin name of the city

======================================================================
  Loaded best_phase4_K32_QKVO.pt — 32/32 performer heads
CHECKPOINT: phase4_K32_QKVO (32/32 performer heads)
======================================================================
Perplexity: 250.91

Generations:

  Prompt: 'The history of artificial intelligence begins'
  Output: in the first and other ways of the city. 2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

  Prompt: 'In mathematics, a prime number is'
  Output: not very little more than
February, 200360002080000000000000000000000000000000000012000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000

  Prompt: 'The capital of France is Paris. The Eiffel Tower'
  Output: , ascentrate, and an artwork of the city of the 180000000000000000000000000000000016000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

Done.