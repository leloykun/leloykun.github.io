---
title: "Llama.cpp"
# date: 2023-07-25
tags: ["Machine Learning", "C++"]
author: "Franz Louis Cesista"
description: "A C++ implementation of Meta's Llama2 generative large-language model. I also optimized the original C implementation by Karpathy by adding parallelization on
the multi-head attention layer."
summary: "A C++ implementation of Meta's Llama2 generative large-language model. I also optimized the original C implementation by Karpathy by adding parallelization on
the multi-head attention layer."
cover:
    image: "llama_cute.jpg"
    alt: "Cute Llama"
editPost:
    URL: "https://github.com/leloykun/llama2.cpp"
    Text: "Github Repository"
weight: 2
---

![llama_cute.jpg](llama_cute.jpg)

---

With this code you can train the Llama 2 LLM architecture from scratch in PyTorch, then save the weights to a raw binary file, then load that into one ~simple 425-line C++ file (run.cpp) that inferences the model, simply in fp32 for now. On my cloud Linux devbox a dim 288 6-layer 6-head model (~15M params) inferences at ~100 tok/s in fp32, and about the same on my M1 MacBook Air. I was somewhat pleasantly surprised that one can run reasonably sized models (few ten million params) at highly interactive rates with an approach this simple.

Please note that this is just a weekend project: I took nanoGPT, tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C++ inference engine in run.cpp. As such, this is not really meant to be a production-grade library right now.
