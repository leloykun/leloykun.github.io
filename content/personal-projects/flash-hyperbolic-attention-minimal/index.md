---
title: "Flash Hyperbolic Attention Minimal [WIP]"
date: 2024-04-16
tags: ["Machine Learning", "C++", "CUDA", "PyTorch", "Non-Euclidean Geometry", "Flash Attention", "Hyperbolic Geometry"]
author: "Franz Louis Cesista"
description: "A minimal implementation of Flash Attention 1 & 2 in just ~350 lines of CUDA code. This is still a work-in-progress, but the ultimate goal is to implement the
various variations of Hyperbolic Attention in CUDA."
summary: "A minimal implementation of Flash Attention 1 & 2 in just ~350 lines of CUDA code. This is still a work-in-progress, but the ultimate goal is to implement the
various variations of Hyperbolic Attention in CUDA."
editPost:
    URL: "https://github.com/leloykun/flash-hyperbolic-attention-minimal"
    Text: "Github Repository"
---

A minimal re-implementation of Flash Attention with CUDA and PyTorch. The [official implementation](https://github.com/Dao-AILab/flash-attention) can be quite daunting for a CUDA beginner (like myself), so this repo tries to be small and educational.

- The end goal of this repo is to implement Flash Attention-like kernels for the various hyperbolic attention algorithms, finally making them production-ready.
- This was forked from Peter Kim's [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) repo.
- The variable names follow the notations from the [original paper](https://arxiv.org/abs/2205.14135).
