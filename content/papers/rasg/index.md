---
title: "Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Use"
date: 2024-04-15
tags: ["Machine Learning", "Retrieval Augmented Generation", "Structured Generation", "Structured Prompting", "Supervised Finetuning", "Document Information Extraction"]
author: "Franz Louis Cesista, Rui Aguiar, Jason Kim, Paolo Acilo"
description: "Business Document Information Extraction (BDIE) is the problem of transforming a blob of unstructured information (raw text, scanned documents, etc.) into a structured format that downstream systems can parse and use. It has two main tasks: Key-Information Extraction (KIE) and Line Items Recognition (LIR). In this paper, we argue that BDIE is best modeled as a Tool Use problem, where the tools are these downstream systems. We then present Retrieval Augmented Structured Generation (RASG), a novel general framework for BDIE that achieves state of the art (SOTA) results on both KIE and LIR tasks on BDIE benchmarks.

The contributions of this paper are threefold: (1) We show, with ablation benchmarks, that Large Language Models (LLMs) with RASG are already competitive with or surpasses current SOTA Large Multimodal Models (LMMs) without RASG on BDIE benchmarks. (2) We propose a new metric class for Line Items Recognition, General Line Items Recognition Metric (GLIRM), that is more aligned with practical BDIE use cases compared to existing metrics, such as ANLS*, DocILE, and GriTS. (3) We provide a heuristic algorithm for backcalculating bounding boxes of predicted line items and tables without the need for vision encoders. Finally, we claim that, while LMMs might sometimes offer marginal performance benefits, LLMs + RASG is oftentimes superior given real-world applications and constraints of BDIE."
summary: "[IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR) 2024] This paper presents Retrieval Augmented Structured Generation (RASG), a novel general framework for Business Document Information Extraction that achieves state of the art (SOTA) results on both Key-Information Extraction (KIE) and Line Items Recognition (LIR)."
cover:
    image: cover.png
    alt: "Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Use"
    relative: true
citation:
    title: "Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Use"
    author:
        - "Franz Louis Cesista"
        - "Rui Aguiar"
        - "Jason Kim"
        - "Paolo Acilo"
    publication_date: "2024/04/15"
---

Authors: [Franz Louis Cesista](mailto:franzlouiscesista@gmail.com), Rui Aguiar, Jason Kim, Paolo Acilo

Links to paper:

- On IEEE: https://ieeexplore.ieee.org/document/10708044
- On Arxiv: https://arxiv.org/abs/2405.20245

---

## Abstract

Business Document Information Extraction (BDIE) is the problem of transforming a blob of unstructured information (raw text, scanned documents, etc.) into a structured format that downstream systems can parse and use. It has two main tasks: Key-Information Extraction (KIE) and Line Items Recognition (LIR). In this paper, we argue that BDIE is best modeled as a Tool Use problem, where the tools are these downstream systems. We then present Retrieval Augmented Structured Generation (RASG), a novel general framework for BDIE that achieves state of the art (SOTA) results on both KIE and LIR tasks on BDIE benchmarks.

The contributions of this paper are threefold:

1. We show, with ablation benchmarks, that Large Language Models (LLMs) with RASG are already competitive with or surpasses current SOTA Large Multimodal Models (LMMs) without RASG on BDIE benchmarks.
2. We propose a new metric class for Line Items Recognition, General Line Items Recognition Metric (GLIRM), that is more aligned with practical BDIE use cases compared to existing metrics, such as ANLS*, DocILE, and GriTS.
3. We provide a heuristic algorithm for backcalculating bounding boxes of predicted line items and tables without the need for vision encoders.

Finally, we claim that, while LMMs might sometimes offer marginal performance benefits, LLMs + RASG is oftentimes superior given real-world applications and constraints of BDIE.

## Citation

```bibtex
@INPROCEEDINGS{cesista2024rasg,
  author={Cesista, Franz Louis and Aguiar, Rui and Kim, Jason and Acilo, Paolo},
  booktitle={2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR)}, 
  title={Retrieval Augmented Structured Generation: Business Document Information Extraction as Tool Use}, 
  year={2024},
  volume={},
  number={},
  pages={227-230},
  keywords={Measurement;Large language models;Heuristic algorithms;Information processing;Benchmark testing;Predictive models;Information retrieval;Prediction algorithms;Data mining;Business;document information extraction;key information extraction;line items recognition;retrieval augmented generation;structured generation;table detection},
  doi={10.1109/MIPR62202.2024.00042}
}
```
