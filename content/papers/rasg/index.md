---
title: "Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Use [Preprint]"
date: 2024-04-15
tags: ["Machine Learning", "Retrieval Augmented Generation", "Structured Generation", "Structured Prompting", "Supervised Finetuning", "Document Information Extraction"]
author: "Franz Louis Cesista"
description: "Business Document Information Extraction (BDIE) is the problem of transforming a blob of unstructured information (raw text, scanned documents, etc.) into a structured format that downstream systems can parse and use. It has two main tasks: Key-Information Extraction (KIE) and Line Items Recognition (LIR). And subtasks such as Optical Character Recognition (OCR) and Table Structure Recognition (TSR) are means to these ends. In this paper, we argue that BDIE is best modeled as a Tool Use problem, where the tools are these downstream systems. We then present Retrieval Augmented Structured Generation (RASG), a novel general framework for BDIE that achieves state of the art (SOTA) results on both KIE and LIR tasks on BDIE benchmarks.

The contributions of this paper are threefold: (1) We show, with ablation benchmarks, that Large Language Models (LLMs) with RASG are already competitive with or surpasses current SOTA Large Multi-Modal Models (LMMMs) without RASG such as LayoutLMv3 and Roberta + DeTR on BDIE benchmarks. (2) We propose a new metric class for Line Items Recognition, General Line Items Recognition Metric (GLIRM), that is more aligned with practical BDIE use cases compared to existing metrics, such as ANLS*, DocILE, and GriTS. (3) We provide a heuristic algorithm for backcalculating bounding boxes - that is, pairs of (x, y) coordinates containing relevant text of predicted line items and tables without the need for vision encoders. Finally, we claim that, while LMMMs might sometimes offer marginal performance benefits, LLMs + RASG is oftentimes superior given real-world applications and constraints of BDIE."
summary: "[Under Review] This paper presents Retrieval Augmented Structured Generation (RASG), a novel general framework for Business Document Information Extraction that achieves state of the art (SOTA) results on both Key-Information Extraction (KIE) and Line Items Recognition (LIR)"
---

Download: [Preprint](/RASG-ieee-mipr.pdf)

Authors: [Franz Louis Cesista](mailto:franzlouiscesista@gmail.com), [Rui Aguiar](mailto:rui@expedock.com), [Jason Kim](mailto:jasonminsookim@gmail.com), [Paolo Acilo](mailto:paolo@expedock.com)

---

## Abstract

Business Document Information Extraction (BDIE) is the problem of transforming a blob of unstructured information (raw text, scanned documents, etc.) into a structured format that downstream systems can parse and use. It has two main tasks: Key-Information Extraction (KIE) and Line Items Recognition (LIR). And subtasks such as Optical Character Recognition (OCR) and Table Structure Recognition (TSR) are means to these ends. In this paper, we argue that BDIE is best modeled as a *Tool Use* problem, where the tools are these downstream systems. We then present Retrieval Augmented Structured Generation (RASG), a novel general framework for BDIE that achieves state of the art (SOTA) results on both KIE and LIR tasks on BDIE benchmarks.

The contributions of this paper are threefold:

1. We show, with ablation benchmarks, that Large Language Models (LLMs) with RASG are already competitive with or surpasses current SOTA Large Multi-Modal Models (LMMMs) without RASG such as LayoutLMv3 and Roberta + DeTR on BDIE benchmarks.
2. We propose a new metric class for Line Items Recognition, General Line Items Recognition Metric (GLIRM), that is more aligned with practical BDIE use cases compared to existing metrics, such as ANLS*, DocILE, and GriTS.
3. We provide a heuristic algorithm for backcalculating bounding boxes - that is, pairs of (x, y) coordinates containing relevant text of predicted line items and tables without the need for vision encoders.

Finally, we claim that, while LMMMs might sometimes offer marginal performance benefits, LLMs + RASG is oftentimes superior given real-world applications and constraints of BDIE.
