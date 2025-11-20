---
title: "Training Transformers with Enforced Lipschitz Bounds"
date: 2025-07-17
tags: ["Machine Learning", "Optimizers", "Architecture-Optimizer Codesign", "Muon"]
author: "Laker Newhouse* and R. Preston Hess* and Franz Cesista* and Andrii Zahorodnii and Jeremy Bernstein and Phillip Isola"
description: "Neural networks are often highly sensitive to input and weight perturbations. This sensitivity has been linked to pathologies such as vulnerability to adversarial examples, divergent training, and overfitting. To combat these problems, past research has looked at building neural networks entirely from Lipschitz components. However, these techniques have not matured to the point where researchers have trained a modern architecture such as a transformer with a Lipschitz certificate enforced beyond initialization. To explore this gap, we begin by developing and benchmarking novel, computationally-efficient tools for maintaining norm-constrained weight matrices. Applying these tools, we are able to train transformer models with Lipschitz bounds enforced throughout training. We find that optimizer dynamics matter: switching from AdamW to Muon improves standard methods -- weight decay and spectral normalization -- allowing models to reach equal performance with a lower Lipschitz bound. Inspired by Muon's update having a fixed spectral norm, we co-design a weight constraint method that improves the Lipschitz vs. performance tradeoff on MLPs and 2M parameter transformers. Our 2-Lipschitz transformer on Shakespeare text reaches validation accuracy 60%. Scaling to 145M parameters, our 10-Lipschitz transformer reaches 21% accuracy on internet text. However, to match the NanoGPT baseline validation accuracy of 39.4%, our Lipschitz upper bound increases to 10^264. Nonetheless, our Lipschitz transformers train without stability measures such as layer norm, QK norm, and logit tanh softcapping."
summary: "Neural networks are often highly sensitive to input and weight perturbations. This sensitivity has been linked to pathologies such as vulnerability to adversarial examples, divergent training, and overfitting. To combat these problems, past research has looked at building neural networks entirely from Lipschitz components. However, these techniques have not matured to the point where researchers have trained a modern architecture such as a transformer with a Lipschitz certificate enforced beyond initialization. To explore this gap, we begin by developing and benchmarking novel, computationally-efficient tools for maintaining norm-constrained weight matrices. Applying these tools, we are able to train transformer models with Lipschitz bounds enforced throughout training. We find that optimizer dynamics matter: switching from AdamW to Muon improves standard methods -- weight decay and spectral normalization -- allowing models to reach equal performance with a lower Lipschitz bound. Inspired by Muon's update having a fixed spectral norm, we co-design a weight constraint method that improves the Lipschitz vs. performance tradeoff on MLPs and 2M parameter transformers. Our 2-Lipschitz transformer on Shakespeare text reaches validation accuracy 60%. Scaling to 145M parameters, our 10-Lipschitz transformer reaches 21% accuracy on internet text. However, to match the NanoGPT baseline validation accuracy of 39.4%, our Lipschitz upper bound increases to 10^264. Nonetheless, our Lipschitz transformers train without stability measures such as layer norm, QK norm, and logit tanh softcapping."
# cover:
#     image: cover.png
#     alt: "Training Transformers with Enforced Lipschitz Bounds"
#     relative: true
citation:
    title: "Training Transformers with Enforced Lipschitz Bounds"
    author:
        - "Laker Newhouse"
        - "R. Preston Hess"
        - "Franz Cesista"
        - "Andrii Zahorodnii"
        - "Jeremy Bernstein"
        - "Phillip Isola"
    publication_date: "2025/07/17"
    journal: "arXiv"
    pdf: "https://arxiv.org/pdf/2507.13338"
---

Authors: Laker Newhouse* and R. Preston Hess* and [Franz Cesista*](mailto:franzlouiscesista@gmail.com) and Andrii Zahorodnii and Jeremy Bernstein and Phillip Isola

Arxiv: [Abstract](https://arxiv.org/abs/2507.13338)

Code on GitHub: https://github.com/Arongil/lipschitz-transformers

---

## Abstract

Neural networks are often highly sensitive to input and weight perturbations. This sensitivity has been linked to pathologies such as vulnerability to adversarial examples, divergent training, and overfitting. To combat these problems, past research has looked at building neural networks entirely from Lipschitz components. However, these techniques have not matured to the point where researchers have trained a modern architecture such as a transformer with a Lipschitz certificate enforced beyond initialization. To explore this gap, we begin by developing and benchmarking novel, computationally-efficient tools for maintaining norm-constrained weight matrices. Applying these tools, we are able to train transformer models with Lipschitz bounds enforced throughout training. We find that optimizer dynamics matter: switching from AdamW to Muon improves standard methods -- weight decay and spectral normalization -- allowing models to reach equal performance with a lower Lipschitz bound. Inspired by Muon's update having a fixed spectral norm, we co-design a weight constraint method that improves the Lipschitz vs. performance tradeoff on MLPs and 2M parameter transformers. Our 2-Lipschitz transformer on Shakespeare text reaches validation accuracy 60%. Scaling to 145M parameters, our 10-Lipschitz transformer reaches 21% accuracy on internet text. However, to match the NanoGPT baseline validation accuracy of 39.4%, our Lipschitz upper bound increases to 10^264. Nonetheless, our Lipschitz transformers train without stability measures such as layer norm, QK norm, and logit tanh softcapping.

## Citation

```bibtex
@misc{newhouse2025training,
    title={{T}raining {T}ransformers with {E}nforced {L}ipschitz {B}ounds},
    author={Laker Newhouse and R. Preston Hess and Franz Cesista and Andrii Zahorodnii and Jeremy Bernstein and Phillip Isola},
    year={2025},
    eprint={2507.13338},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
