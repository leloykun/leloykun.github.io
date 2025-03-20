---
title: "CASPR Without Accumulation is Muon"
date: 2025-02-13
tags: ["Machine Learning", "Muon"]
author: "Franz Louis Cesista"
description: "The CASPR optimizer, a variant of Shampoo, reduces to Muon when we remove the accumulation on the preconditioners."
summary: "The CASPR optimizer, a variant of Shampoo, reduces to Muon when we remove the accumulation on the preconditioners."
editPost:
    URL: "https://x.com/leloykun/status/1889996276512796855"
    Text: "Crossposted from X (formerly Twitter)"
---

CASPR is a variant of the Shampoo optimizer that finds different preconditioners for each axis of the parameter.

## CASPR's update rule

CASPR updates the parameters using the following update rule:

$$L_t := L_{t-1} + G_t G_t^T, \quad\quad R_t := R_{t-1} + G_t G_t^T$$
$$\tilde{L}_t := L_t + \epsilon I_m, \quad\quad \tilde{R}_t := R_t + \epsilon I_n$$
$$\Delta W = (\tilde{L}^{-1/2}_t G_t + 2 \tilde{L}^{-1/4}_t G_t \tilde{R}^{-1/4}_t + G_t \tilde{R}^{-1/2}_t)/4.$$

## CASPR without accumulation

If we turn off accumulation on the preconditioners, we get:
$$\Delta W = (1/4) \cdot ((G G^T + \epsilon I_m)^{-1/2} G + 2 (G G^T + \epsilon I_m)^{-1/4} G (G^T G + \epsilon I_n)^{-1/4} + G (G^T G + \epsilon I_n)^{-1/2})$$

To simplify this, let's first decompose $G$ via its singular value decomposition (SVD): $$G = U \Sigma V^T,$$ where $U$ and $V$ are orthogonal matrices and $\Sigma$ is a diagonal matrix with the singular values of $G$.

Thus,
$$\begin{aligned}
    \Delta W &= (1/4) \cdot [[(U \Sigma V^T) (U \Sigma V^T)^T + \epsilon U I U^T]^{-1/2}(U \Sigma V^T)\\\\
        &\quad\quad\quad + 2[(U \Sigma V^T) (U \Sigma V^T)^T + \epsilon U I U^T]^{-1/4}(U \Sigma V^T) [(U \Sigma V^T)^T (U \Sigma V^T) + \epsilon V I V^T]^{-1/4}\\\\
        &\quad\quad\quad + (U \Sigma V^T) [(U \Sigma V^T)^T (U \Sigma V^T) + \epsilon V I V^T]^{-1/2}]\\\\
             &= (1/4) \cdot [U(\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T + U(2\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T + U(\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T]\\\\
             &= (1/4) \cdot [U(1+2+1)(\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T]\\\\
             &= U\left(\frac{\Sigma}{\sqrt{\Sigma^2 + \epsilon I}} \right)V^T \\\\
    \Delta W &\approx UV^T
\end{aligned}$$
which is just the update rule for Muon.

## How to cite

```bibtex
@misc{cesista2025casprmuon,
  author = {Franz Louis Cesista},
  title = {{CASPR} Without Accumulation is {M}uon},
  year = {2025},
  url = {https://leloykun.github.io/ponder/caspr-wo-accum-is-muon/},
}
```

## References

1. Surya, S., Duvvuri, Devvrit, F., Anil, R., Hsieh, C., & Dhillon, I.S. (2024). Combining Axes Preconditioners through Kronecker Approximation for Deep Learning. International Conference on Learning Representations.
2. Vineet Gupta, Tomer Koren, Yoram Singer (2018). Shampoo: Preconditioned Stochastic Tensor Optimization. URL https://arxiv.org/abs/1802.09568
3. Rohan Anil et al. “Scalable second order optimization for deep learning.” arXiv preprint arXiv:2002.09018 (2020).
4. Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza Vlado, Jiacheng You, Franz Cesista, Braden Koszarsky, and @Grad62304977. modded-nanogpt: Speedrunning the NanoGPT baseline. 2024. Available at: https://github.com/KellerJordan/modded-nanogpt.
5. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/.
