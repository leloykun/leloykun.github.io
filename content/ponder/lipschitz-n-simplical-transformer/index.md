---
title: "Sensitivity and Sharpness of n-Simplicial Attention"
date: 2025-07-06
tags: ["Machine Learning", "Optimizers", "Architecture-Optimizer Codesign"]
author: "Franz Louis Cesista"
description: "Towards a maximal update parameterization of n-simplicial attention"
summary: "Towards a maximal update parameterization of n-simplicial attention"
# cover:
#     image: cover.jpg
#     alt: "Cover"
#     relative: true
editPost:
    URL: "https://x.com/leloykun/status/1943779237942768015"
    Text: "Crossposted on X (formerly Twitter)"
citation:
    title: "Sensitivity and Sharpness of n-Simplicial Attention"
    author:
        - "Franz Louis Cesista"
    publication_date: "2025/07/06"
---

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun


## Introduction

A team from Meta have recently shown that 2-simplicial attention improves the exponent in the scaling laws vs. vanilla attention ([Roy et al., 2025](https://arxiv.org/abs/2507.02754v1); [Clift et al., 2019](https://arxiv.org/abs/1909.00668), [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). This means that while it may be worse than vanilla attention flops-vs-loss-wise at smaller scales, the trade-off gets better and better at larger scales. This could also be useful for e.g. large-scale reasoning-LLM training runs where context lengths could blow up to millions, even billions of tokens. It is also very Bitter Lesson-pilled: compute exponentially scales over time and having a compute sponge which we can pour more compute into and get better results is great.

And if we are to scale this up, we have to consider two questions:
1. If 2-simplicial attention is better than (vanilla) 1-simplicial attention at scale, then would $n$-simplicial attention be better than 2-simplicial attention for $n \geq 3$?
2. How do we guarantee that our activation and gradient norms are 'stable' during training as we scale up the model?

In this blog post, we will focus on the latter, however we will consider $n$-simplicial attention in general in our analyses.

> **Definition 1 (n-Simplicial Attention):** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length, $d$ is the model width, and $n$ is the number of key-value pairs per token. And let $s_1, s_2 \in \mathbb{R}$ be scaling factors. Then we define n-simplicial attention $\texttt{F}$ as follows,
> $$\begin{aligned}
    \texttt{F}(q, k^{(1:n)}, v^{(1:n)})
        &= {\color{blue}s_2} \texttt{softmax}\left({\color{blue}s_1} \langle q, k^{(1)}, k^{(2)}, \ldots, k^{(n)} \rangle + \texttt{mask}\right) ( v^{(1)} \circ v^{(2)} \circ \ldots \circ v^{(n)} )\\
        &= {\color{blue}s_2} \texttt{softmax}\left({\color{blue}s_1} \left\langle q, \left( \prod_{t=1}^n \circ k^{(t)} \right) \right\rangle + \texttt{mask}\right) \left( \prod_{t=1}^n \circ v^{(t)} \right)
\end{aligned}$$
> where,
> * $\texttt{softmax}(\cdot): \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n+1}} \to \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n+1}}$ is applied to all indices except the first,
> * $\circ: \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n}\times d} \times \mathbb{R}^{\overbrace{T\times\ldots\times T}^{m}\times d} \to \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n+m}\times d}$ is the Hadamard (elementwise) product over the $d$-dimension, and
> * $\langle \cdot, \cdot \rangle: \mathbb{R}^{n \times d} \times \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n}\times d} \to \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n+1}}$ is the dot product over the $d$-dimension.
> 
> Note that the operation $\left(\prod_{t=1}^n \circ\right)$ above produces an $(n+1)$-dimensional tensor, $n$ from the sequence dimensions of the keys/values and one from the $d$-dimension. That is, we only reduce the last index.

Examples:
1. Vanilla Attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), $$\texttt{F}(q, k, v) = \texttt{softmax}\left(\frac{1}{\sqrt{d}} qk^T + \texttt{mask}\right) v$$
2. 2-Simplicial Attention ([Roy et al., 2025](https://arxiv.org/abs/2507.02754v1); [Clift et al., 2019](https://arxiv.org/abs/1909.00668)), $$\texttt{F}(q, k^{(1)}, k^{(2)}, v^{(1)}, v^{(2)}) = \texttt{softmax}\left(\frac{1}{\sqrt{d}} \langle q, k^{(1)}, k^{(2)} \rangle + \texttt{mask}\right) ( v^{(1)} \circ v^{(2)} )$$

Note that for both of these examples, $s_1 = 1/\sqrt{d}$ and $s_2 = 1$.

### Module sensitivity and sharpness

More formally, what we mean by activation norms being "stable" is that tiny changes in the inputs should not cause unexpectedly large changes in the outputs. We call this property *module sensitivity*. Likewise, we want the gradients to not blow up either, i.e. tiny changes in the inputs should not cause unexpectedly large changes in the gradients. We call this property *module sharpness*. And following [Large et al. (2024)](https://arxiv.org/abs/2405.14813), we formalize module sensitivity and sharpness as follows,

> **Definition 2 (Sensitivity):** Let $\texttt{M}$ be a module on $(\mathcal{X}, \mathcal{Y}, \mathcal{W})$ where $\mathcal{X}$ is the input space with norm $\|\cdot\|_{\mathcal{X}}$, $\mathcal{Y}$ is the output space with norm $\|\cdot\|_\mathcal{Y}$, and $\mathcal{W}$ is the parameter space. We define $\texttt{M}$ to be $\sigma$-sensitive if,
> $$\begin{equation}
    \| \nabla \texttt{M}(w, x) \diamond \Delta x \|_{\mathcal{Y}} \leq \sigma \| \Delta x \|_{\mathcal{X}} \qquad\forall w \in \mathcal{W}; x, \Delta x \in \mathcal{X}
\end{equation}$$

> **Definition 3 (Sharpness):** Let $\texttt{M}$ be a module on $(\mathcal{X}, \mathcal{Y}, \mathcal{W})$ where $\mathcal{X}$ is the input space with norm $\|\cdot\|_{\mathcal{X}}$, $\mathcal{Y}$ is the output space with norm $\|\cdot\|_\mathcal{Y}$, and $\mathcal{W}$ is the parameter space. We define $\texttt{M}$ to be $\gamma$-sharp if,
> $$\begin{equation}
    \| \tilde{\Delta} x \diamond \nabla^2 \texttt{M}(w, x) \diamond \Delta x \|_{\mathcal{Y}} \leq \gamma \| \Delta x \|_{\mathcal{X}} \| \tilde{\Delta} x \|_{\mathcal{X}} \qquad\forall w \in \mathcal{W}; x, \Delta x, \tilde{\Delta} x \in \mathcal{X}
\end{equation}$$

Note that if $\mathcal{X}$ and $\mathcal{Y}$ are normed vector spaces, then the sensitivity bounds the (forward) Lipschitz constant of the module, and the sharpness bounds the (backward) *gradient* Lipschitz constant. Having unit sensitivity means that a small change in the input can only cause at most as much change in the output. Likewise, having unit sharpness means that a small change in the input can only cause at most as much change in the gradient.

In this blog post, we will show that $n$-simplicial attention is unit sensitive and $3$-sharp under the $\infty RMS$ operator norm given that the inputs have unit RMS norm.

> **Claim 4 (Sensitivity and sharpness of n-Simplicial Attention):** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length, $d$ is the model width, and $n$ is the number of key-value pairs per token. For unit RMS norm inputs, i.e. $\| q \|_{\infty RMS} = \| k^{(t)} \|_{\infty RMS} = \| v^{(t)} \|_{\infty RMS} = 1$ for all $1 \leq t \leq n$, $n$-simplicial attention parameterized as follows,
$$\begin{equation}
    \texttt{F}(q, k^{(1:n)}, v^{(1:n)}) = {\color{blue}\frac{1}{d^{(n-1)/2}}} \texttt{softmax}\left({\color{blue}\frac{1}{d^{(n+1)/2}}} \left\langle q, \left( \prod_{t=1}^n \circ k^{(t)} \right) \right\rangle + \texttt{mask}\right) \left( \prod_{t=1}^n \circ v^{(t)} \right)
\end{equation}$$
> is unit sensitive and $3$-sharp under the $\infty RMS$ operator norm.

Note that for $n=1$, $s_1 = 1/d^{(1+1)/2} = 1/d$ and $s_2 = 1/d^{(1-1)/2} = 1$ which matches [Large et al.'s (2024)](https://arxiv.org/abs/2405.14813) parametrization, but not the standard parametrization discussed above.

## Preliminaries

### Choice of norms

In practice, we typically RMS-normalize activations before and/or after passing them through modules. Thus, it is natural to equip them with the $RMS$ norm. We also typically consider the maximum norm of the activations over a sequence of tokens to e.g. check for outliers or monitor stability. Thus, it is also natural to equip the sequence of activations with the $\infty RMS$ operator norm. More formally for $x \in \mathbb{R}^{T\times d}$,
$$\| x \|_{\infty RMS} = \max_{1 \leq i \leq T} \| x_i \|_{RMS} = \max_{1 \leq i \leq T} \frac{1}{\sqrt{d}} \sqrt{\sum_{j=1}^d x_{ij}^2}$$

More generally for $x \in \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n}\times d}$,
$$\| x \|_{\infty RMS} = \max_{I} \| x_I \|_{RMS} = \max_{I} \frac{1}{\sqrt{d}} \sqrt{\sum_{j=1}^d x_{Ij}^2}$$
where $I$ is a tuple of indices $I = (i_1, i_2, \ldots, i_n)$ and $1 \leq i_t \leq T$.

And because of the softmax, it is also natural to equip the $T \times T$ attention scores and probability matrices with the $\infty\text{-op}$ operator norm, which is defined as follows,
$$\| B \|_{\infty -op} = \max_{1 \leq i \leq T} \sum_{j=1}^T | B_{ij} |$$

For higher-order $\overbrace{T\times\ldots\times T}^{n}$ attention scores and probability tensors, we generalize this as,
$$\| B \|_{\infty -op} = \max_{1 \leq i \leq T} \sum_{J} | B_{iJ} |$$
where $J$ is a tuple of indices $J = (j_1, j_2, \ldots, j_n)$ and $1 \leq j_t \leq T$. We apply the sum over all indices except the first because we do the same for softmax.

Finally, observe that,
$$\| Bx \|_{\infty RMS} \leq \| B \|_{\infty -op} \| x \|_{\infty RMS}$$

### Choice of scaling factors

Let's rewrite n-Simplicial Attention in Claim 4 above as follows,

$$\begin{align}
    K &= \prod_{t=1}^n \circ k^{(t)} \\
    S &= s_1 \left\langle q, K \right\rangle \qquad & s_1 &= \frac{1}{d^{(n+1)/2}} \\
    A &= \texttt{softmax}\left(S + \texttt{mask}\right) \\
    V &= s_2 \prod_{t=1}^n \circ v^{(t)} \qquad & s_2 &= \frac{1}{d^{(n-1)/2}}\\
    \texttt{F}_{iJ} &= A_{iJ} V_J
\end{align}$$

We chose the scaling factor $s_2 = \frac{1}{d^{(n-1)/2}}$ so that $\| V \|_{\infty RMS} \leq 1$ for unit RMS norm values. This follows directly from Lemma 6 below. As for the scaling factor $s_1 = \frac{1}{d^{(n+1)/2}}$, we chose it so that the sensitivity and sharpness bounds we derive in our proofs below are width-independent.

> **Proposition 5 (RMS norm of hadamard product of vectors):** Let $x, y \in \mathbb{R}^d$ be vectors. Then the RMS norm of their hadamard product is bounded by the RMS norms of the individual vectors,
> $$\begin{equation}\| x \circ y \|_{RMS} \leq \sqrt{d} \| x \|_{RMS} \| y \|_{RMS} \end{equation}$$
> More generally for $x, y \in \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n} \times d}$,
> $$ \| x \circ y \|_{\infty RMS} \leq \sqrt{d} \| x \|_{\infty RMS} \| y \|_{\infty RMS} $$ 

{{< collapse summary="Show **proof of Proposition 5**" openByDefault=false >}}
> **Proof:** It is well-known that,
> $$\| x \circ y \|_2 \leq \| x \|_2 \| y \|_2$$
> This can be proven via Cauchy-Schwarz and Jensen's Lemma. Thus,
> $$\begin{aligned}
    \sqrt{d}\| x \circ y \|_{RMS} &\leq (\sqrt{d}\| x \|_{RMS}) \cdot (\sqrt{d}\| y \|_{RMS}) \\
    \| x \circ y \|_{RMS} &\leq \sqrt{d} \| x \|_{RMS} \| y \|_{RMS}
\end{aligned}$$
> For the more general case,
> $$\| x \circ y \|_{\infty RMS} = \max_{I} \| x_{I} \circ y_{I} \|_{RMS} \leq \max_{I} \sqrt{d} \| x_{I} \|_{RMS} \| y_{I} \|_{RMS} \leq \sqrt{d} \| x \|_{\infty RMS} \| y \|_{\infty RMS}  \quad\blacksquare$$
{{< /collapse >}}

> **Lemma 6 (RMS norm of hadamard product of *unit RMS norm* vectors):** Let $x^{(1)}, x^{(2)}, \ldots, x^{(n)} \in \mathbb{R}^d$ be vectors with $\| x^{(t)} \|_{RMS} \leq 1$ for all $1 \leq t \leq n$. Then,
> $$\begin{equation}\left\| \prod_{t=1}^n \circ x^{(t)} \right\|_{RMS} \leq d^{(n-1)/2}\end{equation}$$
> More generally for $x^{(1)}, x^{(2)}, \ldots, x^{(n)} \in \mathbb{R}^{\overbrace{T\times\ldots\times T}^{n} \times d}$ with $\| x^{(t)} \|_{\infty RMS} \leq 1$ for all $1 \leq t \leq n$,
> $$ \left\| \prod_{t=1}^n \circ x^{(t)} \right\|_{\infty RMS} \leq d^{(n-1)/2} $$

The proof follows directly from Proposition 5.

### Useful shorthands

Following [Large et al. (2024)](https://arxiv.org/abs/2405.14813), we use the following shorthand which is crucial for our proofs below.

> **Definition 7 (Bracket notation):** Let $B$ be a $\underbrace{T \times T \times \ldots \times T}_{n+1}$ tensor and $x$ be a $\underbrace{T \times T \times \ldots \times T}_{n}\times d$ tensor. Then,
> $$[B, x]_{iJ} := x_J - \sum_M B_{iM}x_{M}$$
> where $M \neq J$ is a tuple of indices $M = (m_1, m_2, \ldots, m_n)$ and $1 \leq m_t \leq T$

> **Proposition 8 (Crucial inequalities regarding $[B, x]$):** For any $\underbrace{T \times T \times \ldots \times T}_{n+1}$ tensor $B$ with non-negative entries and $\sum_J B_{iJ} = 1$ for all $1 \leq i \leq T$,
> $$\begin{aligned}
    \sum_{J} B_{iJ} \| [B, x]_{iJ} \| &\leq \max_J \| x_J \| \\
    \sum_{J} B_{iJ} \| [B, x]_{iJ} \|^2 &\leq \max_J \| x_J \|^2 \\
    \sum_{J} B_{iJ} \| [B, x]_{iJ} \| \| [B, y]_{iJ} \| &\leq (\max_J \| x_J \|)(\max_J \| y_J \|) \\
\end{aligned}$$
> All three inequalities can be proven via standard probability theory.

## Sensitivity of n-Simplicial Attention

We wish to show that the n-simplicial attention is unit sensitive for unit RMS norm inputs $(q, k^{(1:n)}, v^{(1:n)}) \in \mathcal{X}$.

> **Claim 9:** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length and $d$ is the model width. For $\| q \|_{\infty RMS} = \| k^{(t)} \|_{\infty RMS} = \| v^{(t)} \|_{\infty RMS} = 1$ for all $1 \leq t \leq n$, the n-simplicial attention function $\texttt{F}$ is unit sensitive under the $\infty RMS$ operator norm. That is, for any perturbation $(\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}) \in \mathcal{X}$, we have,
> $$\begin{aligned}
    \| \nabla \texttt{F} \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) \|_{\infty RMS}
        &\leq \| (\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}) \|_{\infty RMS} \\
        &\leq \| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta v^{(t)} \|_{\infty RMS}\\
\end{aligned}$$

To prove this, let's first take the derivative of $\texttt{F}$ towards $(\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)})$,

$$\begin{align}
    \nabla \texttt{F} \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) &= (\Delta A) V + A (\Delta V) \label{eq:ffirstdiv} \\
    \| \nabla \texttt{F} \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) \|_{\infty RMS}
        &\leq \| (\Delta A) V \|_{\infty RMS} + \| A (\Delta V) \|_{\infty RMS} \nonumber \\
        &\leq \| \Delta A \|_{\infty -op} \| V \|_{\infty RMS} + \| A \|_{\infty -op} \| \Delta V \|_{\infty RMS} \label{eq:ffirstdivnorm} \\
\end{align}$$

By construction we have,
$$\| V \|_{\infty RMS} \leq 1 \qquad\text{ and }\qquad \| A \|_{\infty -op} = 1$$
And so we only need to derive $\| \Delta A \|_{\infty -op}$ and $\| \Delta V \|_{\infty RMS}$.

---

To calculate the norm of $\Delta V$, we have by the product rule,

$$\begin{align}
    \Delta V
        &= \frac{1}{d^{(n-1)/2}}  \sum_{t=1}^{n} \left[ \Delta v^{(t)} \circ \left( \prod_{s=1,s\neq t}^n \circ v^{(s)} \right)\right] \label{eq:vfirstdiv}
\end{align}$$

Thus,

$$\begin{align}
    \| \Delta V \|_{\infty RMS}
        &\leq \frac{1}{d^{(n-1)/2}}  \sum_{t=1}^{n} \left\| \Delta v^{(t)} \circ \prod_{s=1,s\neq t}^n \circ v^{(s)} \right\|_{\infty RMS}\nonumber\\
        &\leq \cancel{\frac{d^{(n-1)/2}}{d^{(n-1)/2}}}  \sum_{t=1}^{n} \| \Delta v^{(t)} \|_{\infty RMS} \prod_{s=1,s\neq t}^n \left\| v^{(s)} \right\|_{\infty RMS} &\text{(from Proposition 5)}\nonumber\\
    \| \Delta V \|_{\infty RMS} &\leq \sum_{t=1}^{n} \| \Delta v^{(t)} \|_{\infty RMS} \label{eq:vfirstdivnorm}
\end{align}$$

Following the same calculation for $\Delta K$ yields,
$$\begin{equation}
    \| \Delta K \|_{\infty RMS} \leq d^{(n-1)/2} \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \label{eq:kfirstdivnorm}
\end{equation}$$

---

Following [Large et al. (2024)](https://arxiv.org/abs/2405.14813), a direct calculation of $\Delta A$ yields,

$$\begin{equation}
    \Delta A_{iJ}
        = \frac{1}{d^{(n+1)/2}} A_{iJ} (\langle \Delta q_i, [A, K]_{iJ} \rangle + \langle q_i, [A, \Delta K]_{iJ} \rangle) \label{eq:afirstdiv}
\end{equation}$$

Thus,

$$\begin{align}
    \| \Delta A_{iJ} \|_{\infty -op}
        &= \max_i \sum_J \| \Delta A_{iJ} \| \nonumber \\
        &= \max_i \sum_J \left| \frac{1}{d^{(n+1)/2}} A_{iJ} (\langle \Delta q_i, [A, K]_{iJ} \rangle + \langle q_i, [A, \Delta K]_{iJ} \rangle) \right| \nonumber \\
        &\leq \frac{1}{d^{(n+1)/2}} \max_i \sum_J A_{iJ} ( \| \Delta q_i \|_{2} \left\| [A, K]_{iJ} \right\|_{2} + \| q_i \|_{2} \left\| [A, \Delta K]_{iJ} \right\|_{2} ) \nonumber \\
        &\leq \frac{d}{d^{(n+1)/2}} \max_i \| \Delta q_i \|_{RMS} \sum_J A_{iJ} \left\| [A, K]_{iJ} \right\|_{RMS} \nonumber \\
            &\quad+ \frac{d}{d^{(n+1)/2}} \max_i \| q_i \|_{RMS} \sum_J A_{iJ} \left\| [A, \Delta K]_{iJ} \right\|_{RMS} \nonumber \\
        &\leq \frac{1}{d^{(n-1)/2}} \max_i \| \Delta q_i \|_{RMS} \max_J \left\| K_{J} \right\|_{RMS} \nonumber\\
            &\quad+ \frac{1}{d^{(n-1)/2}} \max_i \| q_i \|_{RMS} \max_J \left\| \Delta K_{J} \right\|_{RMS} \nonumber \\
        &\leq \frac{1}{d^{(n-1)/2}} \left(\| \Delta q \|_{\infty RMS} \left\| K \right\|_{\infty RMS} + \| q \|_{\infty RMS} \left\| \Delta K \right\|_{\infty RMS} \right) \\
        &\leq \cancel{\frac{d^{(n-1)/2}}{d^{(n-1)/2}}} \left(\| \Delta q \|_{\infty RMS}
            + \sum_{t=1}^{n} \left\| \Delta k^{(t)} \right\|_{\infty RMS} \right) \nonumber \\
    \| \Delta A_{iJ} \|_{\infty -op}
        &\leq \| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \label{eq:afirstdivnorm}
\end{align}$$
where the first inequality follows from Cauchy-Schwarz, the second from Proposition 5, the third from Proposition 8, and the second last from Lemma 6 and Inequality $\eqref{eq:kfirstdivnorm}$.

---

Combining Inequalities $\eqref{eq:ffirstdivnorm}$, $\eqref{eq:vfirstdivnorm}$, and $\eqref{eq:afirstdivnorm}$ then yields,

$$\begin{aligned}
    \| \nabla \texttt{F} \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle \|_{\infty RMS}
        &\leq \| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta v^{(t)} \|_{\infty RMS}\\
    \| \nabla \texttt{F} \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle \|_{\infty RMS}
        &\leq \| (q, k^{(1:n)}, v^{(1:n)}) \|_{\infty RMS}
\end{aligned}$$

Hence, n-simplicial attention is unit sensitive under the $\infty RMS$ operator norm as claimed.

## Sharpness of n-Simplicial Attention

Next, we wish to show that the n-simplicial attention is $3$-sharp for unit RMS norm inputs $(q, k^{(1:n)}, v^{(1:n)}) \in \mathcal{X}$. More formally,

> **Claim 10:** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length and $d$ is the model width. For $\| q \|_{\infty RMS} = \| k^{(t)} \|_{\infty RMS} = \| v^{(t)} \|_{\infty RMS} = 1$ for all $1 \leq t \leq n$, the n-simplicial attention function $\texttt{F}$ is $3$-sharp under the $\infty RMS$ operator norm. That is, for any pair of perturbations $(\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}), (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \in \mathcal{X}$, we have,
> $$\begin{aligned}
    &\| (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \diamond \nabla^2 \texttt{F} \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) \|_{\infty RMS}\\
        &\qquad\qquad \leq 3\| (\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}) \|_{\infty RMS} \| (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \|_{\infty RMS} \\
        &\qquad\qquad \leq 3\left(\| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} v^{(t)} \|_{\infty RMS}\right)\\
        &\qquad\qquad\qquad \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} v^{(t)} \|_{\infty RMS}\right)
\end{aligned}$$
> To simplify notation, let's define,
> $$\Delta^2 \texttt{F} := (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \diamond \nabla^2 \texttt{F} \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} )$$

To prove this, let's first take the derivative of Equation $\eqref{eq:ffirstdiv}$ towards $(\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)})$,

$$\begin{align}
    \Delta^2 \texttt{F}
        &= (\Delta^2 A) V + (\tilde{\Delta} A) (\Delta V) + (\Delta A) (\tilde{\Delta} V) + A (\Delta^2 V) \label{eq:fseconddiv} \\
    \| \Delta^2 \texttt{F}\|_{\infty RMS}
        &\leq \| (\Delta^2 A) V \|_{\infty RMS} + \| (\tilde{\Delta} A) (\Delta V) \|_{\infty RMS} \nonumber\\
        &\quad+ \| (\Delta A) (\tilde{\Delta} V) \|_{\infty RMS} + \| A (\Delta^2 V) \|_{\infty RMS} \nonumber\\
        &\leq \| \Delta^2 A \|_{\infty -op} \cancel{\| V \|_{\infty RMS}} + \| \tilde{\Delta} A \|_{\infty -op} \| \Delta V \|_{\infty RMS} \nonumber\\
        &\quad + \| \Delta A \|_{\infty -op} \| \tilde{\Delta} V \|_{\infty RMS} + \cancel{\| A \|_{\infty -op}} \| \Delta^2 V \|_{\infty RMS} \label{eq:fseconddivnorm} \\
\end{align}$$

We have already derived $\| \Delta A \|_{\infty -op}$ and $\| \Delta V \|_{\infty RMS}$ in the previous section. And for $\| \tilde{\Delta} A \|_{\infty -op}$ and $\| \tilde{\Delta} V \|_{\infty RMS}$, it would suffice to replace $\Delta$ with $\tilde{\Delta}$ in the previous derivations. Again, we also have $\| V \|_{\infty RMS} \leq 1$ and $ \| A \|_{\infty -op} = 1$ by construction. So, we only need to derive $\| \Delta^2 A \|_{\infty -op}$ and $\| \Delta^2 V \|_{\infty RMS}$.

---

To calculate the norm of the $\Delta^2 V$ term, let's take the derivative of Equation $\eqref{eq:vfirstdiv}$ towards $\tilde{\Delta}$,

$$\begin{align}
    \Delta^2 V
        &= \frac{1}{d^{(n-1)/2}}  \sum_{1 \leq t < s \leq n} \left[ \Delta v^{(t)} \circ \tilde{\Delta} v^{(s)} \circ \left( \prod_{r=1,r\neq t,r\neq s}^n \circ v^{(r)} \right)\right] \label{eq:vseconddiv}
\end{align}$$

Thus,

$$\begin{align}
    \| \Delta^2 V \|_{\infty RMS}
        &\leq \frac{1}{d^{(n-1)/2}}  \sum_{1 \leq t < s \leq n} \left\| \Delta v^{(t)} \circ \tilde{\Delta} v^{(s)} \circ \prod_{r=1,r\neq t,r\neq s}^n \circ v^{(r)} \right\|_{\infty RMS} \nonumber \\
        &\leq \frac{d}{d^{(n-1)/2}}  \sum_{1 \leq t < s \leq n} \| \Delta v^{(t)} \|_{\infty RMS} \| \tilde{\Delta} v^{(s)} \|_{\infty RMS} \left\| \prod_{r=1,r\neq t,r\neq s}^n \circ v^{(r)} \right\|_{\infty RMS} \nonumber \\
        &\leq \cancel{\frac{dd^{(n-3)/2}}{d^{(n-1)/2}}} \sum_{1 \leq t < s \leq n} \| \Delta v^{(t)} \|_{\infty RMS} \| \tilde{\Delta} v^{(s)} \|_{\infty RMS} \nonumber \\
    \| \Delta^2 V \|_{\infty RMS}
        &\leq \left( \sum_{t=1}^n \| \Delta v^{(t)} \|_{\infty RMS} \right) \left( \sum_{t=1}^n \| \tilde{\Delta} v^{(t)} \|_{\infty RMS} \right) \label{eq:vseconddivnorm}
\end{align}$$
where the second inequality follows from Proposition 5 and the third from Lemma 6.

---

To calculate the norm of $\Delta^2 A$, let's first take the derivative of Equation $\eqref{eq:afirstdiv}$ towards $\tilde{\Delta}$,
$$\begin{align}
    \Delta^2 A_{iJ}
        &= \frac{1}{d^{(n+1)/2}} A_{iJ} \langle \Delta q_i, [A, \tilde{\Delta} K]_{iJ} \rangle
            + \frac{1}{d^{(n+1)/2}} A_{iJ} \langle \tilde{\Delta} q_i, [ A, \Delta K]_{iJ} \rangle\nonumber\\
        &\quad+ \frac{1}{d^{(n+1)/2}} \tilde{\Delta} A_{iJ} \langle \Delta q_i, [A, K]_{iJ} \rangle
            + \frac{1}{d^{(n+1)/2}} \tilde{\Delta} A_{iJ} \langle q_i, [A, \Delta K]_{iJ} \rangle\nonumber\\
        &\quad+ \frac{1}{d^{(n+1)/2}} A_{iJ} \langle \Delta q_i, -\sum_M(\tilde{\Delta} A)_{iM}K_M \rangle
            + \frac{1}{d^{(n+1)/2}} A_{iJ} \langle q_i, -\sum_M(\tilde{\Delta} A)_{iM}\Delta K_M \rangle\nonumber
\end{align}$$

where $\tilde{\Delta} A$ is just Equation $\eqref{eq:afirstdiv}$, except we replace $\Delta$ with $\tilde{\Delta}$.

As for the first two terms, following our reasoning in the previous section yields,

$$\begin{align}
    &\| \text{ [term 1] } + \text{ [term 2] } \|_{\infty -op} \nonumber \\
        &\qquad\leq \max_i\sum_J \left|\frac{1}{d^{(n+1)/2}} A_{iJ} \langle \Delta q_i, [A, \tilde{\Delta} K]_{iJ} \rangle
            + \frac{1}{d^{(n+1)/2}} A_{iJ} \langle \tilde{\Delta} q_i, [ A, \Delta K]_{iJ} \rangle \right| \nonumber \\
        &\qquad\leq \frac{1}{d^{(n-1)/2}} \left(\| \Delta q \|_{\infty RMS} \| \tilde{\Delta} K \|_{\infty RMS}
            + \| \tilde{\Delta} q \|_{\infty RMS} \| \Delta K \|_{\infty RMS} \right)\nonumber \\
        &\qquad\leq \cancel{\frac{d^{(n-1)/2}}{d^{(n-1)/2}}}  \left( \| \tilde{\Delta} q \|_{\infty RMS} \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} + \| \Delta q \|_{\infty RMS} \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right) \nonumber \\
    &\| \text{ [term 1] } + \text{ [term 2] } \|_{\infty -op} \nonumber \\
        &\qquad\leq \left(\| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right)
\end{align}$$

As for the third term,

$$\begin{align}
    \| \text{ [term 3] } \|_{\infty -op}
        &\leq \max_i\sum_J \left|\frac{1}{d^{(n+1)/2}} (\tilde{\Delta} A_{iJ}) \langle \Delta q_i, [A, K]_{iJ} \rangle \right| \nonumber \\
        &= \frac{1}{d^{n+1}} \max_i\sum_J A_{iJ} \left| \left( \langle \tilde{\Delta} q_i, [A, K]_{iJ} \rangle + \langle q_i, [A, \tilde{\Delta} K]_{iJ} \rangle \right)\langle \Delta q_i, [A, K]_{iJ} \rangle \right| \nonumber \\
        &\leq \frac{1}{d^{n+1}} \max_i\sum_J A_{iJ} \| \tilde{\Delta} q_i \|_2 \| \Delta q_i \|_2 \| [A, K]_{iJ} \|_2^2 \nonumber \\
            &\quad+ \frac{1}{d^{n+1}} \max_i\sum_J A_{iJ}\| q_i \|_2 \| \Delta q_i \|_2 \| [A, \tilde{\Delta} K]_{iJ} \|_2 \| [A, K]_{iJ} \|_2 \nonumber \\
        &= \frac{d^2}{d^{n+1}} \max_i \| \tilde{\Delta} q_i \|_{RMS} \| \Delta q_i \|_{RMS} \sum_J A_{iJ} \| [A, K]_{iJ} \|_{RMS}^2 \nonumber \\
            &\quad+ \frac{d^2}{d^{n+1}}\max_i \| q_i \|_{RMS} \| \Delta q_i \|_{RMS} \sum_J A_{iJ} \| [A, \tilde{\Delta} K]_{iJ} \|_{RMS} \| [A, K]_{iJ} \|_{RMS}  \nonumber \\
        &\leq \frac{1}{d^{n-1}} \max_i \| \tilde{\Delta} q_i \|_{RMS} \| \Delta q_i \|_{RMS} \max_{J} \| K_{iJ} \|_{RMS}^2 \nonumber \\
            &\quad+ \frac{1}{d^{n-1}}\max_i \| q_i \|_{RMS} \| \Delta q_i \|_{RMS} (\max_{J} \| \tilde{\Delta} K_{J} \|_{RMS}) (\max_{J} \| K_{J} \|_{RMS}) \nonumber \\
        &\leq \frac{1}{d^{n-1}} \| \tilde{\Delta} q \|_{\infty RMS} \| \Delta q \|_{\infty RMS} \| K \|_{\infty RMS}^2 \nonumber \\
            &\quad+ \frac{1}{d^{n-1}} \| q \|_{\infty RMS} \| \Delta q \|_{\infty RMS} \| \tilde{\Delta} K \|_{\infty RMS} \| K \|_{\infty RMS} \nonumber \\
        &\leq \cancel{\frac{(d^{(n-1)/2})^2}{d^{n-1}}} \| \tilde{\Delta} q \|_{\infty RMS} \| \Delta q \|_{\infty RMS} \nonumber \\
            &\quad+ \cancel{\frac{(d^{(n-1)/2})^2}{d^{n-1}}} \| \Delta q \|_{\infty RMS} \sum_{t=1}^n \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \nonumber \\
    \| \text{ [term 3] } \|_{\infty -op}
        &\leq \| \Delta q \|_{\infty RMS} \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^n \| \tilde{\Delta} k^{(t)} \|_{\infty RMS}\right) \nonumber \\
\end{align}$$
where the third inequality follows from Proposition 8 and the fourth from Lemma 6 and Inequality $\eqref{eq:kfirstdivnorm}$.

Similarly for the fourth term,

$$\begin{align}
    \| \text{ [term 4] } \|_{\infty -op}
        &\leq \frac{1}{d^{n-1}} \| \tilde{\Delta} q \|_{\infty RMS} \| q \|_{\infty RMS} \| K \|_{\infty RMS} \| \Delta K \|_{\infty RMS} \nonumber\\
            &\quad+ \frac{1}{d^{n-1}} \| q \|_{\infty RMS}^2 \| \tilde{\Delta} K \|_{\infty RMS} \| \Delta K \|_{\infty RMS} \nonumber\\
        &\leq \left(\sum_{t=1}^n \| \Delta k^{(t)} \|_{\infty RMS} \right)\left( \| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^n \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right) \nonumber \\
\end{align}$$

Thus,

$$\begin{align}
    &\| \text{ [term 3] } + \text{ [term 4] } \|_{\infty -op} \nonumber \\
        &\qquad\leq \left(\| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right)
\end{align}$$

As for the fifth term, first observe that,

$$\begin{align}
    \max_i \left\| \sum_{M} (\tilde{\Delta} A)_{iM} K_M \right\|_{RMS}
        &\leq \max_i \sum_{M} |(\tilde{\Delta} A)_{iM}| \| K_M \|_{RMS} \nonumber\\
        &\leq \max_i \sum_{M} |(\tilde{\Delta} A)_{iM}| \left(\max_M \| K_M \|_{RMS}\right) \nonumber\\
        &= \| \tilde{\Delta} A \|_{\infty -op} \| K \|_{\infty RMS} \nonumber\\
        &\leq d^{(n-1)/2} \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS}\right) \nonumber
\end{align}$$

and that,

$$\begin{aligned}
    \max_i \left\| \sum_{M} (\tilde{\Delta} A)_{iM} \Delta K_M \right\|_{RMS}
        &\leq \| \tilde{\Delta} A \|_{\infty -op} \| \Delta K \|_{\infty RMS} \nonumber\\
        &\leq d^{(n-1)/2} \left(\sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS}\right)
\end{aligned}$$

Thus,
$$\begin{align}
    \| \text{ [term 5] } \|_{\infty -op}
        &= \max_i \sum_J \left\| \frac{1}{d^{(n+1)/2}} A_{iJ} \langle \Delta q_i, -\sum_M(\tilde{\Delta} A)_{iM}K_M \rangle \right\| \nonumber \\
        &\leq \frac{d}{d^{(n+1)/2}} \max_i \| \Delta q_i \|_{RMS} \left\| \sum_{M} (\tilde{\Delta} A)_{iM} K_M \right\|_{RMS} \cancel{\sum_J A_{iJ}} \nonumber \\
        &\leq \frac{1}{d^{(n-1)/2}} \| \Delta q \|_{\infty RMS} \| \tilde{\Delta} A \|_{\infty -op} \| K \|_{\infty RMS} \nonumber \\
        &\leq \cancel{\frac{d^{(n-1)/2}}{d^{(n-1)/2}}} \| \Delta q \|_{\infty RMS} \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS}\right) \nonumber
\end{align}$$

Similarly for the sixth term,

$$\begin{align*}
    \| \text{ [term 6] } \|_{\infty -op}
        &\leq \frac{1}{d^{(n-1)/2}} \| q \|_{\infty RMS} \| \tilde{\Delta} A \|_{\infty -op} \| \Delta K \|_{\infty RMS} \nonumber \\
        &\leq \left(\sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS}\right)
\end{align*}$$

Thus,

$$\begin{align}
    &\| \text{ [term 5] } + \text{ [term 6] } \|_{\infty -op} \nonumber \\
        &\qquad\leq \left(\| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right)
\end{align}$$

Taking them all together then yields,
$$\begin{equation}
    \| \Delta^2 A \|_{\infty -op} \leq 3\left(\| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left(\| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right) \label{eq:aseconddivnorm}
\end{equation}$$

---

Combining Inequalities $\eqref{eq:fseconddivnorm}$, $\eqref{eq:vseconddivnorm}$, and $\eqref{eq:aseconddivnorm}$ then yields,

$$\begin{align}
    \| \Delta^2 \texttt{F} \|_{\infty RMS}
        &\leq 3 \left( \| \Delta q \|_{\infty RMS} + \sum_{t=1}^n \| \Delta k^{(t)} \|_{\infty RMS} \right) \left( \| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^n \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right) \nonumber\\
        &\qquad + \left( \| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} \right) \left( \sum_{t=1}^{n} \| \Delta v^{(t)} \|_{\infty RMS} \right) \nonumber \\
        &\qquad + \left( \| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} \right) \left( \sum_{t=1}^{n} \| \tilde{\Delta} v^{(t)} \|_{\infty RMS} \right) \nonumber \\
        &\qquad + \left( \sum_{t=1}^n \| \Delta v^{(t)} \|_{\infty RMS} \right) \left( \sum_{t=1}^n \| \tilde{\Delta} v^{(t)} \|_{\infty RMS} \right) \nonumber \\
        &\leq 3 \left( \| \Delta q \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta k^{(t)} \|_{\infty RMS} + \sum_{t=1}^{n} \| \Delta v^{(t)} \|_{\infty RMS} \right) \nonumber \\
        &\qquad \left( \| \tilde{\Delta} q \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} k^{(t)} \|_{\infty RMS} + \sum_{t=1}^{n} \| \tilde{\Delta} v^{(t)} \|_{\infty RMS} \right) \nonumber \\
    \| \Delta^2 \texttt{F} \|_{\infty RMS}
        &\leq 3 \| (\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}) \|_{\infty RMS} \| (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \|_{\infty RMS}
\end{align}$$

Hence, n-simplicial attention is $3$-sharp under the $\infty RMS$ operator norm as claimed.

## Discussion

Here we have devised a parametrization that allows us to have width-independent sensitivity and sharpness bounds for n-simplicial attention. We hope that this will allow us to construct a maximum update parametrization of some sort for such modules and networks containing them.

Note however that for $n = 1$, we have to set the scaling factor $s_1 = \frac{1}{d^{(1+1)/2}} = \frac{1}{d}$, which is the same scaling factor suggested by [Large et al. (2024)](https://arxiv.org/abs/2405.14813), but is different from the more standard $s_1 = \frac{1}{\sqrt{d}}$. Likewise, for 2-simplicial attention, we have to set the scaling factor $s_1 = \frac{1}{d^{(2+1)/2}} = \frac{1}{d^{3/2}}$, which is different from the $s_1 = \frac{1}{\sqrt{d}}$ used by [Roy et al. (2025)](https://arxiv.org/abs/2507.02754v1). Additionally, we also have to set $s_2 = \frac{1}{d^{(2-1)/2}} = \frac{1}{\sqrt{d}}$ for the outer scale in 2-simplicial attention, which, for larger dimensions, scales down the outputs significantly. Empirically, such parametrization leads to worse performance early in training, but guarantees stable training, especially at the tail end of training where the queries, keys, and values are more often aligned than not.

The main benefit of having low (and width-independent) sensitivity and sharpness really is that it allows us to have larger update step sizes without worrying about suddenly exploding or vanishing activations and gradients. Additionally, bounding the sensitivity allows us to control how much the gradients change as they pass through the module via backpropagation--the smaller the sensitivity, the smaller the change in the gradients. And bounding the sharpness allows us to have more trust in the momentum term more knowing that gradient spikes would rarely happen, if at all. These gradient spikes notoriously 'break' the momentum term at larger traning runs, especially near the end of training.

This parametrization could also be useful in distributed training setups where gradient all-reduces are expensive and thus sparsifying the gradients before sending them over the network is a must ([Douillard et al., 2024](https://arxiv.org/abs/2311.08105); [Thérien et al., 2025](https://arxiv.org/abs/2505.23725)). Problem arises when the gradients have outliers, requiring us to use more expensive quantization schemes to avoid losing information. But having control over the gradient norms should allow us to eliminate such outliers and get low-precision (and thus low-communication) training basically "for free".

Lastly, this could also be used to parametrize a continuous n-simplicial attention module, where $n$ is continous instead of discrete. At test time, we could then scale $n$ as a sort of test-time scaling.

## How to Cite

```bibtex
@misc{cesista2025sensitivitysharpnessnsimplicalattention,
  author = {Franz Louis Cesista},
  title = {{S}ensitivity and {S}harpness of n-{S}implicial {A}ttention},
  year = {2025},
  month = {July},
  day = {6},
  url = {https://leloykun.github.io/ponder/lipschitz-n-simplical-transformer/},
}
```

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

## References

1. Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil (2025). Fast and Simplex: 2-Simplicial Attention in Triton. URL https://arxiv.org/abs/2507.02754v1
2. James Clift, Dmitry Doryn, Daniel Murfet, James Wallbridge (2019). Logic and the 2-Simplicial Transformer. URL https://arxiv.org/abs/1909.00668
3. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017). Attention is all you need. URL https://arxiv.org/abs/1706.03762
4. Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein (2024). Scalable Optimization in the Modular Norm. URL https://arxiv.org/abs/2405.14813
5. Benjamin Thérien, Xiaolong Huang, Irina Rish, Eugene Belilovsky (2025). MuLoCo: Muon is a practical inner optimizer for DiLoCo. URL https://arxiv.org/abs/2505.23725
6. Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen (2024). DiLoCo: Distributed Low-Communication Training of Language Models. URL https://arxiv.org/abs/2311.08105
