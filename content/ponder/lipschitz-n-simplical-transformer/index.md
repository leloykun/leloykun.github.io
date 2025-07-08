---
title: "Sensitivity and Sharpness of n-Simplical Attention"
date: 2025-07-06
tags: ["Machine Learning", "Optimizers", "Architecture-Optimizer Codesign"]
author: "Franz Louis Cesista"
description: "Towards a maximal update parameterization of n-simplical attention"
summary: "Towards a maximal update parameterization of n-simplical attention"
# cover:
#     image: cover.jpg
#     alt: "Cover"
#     relative: true
# editPost:
#     URL: "https://x.com/leloykun/status/1941067659157913625"
#     Text: "Crossposted on X (formerly Twitter)"
---

## Introduction

A team from Meta have recently shown that 2-simplical attention improves the exponent in the scaling laws vs. vanilla attention (Roy et al., 2025; Clift et al., 2019, Vaswani et al., 2017). This means that while it may not be as good or even worse than vanilla attention flops-vs-loss-wise at smaller scales, the trade-off gets better as the model scales up. This would be useful in e.g. large-scale reasoning-LLM training runs where context lengths could blow up to millions, even billions of tokens. It is also very Bitter Lesson-pilled: compute exponentially scales over time and having a compute sponge which we can pour more compute into and get better results is great.

And if we are to scale this up, we have to consider two questions:
1. If 2-simplical attention is better than (vanilla) 1-simplical attention at scale, then would $n$-simplical attention be better than 2-simplical attention for $n \geq 3$?
2. How do we guarantee that our activation and gradient norms are 'stable' during training as we scale up the model?

In this blog post, we will focus on the latter, however we will consider $n$-simplical attention in general in our analyses.

> **Definition 1 (n-Simplical Attention):** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length and $d$ is the model width. The n-simplical attention function $\texttt{F}$ is defined as follows,
> $$\begin{aligned}
    \texttt{F}(q, k^{(1:n)}, v^{(1:n)})
        &= {\color{blue}s_2} \texttt{softmax}\left({\color{blue}s_1} \langle q, k^{(1)}, k^{(2)}, \ldots, k^{(n)} \rangle + M\right) ( v^{(1)} \circ v^{(2)} \circ \ldots \circ v^{(n)} )\\\\
        &= {\color{blue}s_2} \texttt{softmax}\left({\color{blue}s_1} \left\langle q, \left( \prod\_{t=1}^n \circ k^{(t)} \right) \right\rangle + M\right) \left( \prod\_{t=1}^n \circ v^{(t)} \right)
\end{aligned}$$
> where $\texttt{softmax}(\cdot)$ is applied to all indices except the first.
>
> Examples:
> 1. Vanilla Attention (Vaswani et al., 2017), $$\texttt{F}(q, k, v) = \texttt{softmax}\left(\frac{1}{\sqrt{d}} qk^T + M\right) v$$
> 2. 2-Simplical Attention (Clift et al., 2019), $$\texttt{F}(q, k^{(1)}, k^{(2)}, v^{(1)}, v^{(2)}) = \texttt{softmax}\left(\frac{1}{\sqrt{d}} \langle q, k^{(1)}, k^{(2)} \rangle + M\right) ( v^{(1)} \circ v^{(2)} )$$
> 
> Note that for both of these examples, $s_1 = 1/\sqrt{d}$ and $s_2 = 1$.

And more formally, what we mean by activation norms being "stable" is that tiny changes in the inputs should not cause unexpectedly large changes in the outputs. We call this property *module sensitivity*. Likewise, we want the gradients to not blow up either, i.e. tiny changes in the inputs should not cause unexpectedly large changes in the gradients. We call this property *module sharpness*. And following Large et al. (2024), we formalize module sensitivity and sharpness as follows,

> **Definition 2 (Sensitivity):** Let $M$ be a module on $(\mathcal{X}, \mathcal{Y}, \mathcal{W})$ where $\mathcal{X}$ is the input space with norm $\\|\cdot\\|\_{\mathcal{X}}$, $\mathcal{Y}$ is the output space with norm $\\|\cdot\\|\_\mathcal{Y}$, and $\mathcal{W}$ is the parameter space. We define $M$ to be $\sigma$-sensitive if,
> $$\begin{equation}
    \\| \nabla M(w, x) \diamond \Delta x \\|\_{\mathcal{Y}} \leq \sigma \\| \Delta x \\|\_{\mathcal{X}} \qquad\forall w \in \mathcal{W}; x, \Delta x \in \mathcal{X}
\end{equation}$$

> **Definition 3 (Sharpness):** Let $M$ be a module on $(\mathcal{X}, \mathcal{Y}, \mathcal{W})$ where $\mathcal{X}$ is the input space with norm $\\|\cdot\\|\_{\mathcal{X}}$, $\mathcal{Y}$ is the output space with norm $\\|\cdot\\|\_\mathcal{Y}$, and $\mathcal{W}$ is the parameter space. We define $M$ to be $\gamma$-sharp if,
> $$\begin{equation}
    \\| \tilde{\Delta} x \diamond \nabla^2 M(w, x) \diamond \Delta x \\|\_{\mathcal{Y}} \leq \gamma \\| \Delta x \\|\_{\mathcal{X}} \\| \tilde{\Delta} x \\|\_{\mathcal{X}} \qquad\forall w \in \mathcal{W}; x, \Delta x, \tilde{\Delta} x \in \mathcal{X}
\end{equation}$$

Note that if $\mathcal{X}$ and $\mathcal{Y}$ are normed vector spaces, then the sensitivity bounds the (forward) Lipschitz constant of the module, and the sharpness bounds the (backward) *gradient* Lipschitz constant. Having unit sensitivity means that a small change in the input can only cause at most as much change in the output. Likewise, having unit sharpness means that a small change in the input can only cause at most as much change in the gradient.

In this blog post, we will show that $n$-simplical attention is unit sensitive and $(1 + \tilde{L}\_{\texttt{softmax}})$-sharp under the $\infty RMS$ operator norm, where $\tilde{L}\_{\texttt{softmax}}$ is the *gradient* Lipschitz constant of the softmax function.

> **Claim 4 (Sensitivity and sharpness of n-Simplical Attention):** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length and $d$ is the model width. $n$-simplical attention parameterized as follows,
$$\begin{equation}
    \texttt{F}(q, k^{(1:n)}, v^{(1:n)}) = {\color{blue}\frac{1}{d^{(n-1)/2}}} \texttt{softmax}\left({\color{blue}\frac{1}{d^{(n+1)/2}}} \left\langle q, \left( \prod\_{t=1}^n \circ k^{(t)} \right) \right\rangle + M\right) \left( \prod\_{t=1}^n \circ v^{(t)} \right)
\end{equation}$$
> is unit sensitive and $(1 + \tilde{L}\_{\texttt{softmax}})$-sharp under the $\infty RMS$ operator norm where $d$ is the model width and $\tilde{L}\_{\texttt{softmax}}$ is the *gradient* Lipschitz constant of the softmax function.

## Preliminaries

First, let's rewrite n-Simplical Attention in Claim 4 above as follows,

$$\begin{align}
    S &= s_1 \left\langle q, \left( \prod\_{t=1}^n \circ k^{(t)} \right) \right\rangle \qquad & s_1 &= \frac{1}{d^{(n+1)/2}} \\\\
    A &= \texttt{softmax}\left(S + M\right) \\\\
    W &= s_2 \prod\_{t=1}^n \circ v^{(t)} \qquad & s_2 &= \frac{1}{d^{(n-1)/2}}\\\\
    F &= A W
\end{align}$$

We chose the scaling factor $s_2 = \frac{1}{d^{(n-1)/2}}$ so that $\\| W \\|\_{\infty RMS} \leq 1$ for unit RMS norm values. This follows directly from Lemma 6 below. As for the scaling factor $s_1 = \frac{1}{d^{(n+1)/2}}$, we chose it so that the entries of $S$ are bounded by $1$ (see Lemma 7), making (masked) softmax 1-Lipschitz. This property is crucial for our proofs later on.

> **Proposition 5 (RMS norm of hadamard product of vectors):** Let $x, y \in \mathbb{R}^d$ be vectors. Then the RMS norm of their hadamard product is bounded by the RMS norms of the individual vectors,
> $$\begin{equation}\\| x \circ y \\|\_{RMS} \leq \sqrt{d} \\| x \\|\_{RMS} \\| y \\|\_{RMS} \end{equation}$$

{{< collapse summary="Show **proof of Proposition 5**" openByDefault=false >}}
> **Proof:**
> $$\begin{aligned}
    \left\\| x \circ y \right\\|\_{RMS}^2
        &= \left\\| x \circ y \right\\|\_{RMS}^2 \\\\
        &= \left(\frac{1}{\sqrt{d}}\right)^2\left\\| x \circ y \right\\|\_{2}^2 \\\\
        &= \frac{1}{d} \sum\_{r=1}^d (x_r)^2 (y_r)^2 \\\\
        &\leq \frac{1}{d} \left(\sum\_{r=1}^d (x_r)^4 \right)^{1/2} \left(\sum\_{r=1}^d (y_r)^4\right)^{1/2} &\text{(from Cauchy-Schwarz)}\\\\
        &\leq \frac{1}{d} \left(\sum\_{r=1}^d (x_r)^2 \right) \left(\sum\_{r=1}^d (y_r)^2\right)  &\text{(from Jensen's Lemma)}\\\\
        &\leq \frac{1}{d} \\| x_r \\|\_2^2 \\| y_r \\|\_2^2 \\\\
    \left\\| x \circ y \right\\|\_{RMS}^2
        &\leq d \\| x \\|\_{RMS}^2 \\| y \\|\_{RMS}^2 \\\\
    \left\\| x \circ y \right\\|\_{RMS}
        &\leq \sqrt{d} \\| x \\|\_{RMS}^2 \\| y \\|\_{RMS}^2 \quad\blacksquare
\end{aligned}$$
{{< /collapse >}}

> **Lemma 6 (RMS norm of hadamard product of *unit RMS norm* vectors):** Let $x^{(1)}, x^{(2)}, \ldots, x^{(n)} \in \mathbb{R}^d$ be vectors with $\\| x^{(t)} \\|\_{RMS} \leq 1$ for all $t$. Then,
> $$\begin{equation}\left\\| \prod\_{t=1}^n \circ x^{(t)} \right\\|\_{RMS} \leq d^{(n-1)/2}\end{equation}$$

The proof follows directly from Proposition 5.

> **Lemma 7:** For unit RMS norm query $q$ and keys $k^{(1:n)}$, the choice of scaling factor $s_1 = \frac{1}{d^{(n+1)/2}}$ bounds the entries of $S$ by $1$.

{{< collapse summary="Show **proof of Lemma 7**" openByDefault=false >}}
> **Proof:** From Lemma 6, we have,
> $$\\| w(n) \\|\_{\infty RMS} := \left\\| \prod_{t=1}^n \circ k^{(t)} \right\\|\_{\infty RMS} \leq d^{(n-1)/2}$$
> Thus,
> $$\begin{aligned}
    | \langle q, w(n) \rangle |
        &\leq \\| q \\|\_2 \\| w(n) \\|\_2  \\\\
        &\leq (\sqrt{d} \cancel{\\| q \\|\_{\infty RMS}})( \sqrt{d} \\| w(n) \\|\_{\infty RMS}) \\\\
        &\leq d d^{(n-1)/2} \\\\
    | \langle q, w(n) \rangle |
        &= d^{(n+1)/2}
\end{aligned}$$
> Thus the entries of $S$ are bounded by,
> $$| S_{i,J} | = \frac{1}{d^{(n+1)/2}} | \langle q_i, w(n)_J \rangle | \leq \frac{1}{d^{(n+1)/2}} d^{(n+1)/2} = 1 \quad\blacksquare$$
{{< /collapse >}}

## Sensitivity of n-Simplical Attention

We wish to show that the n-simplical attention is unit sensitive for unit RMS norm inputs $(q, k^{(1:n)}, v^{(1:n)}) \in \mathcal{X}$.

> **Claim 8:** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length and $d$ is the model width. For $\\| q \\|\_{\infty RMS} = \\| k^{(t)} \\|\_{\infty RMS} = \\| v^{(t)} \\|\_{\infty RMS} = 1$ for all $t$, the n-simplical attention function $\texttt{F}$ is unit sensitive under the $\infty RMS$ operator norm. That is, for any perturbation $(\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}) \in \mathcal{X}$, we have,
> $$\begin{aligned}
    \\| \nabla F \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) \\|\_{\infty RMS}
        &\leq \\| (\Delta q, \Delta k^{[1:n]}, \Delta v^{[1:n]}) \\|\_{\infty RMS} \\\\
        &\leq \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} + \sum\_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS}\\\\
\end{aligned}$$

To prove this, let's first take the derivative of $\texttt{F}$ towards $(\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)})$,

$$\begin{align}
    \nabla F \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) &= (\Delta A) W + A (\Delta W) \\\\
    \\| \nabla F \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) \\|\_{\infty RMS}
        &\leq \\| \Delta A \\|\_{\infty RMS} \\| W \\|\_{\infty RMS} + \\| A \\|\_{\infty RMS} \\| \Delta W \\|\_{\infty RMS}\\\\
\end{align}$$

We have already shown earlier that,
$$\\| W \\|\_{\infty RMS} = \\| A \\|\_{\infty RMS} = 1$$
by construction. And so we only need to derive $\\| \Delta A \\|\_{\infty RMS}$ and $\\| \Delta W \\|\_{\infty RMS}$.

---

As for the $\Delta A$ term, by the chain-rule, we have,
$$\begin{equation}\\| \Delta A \\|\_{\infty RMS} \leq \\| \nabla \texttt{softmax}(S)[\Delta S] \\|\_{\infty -op} \\| \Delta S \\|\_{\infty RMS}\end{equation}$$
and since the softmax is 1-Lipschitz with respect to the $\infty RMS$ norm with our parameterization, we have,
$$\\| \Delta A \\|\_{\infty RMS} \leq \\| \Delta S \\|\_{\infty RMS}$$

By the product rule, we have,

$$\begin{align}
    \Delta S
        &= \frac{1}{d^{(n+1)/2}} \left\langle \Delta q, \left( \prod\_{t=1}^n \circ k^{(t)} \right) \right\rangle + \frac{1}{d^{(n+1)/2}}  \sum_{t=1}^{n} \left\langle q, \Delta k^{(t)} \circ \left( \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right) \right\rangle \\\\
    \Delta S
        &= \frac{1}{d^{(n+1)/2}} \left\langle \Delta q, \left( \prod\_{t=1}^n \circ k^{(t)} \right) \right\rangle + \frac{1}{d^{(n+1)/2}}  \sum_{t=1}^{n} \left\langle \Delta k^{(t)}, q \circ \left( \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right) \right\rangle \nonumber\\\\
\end{align}$$

Thus,

$$\begin{aligned}
    \\| \Delta S \\|\_{\infty RMS}
        &\leq \frac{1}{d^{(n+1)/2}} \\| \Delta q \\|\_{2} \left \\| \prod\_{t=1}^n \circ k^{(t)} \right\\|\_{2} + \frac{d}{d^{(n+1)/2}} \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{2} \left\\| q \circ \prod_{s=1,s\neq t}^n \circ k^{(s)} \right\\|\_{2}\\\\
        &\leq \frac{d}{d^{(n+1)/2}} \\| \Delta q \\|\_{\infty RMS} \left \\| \prod\_{t=1}^n \circ k^{(t)} \right\\|\_{\infty RMS}\\\\
        &\quad + \frac{d}{d^{(n+1)/2}} \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \left\\| q \circ \prod_{s=1,s\neq t}^n \circ k^{(s)} \right\\|\_{\infty RMS}\\\\
        &\leq \cancel{\frac{dd^{(n-1)/2}}{d^{(n+1)/2}}} \left( \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \right)\\\\
    \\| \Delta S \\|\_{\infty RMS} &\leq  \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS}
\end{aligned}$$

Thus, $$\begin{equation}
    \\| \Delta A \\|\_{\infty RMS} \leq \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS}
\end{equation}$$

---

As for the $\Delta W$ term, by the product rule, we have,

$$\begin{align}
    \Delta W
        &= \frac{1}{d^{(n-1)/2}}  \sum_{t=1}^{n} \left[ \Delta v^{(t)} \circ \left( \prod\_{s=1,s\neq t}^n \circ v^{(s)} \right)\right]
\end{align}$$

Thus,

$$\begin{align}
    \\| \Delta W \\|\_{\infty RMS}
        &\leq \frac{1}{d^{(n-1)/2}}  \sum\_{t=1}^{n} \left\\| \Delta v^{(t)} \circ \prod\_{s=1,s\neq t}^n \circ v^{(s)} \right\\|\_{\infty RMS}\nonumber\\\\
        &\leq \frac{\sqrt{d}}{d^{(n-1)/2}}  \sum\_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS} \left\\| \prod\_{s=1,s\neq t}^n \circ v^{(s)} \right\\|\_{\infty RMS} &\text{(from Proposition 5)}\nonumber\\\\
        &\leq \cancel{\frac{d^{1/2}d^{(n-2)/2}}{d^{(n-1)/2}}}  \sum\_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS}&\text{(from Lemma 6)}\nonumber\\\\
    \\| \Delta W \\|\_{\infty RMS} &= \sum\_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS}
\end{align}$$

---

Combining Equations (11), (14), and (16) then gives us,

$$\begin{aligned}
    \\| \nabla F \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle \\|\_{\infty RMS}
        &\leq \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} + \sum\_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS}\\\\
    \\| \nabla F \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle \\|\_{\infty RMS}
        &\leq \\| (q, k^{[1:n]}, v^{[1:n]}) \\|\_{\infty RMS}
\end{aligned}$$

Hence, n-simplical attention is unit sensitive under the $\infty RMS$ operator norm as claimed.

## Sharpness of n-Simplical Attention

Next, we wish to show that the n-simplical attention is $(1+\tilde{L}\_{\texttt{softmax}})$-sharp for unit RMS norm inputs $(q, k^{(1:n)}, v^{(1:n)}) \in \mathcal{X}$. More formally,

> **Claim 9:** Let $q, k^{(1:n)}, v^{(1:n)} \in \mathbb{R}^{T \times d}$ be the query, keys, and values, where $T$ is the sequence length and $d$ is the model width. For $\\| q \\|\_{\infty RMS} = \\| k^{(t)} \\|\_{\infty RMS} = \\| v^{(t)} \\|\_{\infty RMS} = 1$ for all $t$, the n-simplical attention function $\texttt{F}$ is unit sensitive under the $\infty RMS$ operator norm. That is, for any pair of perturbations $(\Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)}), (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \in \mathcal{X}$, we have,
> $$\begin{aligned}
    &\\| (\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)}) \diamond \nabla F \diamond ( \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} ) \\|\_{\infty RMS}\\\\
        &\qquad\qquad \leq (1+\tilde{L}\_{\texttt{softmax}})\\| (\Delta q, \Delta k^{[1:n]}, \Delta v^{[1:n]}) \\|\_{\infty RMS} \\| (\tilde{\Delta} q, \tilde{\Delta} k^{[1:n]}, \tilde{\Delta} v^{[1:n]}) \\|\_{\infty RMS} \\\\
        &\qquad\qquad \leq (1+\tilde{L}\_{\texttt{softmax}})\left(\\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} + \sum\_{t=1}^{n} \\| \tilde{\Delta} v^{(t)} \\|\_{\infty RMS}\right)\\\\
        &\qquad\qquad\qquad\qquad\qquad\quad \times \left(\\| \tilde{\Delta} q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} + \sum\_{t=1}^{n} \\| \tilde{\Delta} v^{(t)} \\|\_{\infty RMS}\right)
\end{aligned}$$

To prove this, let's first take the derivative of Equation (10) towards $(\tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)})$,

$$\begin{align}
    &\langle \tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)} \rangle \diamond \nabla^2 F \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle\nonumber\\\\
        &\qquad\qquad= (\Delta^2 A) W + (\tilde{\Delta} A) (\Delta W) + (\Delta A) (\tilde{\Delta} W) + A (\Delta^2 W) \\\\
    &\\| \langle \tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)} \rangle \diamond \nabla^2 F \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle \\|\_{\infty RMS}\nonumber\\\\
        &\qquad\qquad\leq \\| \Delta^2 A \\|\_{\infty RMS} \cancel{\\| W \\|\_{\infty RMS}} + \\| \tilde{\Delta} A \\|\_{\infty RMS} \\| \Delta W \\|\_{\infty RMS} \nonumber\\\\
        &\qquad\qquad\quad + \\| \Delta A \\|\_{\infty RMS} \\| \tilde{\Delta} W \\|\_{\infty RMS} + \cancel{\\| A \\|\_{\infty RMS}} \\| \Delta^2 W \\|\_{\infty RMS} \\\\
\end{align}$$

We have already derived $\\| \Delta A \\|\_{\infty RMS}$ and $\\| \Delta W \\|\_{\infty RMS}$ in the previous section. And for $\\| \tilde{\Delta} A \\|\_{\infty RMS}$ and $\\| \tilde{\Delta} W \\|\_{\infty RMS}$, it would suffice to replace $\Delta$ with $\tilde{\Delta}$ in the previous derivations. Again, we also have $\\| W \\|\_{\infty RMS} = \\| A \\|\_{\infty RMS} = 1$ by construction. So, we only need to derive $\\| \Delta^2 A \\|\_{\infty RMS}$ and $\\| \Delta^2 W \\|\_{\infty RMS}$.

---

For the $\Delta^2 A$ term, let's take the derivative of Equation (12) towards $\tilde{\Delta}$,

$$\begin{aligned}
    \\| \Delta^2 A \\|\_{\infty RMS}
        &\leq \\| \nabla^2 \texttt{softmax}(S)[\Delta S, \tilde{\Delta} S] \\|\_{\infty RMS} \\| \Delta S \\|\_{\infty RMS} \\| \tilde{\Delta} S \\|\_{\infty RMS}\\\\
        &\quad+ \cancel{\\| \nabla \texttt{softmax}(S)[\Delta S] \\|\_{\infty RMS}} \\| \Delta^2 S \\|\_{\infty RMS}\\\\
    \\| \Delta^2 A \\|\_{\infty RMS}
        &\leq \tilde{L}\_{\texttt{softmax}} \\| \Delta S \\|\_{\infty RMS} \\| \tilde{\Delta} S \\|\_{\infty RMS} + \\| \Delta^2 S \\|\_{\infty RMS}\\\\
\end{aligned}$$

We have already derived $\\| \Delta S \\|\_{\infty RMS}$ in the previous section. And for $\\| \tilde{\Delta} S \\|\_{\infty RMS}$, it would suffice to replace $\Delta$ with $\tilde{\Delta}$ in the previous derivation. So, we only need to derive $\\| \Delta^2 S \\|\_{\infty RMS}$. Applying the product rule to Equation (13), we have,

$$\begin{align}
    \Delta^2 S
        &= \frac{1}{d^{(n+1)/2}} \sum\_{t=1}^n\left\langle \Delta q, \tilde{\Delta} k^{(t)} \circ \left( \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right) \right\rangle \nonumber\\\\
        &\quad + \frac{1}{d^{(n+1)/2}}  \sum_{t=1}^{n} \left\langle \tilde{\Delta} q, \Delta k^{(t)} \circ \left( \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right) \right\rangle \nonumber\\\\
        &\quad + \frac{1}{d^{(n+1)/2}}  \sum_{1 \leq t < s \leq n} \left\langle q, \Delta k^{(t)} \circ \tilde{\Delta} k^{(s)} \circ \left( \prod\_{r=1,r\neq t,r\neq s}^n \circ k^{(r)} \right) \right\rangle
\end{align}$$

Thus,

$$\begin{aligned}
    \\| \Delta^2 S \\|\_{\infty RMS}
        &\leq \frac{1}{d^{(n+1)/2}} \\| \Delta q \\|\_{2} \sum\_{t=1}^n \left\\| \tilde{\Delta} k^{(t)} \circ \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right\\|\_{2} \\\\
        &\quad + \frac{1}{d^{(n+1)/2}} \\| \tilde{\Delta} q \\|\_{2} \sum_{t=1}^{n} \left\\| \Delta k^{(t)} \circ \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right\\|\_{2} \\\\
        &\quad + \frac{1}{d^{(n+1)/2}} \\| q \\|\_{2} \sum_{1 \leq t < s \leq n} \left\\| \Delta k^{(t)} \circ \tilde{\Delta} k^{(s)} \circ \prod\_{r=1,r\neq t,r\neq s}^n \circ k^{(r)} \right\\|\_{2} \\\\
        &\leq \frac{d^{3/2}}{d^{(n+1)/2}} \\| \Delta q \\|\_{\infty RMS} \sum\_{t=1}^n \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} \left\\| \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right\\|\_{\infty RMS} \\\\
        &\quad + \frac{d^{3/2}}{d^{(n+1)/2}} \\| \tilde{\Delta} q \\|\_{\infty RMS} \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \left\\| \prod\_{s=1,s\neq t}^n \circ k^{(s)} \right\\|\_{\infty RMS} \\\\
        &\quad + \frac{d^{2}}{d^{(n+1)/2}} \cancel{\\| q \\|\_{\infty RMS}} \sum_{1 \leq t < s \leq n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \\| \tilde{\Delta} k^{(s)} \\|\_{\infty RMS} \left\\| \prod\_{r=1,r\neq t,r\neq s}^n \circ k^{(r)} \right\\|\_{\infty RMS} \\\\
        &\leq \cancel{\frac{d^{3/2}d^{(n-2)/2}}{d^{(n+1)/2}}} \\| \Delta q \\|\_{\infty RMS} \sum\_{t=1}^n \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} \\\\
        &\quad + \cancel{\frac{d^{3/2}d^{(n-2)/2}}{d^{(n+1)/2}}} \\| \tilde{\Delta} q \\|\_{\infty RMS} \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \\\\
        &\quad + \cancel{\frac{d^{2}d^{(n-3)/2}}{d^{(n+1)/2}}} \sum_{1 \leq t < s \leq n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \\| \tilde{\Delta} k^{(s)} \\|\_{\infty RMS} \\\\
    \\| \Delta^2 S \\|\_{\infty RMS}
        &\leq \left( \\| \Delta q \\|\_{\infty RMS} + \sum\_{t=1}^n \\| \Delta k^{(t)} \\|\_{\infty RMS} \right) \left( \\| \tilde{\Delta} q \\|\_{\infty RMS} + \sum\_{t=1}^n \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} \right)
\end{aligned}$$

Thus,

$$\begin{align}
    \\| \Delta^2 A \\|\_{\infty RMS} &\leq (1+\tilde{L}\_{\texttt{softmax}}) \left( \\| \Delta q \\|\_{\infty RMS} + \sum\_{t=1}^n \\| \Delta k^{(t)} \\|\_{\infty RMS} \right) \left( \\| \tilde{\Delta} q \\|\_{\infty RMS} + \sum\_{t=1}^n \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} \right)
\end{align}$$

---

For the $\Delta^2 W$ term, let's take the derivative of Equation (15) towards $\tilde{\Delta}$,

$$\begin{align}
    \Delta^2 W
        &= \frac{1}{d^{(n-1)/2}}  \sum_{1 \leq t < s \leq n} \left[ \Delta v^{(t)} \circ \tilde{\Delta} v^{(s)} \circ \left( \prod\_{r=1,r\neq t,r\neq s}^n \circ v^{(r)} \right)\right]
\end{align}$$

$$\begin{align}
    \\| \Delta^2 W \\|\_{\infty RMS}
        &\leq \frac{1}{d^{(n-1)/2}}  \sum_{1 \leq t < s \leq n} \left\\| \Delta v^{(t)} \circ \tilde{\Delta} v^{(s)} \circ \prod\_{r=1,r\neq t,r\neq s}^n \circ v^{(r)} \right\\|\_{\infty RMS} \nonumber\\\\
        &\leq \frac{d}{d^{(n-1)/2}}  \sum_{1 \leq t < s \leq n} \\| \Delta v^{(t)} \\|\_{\infty RMS} \\| \tilde{\Delta} v^{(s)} \\|\_{\infty RMS} \left\\| \prod\_{r=1,r\neq t,r\neq s}^n \circ v^{(r)} \right\\|\_{\infty RMS} \nonumber\\\\
        &\leq \cancel{\frac{dd^{(n-3)/2}}{d^{(n-1)/2}}} \sum_{1 \leq t < s \leq n} \\| \Delta v^{(t)} \\|\_{\infty RMS} \\| \tilde{\Delta} v^{(s)} \\|\_{\infty RMS} \nonumber\\\\
        &\leq \left( \sum\_{t=1}^n \\| \Delta v^{(t)} \\|\_{\infty RMS} \right) \left( \sum\_{t=1}^n \\| \tilde{\Delta} v^{(t)} \\|\_{\infty RMS} \right)
\end{align}$$

---

Combining Equations (18), (20), and (22) then gives us,

$$\begin{aligned}
    &\\| \langle \tilde{\Delta} q, \tilde{\Delta} k^{(1:n)}, \tilde{\Delta} v^{(1:n)} \rangle \diamond \nabla^2 F \diamond \langle \Delta q, \Delta k^{(1:n)}, \Delta v^{(1:n)} \rangle \\|\_{\infty RMS} \\\\
        &\quad \leq (1 + \tilde{L}\_{\texttt{softmax}}) \left( \\| \Delta q \\|\_{\infty RMS} + \sum\_{t=1}^n \\| \Delta k^{(t)} \\|\_{\infty RMS} \right) \left( \\| \tilde{\Delta} q \\|\_{\infty RMS} + \sum\_{t=1}^n \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} \right)\\\\
        &\qquad + \left( \\| \tilde{\Delta} q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} \right) \left( \sum\_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS} \right) \\\\
        &\qquad + \left( \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} \right) \left( \sum\_{t=1}^{n} \\| \tilde{\Delta} v^{(t)} \\|\_{\infty RMS} \right) \\\\
        &\qquad + \left( \sum\_{t=1}^n \\| \Delta v^{(t)} \\|\_{\infty RMS} \right) \left( \sum\_{t=1}^n \\| \tilde{\Delta} v^{(t)} \\|\_{\infty RMS} \right) \\\\
        &\quad \leq (1 + \tilde{L}\_{\texttt{softmax}}) \left( \\| \Delta q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta k^{(t)} \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \Delta v^{(t)} \\|\_{\infty RMS} \right) \\\\
        &\qquad\qquad\qquad\qquad \left( \\| \tilde{\Delta} q \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \tilde{\Delta} k^{(t)} \\|\_{\infty RMS} + \sum_{t=1}^{n} \\| \tilde{\Delta} v^{(t)} \\|\_{\infty RMS} \right)
\end{aligned}$$

Hence, n-simplical attention is $(1+\tilde{L}\_{\texttt{softmax}})$-sharp under the $\infty RMS$ operator norm as claimed.

## Discussion

Here we have devised a parametrization that allows us to have width-independent sensitivity and sharpness bounds for n-simplical attention. We hope that this will allow us to construct a maximum update parametrization of some sort for such modules and networks containing them.

Note however that for $n = 1$, we have to set the scaling factor $s_1 = \frac{1}{d^{(1+1)/2}} = \frac{1}{d}$, which is the same scaling factor suggested by Large et al. (2024), but is different from the more standard $s_1 = \frac{1}{\sqrt{d}}$. Likewise, for 2-simplical attention, we have to set the scaling factor $s_1 = \frac{1}{d^{(2+1)/2}} = \frac{1}{d^{3/2}}$, which is different from the $s_1 = \frac{1}{\sqrt{d}}$ used by Roy et al. (2025). Additionally, we also have to set $s_2 = \frac{1}{d^{(2-1)/2}} = \frac{1}{\sqrt{d}}$ for the outer scale in 2-simplical attention, which, for larger dimensions, scales down the outputs significantly. Empirically, such parametrization leads to worse performance early in training, but guarantees stable training, especially at the tail end of training where the queries, keys, and values are more often aligned than not.

The main benefit of having low (and width-independent) sensitivity and sharpness really is that it allows us to have larger update step sizes without worrying about suddenly exploding or vanishing activations and gradients. Additionally, bounding the sensitivity allows us to control how much the gradients change as they pass through the module via backpropagation--the smaller the sensitivity, the smaller the change in the gradients. And bounding the sharpness allows us to have more trust in the momentum term more knowing that gradient spikes would rarely happen, if at all. These gradient spikes notoriously 'break' the momentum term at larger traning runs, especially near the end of training.

Lastly, this could also be useful in distributed training setups where gradient all-reduces are expensive and thus sparsifying the gradients before sending them over the network is a must (Douillard et al., 2024; Thérien et al., 2025). Problem arises when the gradients have outliers, requiring us to use more expensive quantization schemes to avoid losing information. But having control over the gradient norms should allow us to eliminate such outliers and get low-precision (and thus low-communication) training basically "for free".

## References

1. Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil (2025). Fast and Simplex: 2-Simplicial Attention in Triton. URL https://arxiv.org/abs/2507.02754v1
2. James Clift, Dmitry Doryn, Daniel Murfet, James Wallbridge (2019). Logic and the -Simplicial Transformer. URL https://arxiv.org/abs/1909.00668
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. URL https://arxiv.org/abs/1706.03762
4. Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein (2024). Scalable Optimization in the Modular Norm. URL https://arxiv.org/abs/2405.14813
5. Benjamin Thérien, Xiaolong Huang, Irina Rish, Eugene Belilovsky (2025). MuLoCo: Muon is a practical inner optimizer for DiLoCo. URL https://arxiv.org/abs/2505.23725
6. Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen (2024). DiLoCo: Distributed Low-Communication Training of Language Models. URL https://arxiv.org/abs/2311.08105
