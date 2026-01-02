---
title: "Sensitivity and Sharpness of Gated Linear Attention Mechanisms"
date: 2026-01-01
tags: ["Machine Learning", "Linear Attention", "Test-Time Regression"]
author: "Franz Louis Cesista"
description: "We derive sensitivity and sharpness bounds for Gated DeltaNet and Mamba 2, showing that they can be made 1-Lipschitz with appropriate parameter constraints."
summary: "We derive sensitivity and sharpness bounds for Gated DeltaNet and Mamba 2, showing that they can be made 1-Lipschitz with appropriate parameter constraints."
draft: false
---

## Introduction

In [Ponder: Sensitivity and Sharpness of n-Simplicial Attention](../lipschitz-n-simplical-transformer/), we derived the sensitivity and sharpness bounds for n-Simplicial attention, a generalization of the classic softmax attention that makes attention 'denser', in a sense, by attending to *tuples* of keys instead of individual keys ([Roy et al., 2025](https://arxiv.org/abs/2507.02754v1); [Clift et al., 2019](https://arxiv.org/abs/1909.00668), [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). Here, we derive similar sensitivity and sharpness bounds for the other end of the attention mechanism spectrum: the 'sparser', *linear* attention mechanisms, specifically Gated DeltaNet ([Yang et. al., 2025](https://arxiv.org/abs/2412.06464)) and Mamba 2 ([Dao et. al., 2024](https://proceedings.mlr.press/v235/dao24a.html)). These linear attention mechanisms are particularly interesting because they can be computed in linear time with respect to the sequence length, making them suitable for long-sequence modeling tasks. We also show that both Gated DeltaNet and Mamba 2 can be made 1-Lipschitz by appropriately constraining their learnable parameters.

> Recommended reading:
> - [Ponder: Sensitivity and Sharpness of n-Simplicial Attention](../lipschitz-n-simplical-transformer/)
> - [Ponder: Block Matrix Formulation of Linear Attention Mechanisms](../blockmat-linear-attn/)
> - [Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813)

## Sensitivity and Sharpness of Gated DeltaNet

> **Theorem 1 (Sensitivity and Sharpness of Gated DeltaNet).** Let $T$ be the sequence length, $d$ be the model width, and $q, k, v \in \mathbb{R}^{T \times d}$ be the query, key, and value sequences, respectively, $\text{RMS}$-normalized such that $\| q_t \|_{RMS}, \| k_t \|_{RMS}, \| v_t \|_{RMS} \leq 1$ for all $t$. Also let the initial state $S_0 = 0$, and $\alpha_t, \beta_t \in \mathbb{R}$ be learnable 'decay' and 'step size' parameters, respectively, such that $0 \leq \alpha_t \leq \alpha < 1$, and $0 \leq \beta_t \leq \beta < 2$ for some constants $\alpha, \beta > 0$. Then Gated DeltaNet ([Yang et. al., 2025](https://arxiv.org/abs/2412.06464)) with the following update rule:
$$\begin{align}
    A_t
        &= \alpha_t \left( I - \frac{\beta_t}{d} k_t k_t^T \right) \\
    B_t
        &= \frac{\beta_t}{d} v_t k_t^T \\
    S_t
        &= S_{t-1} A_t + B_t \\
    \texttt{F}_t
        &= S_t q_t
\end{align}$$
has the following sensitivity $\sigma$ and sharpness $\gamma$ bounds:
$$\begin{align}
    \sigma
        &\leq \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \label{eq:gdn-sensitivity} \\
    \gamma
        &\leq \max\Bigg\{
                \frac{6 \alpha \beta^2}{(1 - \alpha)^2} + \frac{8 \alpha^2 \beta^3}{(1 - \alpha)^3},
                \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2}
            \Bigg\} \label{eq:gdn-sharpness}
\end{align}$$
such that for any perturbations $(\Delta q, \Delta k, \Delta v)$ and $(\tilde{\Delta} q, \tilde{\Delta} k, \tilde{\Delta} v)$,
$$\begin{align}
    \| \nabla \texttt{F} \diamond (\Delta q, \Delta k, \Delta v) \|_{\infty-RMS}
        &\leq \sigma \left( \| \Delta q \|_{\infty-RMS} + \| \Delta k \|_{\infty-RMS} + \| \Delta v \|_{\infty-RMS} \right) \\
\end{align}$$
and,
$$\begin{align}
    &\| (\tilde{\Delta} q, \tilde{\Delta} k, \tilde{\Delta} v) \diamond \nabla^2 \texttt{F} \diamond ( \Delta q, \Delta k, \Delta v ) \|_{\infty-RMS} \nonumber \\
        &\qquad\leq \gamma \left( \| \Delta q \|_{\infty-RMS} + \| \Delta k \|_{\infty-RMS} + \| \Delta v \|_{\infty-RMS} \right) \nonumber \\
            &\qquad\qquad\times \left( \| \tilde{\Delta} q \|_{\infty-RMS} + \| \tilde{\Delta} k \|_{\infty-RMS} + \| \tilde{\Delta} v \|_{\infty-RMS} \right)
\end{align}$$

Note that the sensitivity and sharpness bounds above are independent of the sequence length $T$ and model width $d$.

**Proof.** It suffices to first prove the sensitivity and sharpness bounds for arbitrary time step $t$, and then extend the results to the entire sequence by taking the maximum over all time steps. To simplify notation, let $\Delta Q = \| q \|_{\infty-RMS}, \Delta K = \| k \|_{\infty-RMS}, \Delta V = \| v \|_{\infty-RMS}$.

---

To prove the sensitivity bound at time step $t$, we first take the derivative of $F_t$ towards $(\Delta q_t, \Delta k_t, \Delta v_t)$,
$$\begin{align}
    \Delta F_t
        &= \nabla \texttt{F}_t \diamond (\Delta q_t, \Delta k_t, \Delta v_t) \nonumber \\
        &= \Delta S_t q_t + S_t \Delta q_t \label{eq:F_t-derivative} \\
    \| \Delta F_t \|_{2}
        &\leq \| \Delta S_t \|_{op} \| q_t \|_2 + \| S_t \|_{op} \| \Delta q_t \|_2 \nonumber \\
    \| \Delta F_t \|_{RMS}
        &\leq \| \Delta S_t \|_{op} \| q_t \|_{RMS} + \| S_t \|_{op} \| \Delta q_t \|_{RMS} \nonumber \\
        &\leq \| \Delta S_t \|_{op} + \| S_t \|_{op} \| \Delta q_t \|_{RMS} \label{eq:F_t-sensitivity}
\end{align}$$
Since $\| q_t \|_{RMS} \leq 1$ by assumption, we only need to bound $\| S_t \|_{op}$ and $\| \Delta S_t \|_{op}$. And to do so, we need to bound $\| A_t \|_{op}$, $\| B_t \|_{op}$, $\| \Delta A_t \|_{op}$, and $\| \Delta B_t \|_{op}$ first.

$$\begin{align}
    A_t
        &= \alpha_t \left( I - \frac{\beta_t}{d} k_t k_t^T \right) \nonumber \\
    \| A_t \|_{op}
        &\leq \alpha_t \left\| I - \frac{\beta_t}{d} k_t k_t^T \right\|_{op} \nonumber \\
        &\leq \alpha_t \left\| U \left( 1 - \beta_t, 1, \ldots, 1 \right) U^T \right\|_{op} \nonumber \\
        &\leq \alpha_t \max( | 1 - \beta_t |, 1 ) \nonumber \\
        &\leq \alpha_t &&\forall \beta_t \in [0, 2) \nonumber \\
        &\leq \alpha
\end{align}$$

Differentiating $A_t$ then yields,
$$\begin{align}
    \Delta A_t
        &:= \nabla A_t \diamond (\Delta q_t, \Delta k_t, \Delta v_t) \nonumber \\
        &= - \alpha_t \frac{\beta_t}{d} \left( \Delta k_t k_t^T + k_t \Delta k_t^T \right) \label{eq:Delta_A_t} \\
    \| \Delta A_t \|_{op}
        &\leq \alpha_t \beta_t \left( \left\| \frac{1}{d}\Delta k_t k_t^T \right\|_{op} + \left\| \frac{1}{d} k_t \Delta k_t^T \right\|_{op} \right) \nonumber \\
        &\leq \alpha_t \beta_t \left( \| \Delta k_t \|_{RMS} \| k_t \|_{RMS} + \| k_t \|_{RMS} \| \Delta k_t \|_{RMS} \right) \nonumber \\
        &\leq 2 \alpha_t \beta_t \| \Delta k_t \|_{RMS} \nonumber \\
        &\leq 2 \alpha \beta \Delta K
\end{align}$$

Likewise, for $B_t$, we have,
$$\begin{align}
    B_t
        &= \frac{\beta_t}{d} v_t k_t^T \nonumber \\
    \| B_t \|_{op}
        &\leq \beta_t \left\| \frac{1}{d} v_t k_t^T \right\|_{op} \nonumber \\
        &\leq \beta_t \| v_t \|_{RMS} \| k_t \|_{RMS} \nonumber \\
        &\leq \beta_t \nonumber \\
        &\leq \beta
\end{align}$$

And differentiating $B_t$, we get,
$$\begin{align}
    \Delta B_t
        &:= \nabla B_t \diamond (\Delta q_t, \Delta k_t, \Delta v_t) \nonumber \\
        &= \frac{\beta_t}{d} \left( \Delta v_t k_t^T + v_t \Delta k_t^T \right) \label{eq:Delta_B_t} \\
    \| \Delta B_t \|_{op}
        &\leq \beta_t \left( \left\| \frac{1}{d} \Delta v_t k_t^T \right\|_{op} + \left\| \frac{1}{d} v_t \Delta k_t^T \right\|_{op} \right) \nonumber \\
        &\leq \beta_t \left( \| \Delta v_t \|_{RMS} \| k_t \|_{RMS} + \| v_t \|_{RMS} \| \Delta k_t \|_{RMS} \right) \nonumber \\
        &\leq \beta_t \left( \| \Delta v_t \|_{RMS} + \| \Delta k_t \|_{RMS} \right) \nonumber \\
        &\leq \beta (\Delta V + \Delta K)
\end{align}$$

Now, the bounds on $S_t$ and $\Delta S_t$ can be derived as follows:
$$\begin{align}
    S_t
        &= S_{t-1} A_t + B_t \nonumber \\
        &= \sum_{i=1}^{t} B_i \prod_{j=i+1}^{t} A_j \nonumber \\
    \| S_t \|_{op}
        &\leq \sum_{i=1}^{t} \| B_i \|_{op} \prod_{j=i+1}^{t} \| A_j \|_{op} \nonumber \\
        &\leq \sum_{i=1}^{t} \beta \alpha^{t-i} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} \label{eq:S_t-bound}
\end{align}$$

Differentiating $S_t$ with respect to the perturbations $(\Delta q_t, \Delta k_t, \Delta v_t)$, then yields,
$$\begin{align}
    \Delta S_t
        &:= \nabla S_t \diamond (\Delta q_t, \Delta k_t, \Delta v_t) \nonumber \\
        &= \Delta S_{t-1} A_t + S_{t-1} \Delta A_t + \Delta B_t \label{eq:Delta_S_t} \\
        &= \sum_{i=1}^{t} \left( S_{i-1} \Delta A_i + \Delta B_i \right) \prod_{j=i+1}^{t} A_j \nonumber \\
    \| \Delta S_t \|_{op}
        &\leq \sum_{i=1}^{t} \left( \| S_{i-1} \|_{op} \| \Delta A_i \|_{op} + \| \Delta B_i \|_{op} \right) \prod_{j=i+1}^{t} \| A_j \|_{op} \nonumber \\
        &\leq \sum_{i=1}^{t} \left( \frac{\beta}{1 - \alpha} (2 \alpha \beta \Delta K) + \beta (\Delta V + \Delta K) \right) \alpha^{t-i} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} \Delta V + \left( \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right) \Delta K \label{eq:Delta_S_t-bound}
\end{align}$$

Combining Inequalities $\eqref{eq:F_t-sensitivity}$, $\eqref{eq:S_t-bound}$, and $\eqref{eq:Delta_S_t-bound}$, we have,
$$\begin{align}
    \| \Delta F_t \|_{RMS}
        &= \| \Delta S_t \|_{op} + \| S_t \|_{op} \| \Delta q_t \|_{RMS} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} \Delta V + \left( \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right) \Delta K + \frac{\beta}{1 - \alpha} \Delta Q \nonumber \\
        &\leq \left(\frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right)\left( \Delta Q + \Delta K + \Delta V \right) \label{eq:F_t-sensitivity-final} \\
    \| \Delta F \|_{\infty-RMS}
        &\leq \left(\frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right)\left( \Delta Q + \Delta K + \Delta V \right) \qquad \blacksquare \label{eq:F-sensitivity-final}
\end{align}$$

---

To prove the sharpness bound at time step $t$, we first take the derivative of Equation $\eqref{eq:F_t-derivative}$ towards $(\tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t)$,
$$\begin{align}
    \Delta^2 F_t
        &:= ( \tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t ) \diamond \nabla^2 \texttt{F}_t \diamond ( \Delta q_t, \Delta k_t, \Delta v_t ) \nonumber \\
        &= \Delta^2 S_t q_t
            + \Delta S_t \tilde{\Delta} q_t
            + \tilde{\Delta} S_t \Delta q_t
            + \cancel{S_t \Delta^2 q_t} \\
    \| \Delta^2 F_t \|_{2}
        &\leq \| \Delta^2 S_t \|_{op} \| q_t \|_2
            + \| \Delta S_t \|_{op} \| \tilde{\Delta} q_t \|_2
            + \| \tilde{\Delta} S_t \|_{op} \| \Delta q_t \|_2 \nonumber \\
    \| \Delta^2 F_t \|_{RMS}
        &\leq \| \Delta^2 S_t \|_{op} \| q_t \|_{RMS}
            + \| \Delta S_t \|_{op} \| \tilde{\Delta} q_t \|_{RMS}
            + \| \tilde{\Delta} S_t \|_{op} \| \Delta q_t \|_{RMS} \nonumber \\
        &\leq \| \Delta^2 S_t \|_{op}
            + \| \Delta S_t \|_{op} \| \tilde{\Delta} q_t \|_{RMS}
            + \| \tilde{\Delta} S_t \|_{op} \| \Delta q_t \|_{RMS} \label{eq:Delta2_F_t-bound} \\
\end{align}$$

We have already derived bounds for $\| S_t \|_{op}$ and $\| \Delta S_t \|_{op}$ (and $\| \tilde{\Delta} S_t \|_{op}$ by extension) in the sensitivity proof; we only need to bound $\| \Delta^2 S_t \|_{op}$. To do so, we first need to bound $\| \Delta^2 A_t \|_{op}$ and $\| \Delta^2 B_t \|_{op}$.

Differentiating Equations $\eqref{eq:Delta_A_t}$ and $\eqref{eq:Delta_B_t}$ towards $(\tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t)$ yields,
$$\begin{align}
    \Delta^2 A_t
        &:= ( \tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t ) \diamond \nabla^2 A_t \diamond ( \Delta q_t, \Delta k_t, \Delta v_t ) \nonumber \\
        &= - \alpha_t \frac{\beta_t}{d} \left(
            \Delta k_t \tilde{\Delta} k_t^T
            + \tilde{\Delta} k_t \Delta k_t^T \right) \label{eq:Delta2_A_t} \\
    \| \Delta^2 A_t \|_{op}
        &\leq \alpha_t \beta_t \left(
            \left\| \frac{1}{d} \Delta k_t \tilde{\Delta} k_t^T \right\|_{op}
            + \left\| \frac{1}{d} \tilde{\Delta} k_t \Delta k_t^T \right\|_{op} \right) \nonumber \\
        &\leq \alpha_t \beta_t \left(
            \| \Delta k_t \|_{RMS} \| \tilde{\Delta} k_t \|_{RMS}
            + \| \tilde{\Delta} k_t \|_{RMS} \| \Delta k_t \|_{RMS} \right) \nonumber \\
        &\leq 2 \alpha_t \beta_t \| \Delta k_t \|_{RMS} \| \tilde{\Delta} k_t \|_{RMS} \nonumber \\
        &\leq 2 \alpha \beta \Delta K \tilde{\Delta} K \\
    \Delta^2 B_t
        &:= ( \tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t ) \diamond \nabla^2 B_t \diamond ( \Delta q_t, \Delta k_t, \Delta v_t ) \nonumber \\
        &= \frac{\beta_t}{d} \left(
            \Delta v_t \tilde{\Delta} k_t^T
            + \tilde{\Delta} v_t \Delta k_t^T \right) \label{eq:Delta2_B_t} \\
    \| \Delta^2 B_t \|_{op}
        &\leq \beta_t \left(
            \left\| \frac{1}{d} \Delta v_t \tilde{\Delta} k_t^T \right\|_{op}
            + \left\| \frac{1}{d} \tilde{\Delta} v_t \Delta k_t^T \right\|_{op} \right) \nonumber \\
        &\leq \beta_t \left(
            \| \Delta v_t \|_{RMS} \| \tilde{\Delta} k_t \|_{RMS}
            + \| \tilde{\Delta} v_t \|_{RMS} \| \Delta k_t \|_{RMS} \right) \nonumber \\
        &\leq \beta_t \left(
            \| \Delta v_t \|_{RMS} \| \tilde{\Delta} k_t \|_{RMS}
            + \| \tilde{\Delta} v_t \|_{RMS} \| \Delta k_t \|_{RMS} \right) \nonumber \\
        &\leq \beta ( \Delta V \tilde{\Delta} K + \tilde{\Delta} V \Delta K ) \\
\end{align}$$

Now, let us take the derivative of Equation $\eqref{eq:Delta_S_t}$ towards $(\tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t)$,
$$\begin{align}
    \Delta^2 S_t
        &:= ( \tilde{\Delta} q_t, \tilde{\Delta} k_t, \tilde{\Delta} v_t ) \diamond \nabla^2 S_t \diamond ( \Delta q_t, \Delta k_t, \Delta v_t ) \nonumber \\
        &= \Delta^2 S_{t-1} A_t
            + \Delta S_{t-1} \tilde{\Delta} A_t
            + \tilde{\Delta} S_{t-1} \Delta A_t
            + S_{t-1} \Delta^2 A_t
            + \Delta^2 B_t \label{eq:Delta2_S_t} \\
        &= \sum_{i=1}^{t} \Big( S_{i-1} \Delta^2 A_i
            + \Delta S_{i-1} \tilde{\Delta} A_i
            + \tilde{\Delta} S_{i-1} \Delta A_i
            + \Delta^2 B_i \Big) \prod_{j=i+1}^{t} A_j \nonumber \\
    \| \Delta^2 S_t \|_{op}
        &\leq \sum_{i=1}^{t} \Big( \| S_{i-1} \|_{op} \| \Delta^2 A_i \|_{op}
            + \| \Delta S_{i-1} \|_{op} \| \tilde{\Delta} A_i \|_{op} \nonumber \\
            &\qquad\quad+ \| \tilde{\Delta} S_{i-1} \|_{op} \| \Delta A_i \|_{op}
            + \| \Delta^2 B_i \|_{op} \Big) \prod_{j=i+1}^{t} \| A_j \|_{op} \nonumber \\
        &\leq \sum_{i=1}^{t} \Bigg(
            \frac{\beta}{1 - \alpha} (2 \alpha \beta \Delta K \tilde{\Delta} K) \nonumber \\
            &\qquad\qquad+ \frac{\beta}{1 - \alpha} \Delta V (2 \alpha \beta \tilde{\Delta} K)
                + \left( \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right) \Delta K (2 \alpha \beta \tilde{\Delta} K) \nonumber \\
            &\qquad\qquad+ \frac{\beta}{1 - \alpha} \tilde{\Delta} V (2 \alpha \beta \Delta K)
                + \left( \frac{\beta}{1 - \alpha}+ \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right) \tilde{\Delta} K (2 \alpha \beta \Delta K) \nonumber \\
            &\qquad\qquad+ \beta ( \Delta V \tilde{\Delta} K + \tilde{\Delta} V \Delta K ) \Bigg) \alpha^{t-i} \nonumber \\
        &\leq \left( \frac{6 \alpha \beta^2}{(1 - \alpha)^2} + \frac{8 \alpha^2 \beta^3}{(1 - \alpha)^3} \right) \Delta K \tilde{\Delta} K \nonumber \\
            &\qquad+ \left( \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right) \left( \Delta V \tilde{\Delta} K + \tilde{\Delta} V \Delta K \right) \label{eq:Delta2_S_t-bound}
\end{align}$$

Combining Inequalities $\eqref{eq:Delta2_F_t-bound}$, $\eqref{eq:S_t-bound}$, $\eqref{eq:Delta_S_t-bound}$, and $\eqref{eq:Delta2_S_t-bound}$, we have,
$$\begin{align}
    \| \Delta^2 F_t \|_{RMS}
        &= \| \Delta^2 S_t \|_{op}
            + \| \Delta S_t \|_{op} \| \tilde{\Delta} q_t \|_{RMS}
            + \| \tilde{\Delta} S_t \|_{op} \| \Delta q_t \|_{RMS} \nonumber \\
        &\leq \left( \frac{6 \alpha \beta^2}{(1 - \alpha)^2} + \frac{8 \alpha^2 \beta^3}{(1 - \alpha)^3} \right) \Delta K \tilde{\Delta} K \nonumber \\
            &\qquad+ \left( \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \right) \left( \Delta V \tilde{\Delta} K + \tilde{\Delta} V \Delta K + \Delta K \tilde{\Delta} Q + \tilde{\Delta} K \Delta Q \right) \nonumber \\
            &\qquad+ \frac{\beta}{1 - \alpha} \left( \Delta V \tilde{\Delta} Q + \tilde{\Delta} V \Delta Q \right) \nonumber \\
        &\leq \max\Bigg\{
                \frac{6 \alpha \beta^2}{(1 - \alpha)^2} + \frac{8 \alpha^2 \beta^3}{(1 - \alpha)^3},
                \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2}
            \Bigg\} \nonumber \\
            &\qquad \times \left( \Delta Q + \Delta K + \Delta V \right) \left( \tilde{\Delta} Q + \tilde{\Delta} K + \tilde{\Delta} V \right) \label{eq:Delta2_F_t-final} \\
    \| \Delta^2 F \|_{\infty-RMS}
        &\leq \max\Bigg\{
                \frac{6 \alpha \beta^2}{(1 - \alpha)^2} + \frac{8 \alpha^2 \beta^3}{(1 - \alpha)^3},
                \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2}
            \Bigg\} \nonumber \\
            &\qquad \times \left( \Delta Q + \Delta K + \Delta V \right) \left( \tilde{\Delta} Q + \tilde{\Delta} K + \tilde{\Delta} V \right) \qquad \blacksquare \label{eq:Delta2_F-final}
\end{align}$$

### 1-Lipschitz Gated DeltaNet

> **Corollary 2 (1-Lipschitz Gated DeltaNet).** Under the same assumptions as Theorem 1, setting,
$$\begin{equation}
    \beta_t \leq \frac{1 - \alpha_t}{2} \label{eq:1-lipschitz-condition}
\end{equation}$$
for all $t$, guarantees that Gated DeltaNet is unit sensitive and $\frac{5}{2}$-sharp.

**Proof.** Substituting Inequality $\eqref{eq:1-lipschitz-condition}$ into Equations $\eqref{eq:gdn-sensitivity}$ and $\eqref{eq:gdn-sharpness}$ yields,
$$\begin{align}
    \sigma
        &\leq \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2} \nonumber \\
        &\leq \frac{1}{2} + \frac{\alpha}{2} \nonumber \\
        &\leq 1 \nonumber \\
    \gamma
        &\leq \max\Bigg\{
                \frac{6 \alpha \beta^2}{(1 - \alpha)^2} + \frac{8 \alpha^2 \beta^3}{(1 - \alpha)^3},
                \frac{\beta}{1 - \alpha} + \frac{2 \alpha \beta^2}{(1 - \alpha)^2}
            \Bigg\} \nonumber \\
        &\leq \max\Bigg\{
                \frac{3 \alpha}{2} + \alpha^2,
                \frac{1}{2} + \frac{\alpha}{2}
            \Bigg\} \nonumber \\
        &\leq \frac{5}{2} \qquad \blacksquare \nonumber \\
\end{align}$$

## Sensitivity and Sharpness of Mamba 2

> **Theorem 3 (Sensitivity and Sharpness of Mamba 2).** Let $T$ be the sequence length, $d$ be the model width, and $q, k, v \in \mathbb{R}^{T \times d}$ be the query, key, and value sequences, respectively, $\text{RMS}$-normalized such that $\| q_t \|_{RMS}, \| k_t \|_{RMS}, \| v_t \|_{RMS} \leq 1$ for all $t$. Also let the initial state $S_0 = 0$, and $\alpha_t, \beta_t \in \mathbb{R}$ be learnable parameters such that $0 \leq \alpha_t \leq \alpha < 1$, and $0 \leq \beta_t \leq \beta < 1$ for some constants $\alpha, \beta > 0$. Then Mamba 2 ([Dao et. al., 2024](https://proceedings.mlr.press/v235/dao24a.html)) with the following update rule:
$$\begin{align}
    A_t
        &= \text{diag}(\alpha_t I) \\
    B_t
        &= \frac{\beta_t}{d} v_t k_t^T \\
    S_t
        &= S_{t-1} A_t + B_t \\
    \texttt{F}_t
        &= S_t q_t
\end{align}$$
has the following sensitivity $\sigma$ and sharpness $\gamma$ bounds:
$$\begin{align}
    \sigma
        &\leq \frac{\beta}{1 - \alpha} \label{eq:mamba2-sensitivity} \\
    \gamma
        &\leq \frac{\beta}{1 - \alpha} \label{eq:mamba2-sharpness}
\end{align}$$

**Proof.** We follow the same proof structure as in Theorem 1. The main difference lies in the structure of $A_t$: for Mamba 2, $A_t$ does not depend on $k_t$, making $\Delta A_t$ and $\Delta^2 A_t$ equal to zero. Repeating the steps in the sensitivity proof of Theorem 1, we have,

$$\begin{align}
    \Delta S_t
        &= \sum_{i=1}^{t} \left( \cancel{S_{t-1} \Delta A_t} + \Delta B_i \right) \prod_{j=i+1}^{t} A_j \nonumber \\
    \| \Delta S_t \|_{op}
        &\leq \sum_{i=1}^{t} \| \Delta B_i \|_{op} \prod_{j=i+1}^{t} \| A_j \|_{op} \nonumber \\
        &\leq \sum_{i=1}^{t} \beta \left( \| \Delta v_i \|_{RMS} + \| \Delta k_i \|_{RMS} \right) \alpha^{t-i} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} ( \Delta V + \Delta K ) \label{eq:mamba2-Delta_S_t-bound}
\end{align}$$

Combining Inequalities $\eqref{eq:F_t-sensitivity}$, $\eqref{eq:S_t-bound}$, and $\eqref{eq:mamba2-Delta_S_t-bound}$, we have,
$$\begin{align}
    \| \Delta F_t \|_{RMS}
        &= \| \Delta S_t \|_{op} + \| S_t \|_{op} \| \Delta q_t \|_{RMS} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} ( \Delta V + \Delta K ) + \frac{\beta}{1 - \alpha} \Delta Q \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} ( \Delta Q + \Delta K + \Delta V ) \label{eq:mamba2-F_t-sensitivity-final} \\
    \| \Delta F \|_{\infty-RMS}
        &\leq \frac{\beta}{1 - \alpha} ( \Delta Q + \Delta K + \Delta V ) \qquad \blacksquare \label{eq:mamba2-F-sensitivity-final}
\end{align}$$

And for the sharpness proof, we have,
$$\begin{align}
    \| \Delta^2 S_t \|_{op}
        &\leq \sum_{i=1}^{t} \Big(
            \cancel{\| S_{i-1} \|_{op} \| \Delta^2 A_i \|_{op}}
            + \cancel{\| \Delta S_{i-1} \|_{op} \| \tilde{\Delta} A_i \|_{op}} \nonumber \\
            &\qquad\qquad+ \cancel{\| \tilde{\Delta} S_{i-1} \|_{op} \| \Delta A_i \|_{op}}
            + \| \Delta^2 B_i \|_{op} \Big) \prod_{j=i+1}^{t} \| A_j \|_{op} \nonumber \\
        &\leq \sum_{i=1}^{t} \beta ( \Delta V \tilde{\Delta} K + \tilde{\Delta} V \Delta K ) \alpha^{t-i} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} ( \Delta V \tilde{\Delta} K + \tilde{\Delta} V \Delta K ) \label{eq:mamba2-Delta2_S_t-bound}
\end{align}$$

Combining Inequalities $\eqref{eq:Delta2_F_t-bound}$, $\eqref{eq:S_t-bound}$, $\eqref{eq:mamba2-Delta_S_t-bound}$, and $\eqref{eq:mamba2-Delta2_S_t-bound}$ yields,
$$\begin{align}
    \| \Delta^2 F_t \|_{RMS}
        &= \| \Delta^2 S_t \|_{op}
            + \| \Delta S_t \|_{op} \| \tilde{\Delta} q_t \|_{RMS}
            + \| \tilde{\Delta} S_t \|_{op} \| \Delta q_t \|_{RMS} \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} (
            \Delta V \tilde{\Delta} K
            + \tilde{\Delta} V \Delta K
            + \Delta V \tilde{\Delta} Q
            + \tilde{\Delta} V \Delta Q
            + \Delta K \tilde{\Delta} Q
            + \tilde{\Delta} K \Delta Q
            ) \nonumber \\
        &\leq \frac{\beta}{1 - \alpha} ( \Delta Q + \Delta K + \Delta V ) ( \tilde{\Delta} Q + \tilde{\Delta} K + \tilde{\Delta} V ) \label{eq:mamba2-Delta2_F_t-final} \\
    \| \Delta^2 F \|_{\infty-RMS}
        &\leq \frac{\beta}{1 - \alpha} ( \Delta Q + \Delta K + \Delta V ) ( \tilde{\Delta} Q + \tilde{\Delta} K + \tilde{\Delta} V ) \qquad \blacksquare \label{eq:mamba2-Delta2_F-final}
\end{align}$$

### 1-Lipschitz Mamba 2

> **Corollary 4 (1-Lipschitz Mamba 2).** Under the same assumptions as Theorem 3, setting,
$$\begin{equation}
    \beta_t \leq 1 - \alpha_t \label{eq:mamba2-1-lipschitz-condition}
\end{equation}$$
for all $t$, guarantees that Mamba 2 is unit sensitive and unit sharp.

**Proof.** Substituting Inequality $\eqref{eq:mamba2-1-lipschitz-condition}$ into Equations $\eqref{eq:mamba2-sensitivity}$ and $\eqref{eq:mamba2-sharpness}$ yields the desired result. $\blacksquare$

## How to Cite

```bibtex
@misc{cesista2026sensitivitysharpnessgatedlinearattn,
  author = {Franz Louis Cesista},
  title = {{S}ensitivity and {S}harpness of {G}ated {L}inear {A}ttention {M}echanisms},
  year = {2026},
  month = {January},
  day = {2},
  url = {https://leloykun.github.io/ponder/lipschitz-gated-linear-attn/},
}
```

## References

1. Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil (2025). Fast and Simplex: 2-Simplicial Attention in Triton. URL https://arxiv.org/abs/2507.02754v1
2. James Clift, Dmitry Doryn, Daniel Murfet, James Wallbridge (2019). Logic and the 2-Simplicial Transformer. URL https://arxiv.org/abs/1909.00668
3. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017). Attention is all you need. URL https://arxiv.org/abs/1706.03762
4. Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim (2025). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. URL https://arxiv.org/abs/2406.06484
5. Songlin Yang, Jan Kautz, Ali Hatamizadeh (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. URL https://arxiv.org/abs/2412.06464
6. Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Proceedings of the 41st International Conference on MachineLearning, volume 235 of Proceedingsof Machine Learning Research, pp. 10041–10071. PMLR, 2024b. URL https://proceedings.mlr.press/v235/dao24a.html.

## Appendix

For vectors $x, y \in \mathbb{R}^d$, we have,
$$\begin{align}
    \| x y^T \|_{op}
        &\leq d \| x \|_{RMS} \| y \|_{RMS} \nonumber \\
\end{align}$$
Thus, if $\| x \|_{RMS}, \| y \|_{RMS} \leq 1$, then the largest eigenvalue of $x y^T$ is at most $d$, and the remaining eigenvalues are all zero.
