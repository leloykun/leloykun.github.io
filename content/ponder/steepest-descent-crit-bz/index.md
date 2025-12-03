---
title: "Critical Batch Size for Steepest Descent Under Arbitrary Norms"
date: 2025-11-22
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "First-order optimization under arbitrary norms with Nesterov momentum (and weight decay) yields the same critical batch size."
summary: "First-order optimization under arbitrary norms with Nesterov momentum (and weight decay) yields the same critical batch size."
# cover:
#     image: lr_transfer_pdhg_stiefel_spectral.png
#     alt: "Cover"
#     relative: true
# editPost:
#     URL: "https://x.com/leloykun/status/1958915061793075549"
#     Text: "Crossposted from X (formerly Twitter)"
citation:
    title: "Critical Batch Size for Steepest Descent Under Arbitrary Norms"
    author:
        - "Franz Louis Cesista"
    publication_date: "2025/11/22"
---

## 1. Introduction and preliminaries

This work generalizes prior results by [Sato et al. (2025)](https://arxiv.org/abs/2507.01598) on the critical batch size for the Muon optimizer [(Jordan et al., 2025)](https://kellerjordan.github.io/posts/muon/) to steepest descent under arbitrary norms with Nesterov momentum and weight decay. We show that the same critical batch size formula holds universally across all norms.

We consider the following optimization problem:
$$\arg\min_{W \in \mathcal{W}} f(W)$$
where $f(\cdot): \mathcal{W} \to \mathbb{R}$ is a bounded below and differentiable objective function, and $\mathcal{W}$ is a finite-dimensional vector space over $\mathbb{R}$, e.g., $\mathcal{W} = \mathbb{R}^{m \times n}$, equipped with an arbitrary norm $\| \cdot \|$ and its dual norm $\| \cdot \|^{\dagger}$.

More generally, we often take $\mathcal{W}$ to be a product of layers' weight spaces, e.g.,
$$\mathcal{W} = \prod_{l=1}^{L} \mathbb{R}^{m_l \times n_l},$$
for an $L$-layer neural network with weight matrices $(W^{(l)})_{l=1}^L$ where $W^{(l)} \in \mathbb{R}^{m_l \times n_l}$ for each layer $l$. Given layer-wise norms $\| \cdot \|_{(l)}$ and their duals $\| \cdot \|_{(l)}^{\dagger}$, we can then define the product norm and its dual as,
$$\begin{align}
    \| W \| &:= h\left( \| W^{(1)} \|_{(1)}, \| W^{(2)} \|_{(2)}, \ldots, \| W^{(L)} \|_{(L)} \right) \nonumber \\
    \| G \|^{\dagger} &:= h^{\dagger}\left( \| G^{(1)} \|_{(1)}^{\dagger}, \| G^{(2)} \|_{(2)}^{\dagger}, \ldots, \| G^{(L)} \|_{(L)}^{\dagger} \right) \nonumber
\end{align}$$
for some vector norm $h$ and its dual $h^{\dagger}$ on $\mathbb{R}^L$. Our results still hold under this more general setting.

Now, at iteration $t$, we sample an i.i.d. minibatch $S_t = \{ i_1, i_2, \ldots, i_b \}$ of size $b$ from the training dataset. For each data point $i$, we write the per-example stochastic gradient as,
$$ G_{\xi_{t, i}}(W_t) := \nabla f(W_t) - \xi_{t, i},$$
where $\xi_{t,i}$ is the (additive) gradient noise at $(t, i)$. We then write the minibatch stochastic gradient and noise as,
$$\begin{align}
    \nabla f_{S_t}(W_t)
        &:= \frac{1}{b}\sum_{i=1}^{b} G_{\xi_{t,i}}(W_t) \label{eq:def_minibatch_gradient} \\
    \xi_{S_t}
        &:= \nabla f(W_t) - \nabla f_{S_t}(W_t)
\end{align}$$

### 1.1. Nesterov momentum

For a given momentum hyperparameter $\beta \in (0, 1)$, Nesterov momentum is defined in terms of the minibatch stochastic gradients as,
$$\begin{align}
    M_t &= \beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t) \nonumber \\
    C_t &= \beta M_t + (1 - \beta) \nabla f_{S_t}(W_t) \nonumber \\
\end{align}$$
where $M_t$ is the usual momentum accumulator and $C_t$ is the Nesterov "look-ahead" gradient. We then use $C_t$ to compute the steepest descent update direction under the norm $\| \cdot \|$.

### 1.2. Linear Minimization Oracles (LMOs) and dual norms

Given a norm $\| \cdot \|$ on $\mathbb{R}^{m \times n}$ and its dual $\| \cdot \|^{\dagger}$, the linear minimization oracle (LMO) is defined as,
$$\begin{align}
    A_t^*
        &:= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle C_t, A \rangle_F \quad \text{ s.t. } \quad \| A \| \leq 1 \nonumber \\
        &= \texttt{LMO}_{\| \cdot \|}(C_t) \nonumber
\end{align}$$
such that,
$$\begin{align}
    \| A_t^* \|
        &= 1 \label{eq:lmo-norm} \\
    \langle C_t, A_t^* \rangle_F
        &= \langle C_t, \texttt{LMO}_{\| \cdot \|}(C_t) \rangle_F \nonumber \\
        &= \arg\min_{A \leq 1} \langle C_t, A \rangle_F \nonumber \\
        &= -\arg\max_{A \leq 1} \langle C_t, A \rangle_F \nonumber \\
        &= - \| C_t \|^{\dagger} \label{eq:lmo-inner-product}
\end{align}$$

The update rule for steepest descent with step size $\eta > 0$ at time $t$ and weight decay term $\lambda \geq 0$ is then given by,
$$\begin{equation}
    W_{t+1} = (1 - \lambda\eta) W_t + \eta A_t^* \label{eq:updateweightdecay}
\end{equation}$$

### 1.3. Assumptions

> **Assumption 1 (Unbiased gradient noise, per sample).** At each time step $t$ and for each data point $i \in S_t$, the gradient noise satisfies,
$$\begin{equation} \mathbb{E}\left[ \xi_{t, i} | W_t \right] = 0, \end{equation}$$
and the samples $(\xi_{t,i})_{i=1}^b$ are conditionally independent given $W_t$. To simplify notation, we will often omit the conditioning on $W_t$ when it is clear from context.

> **Assumption 2 (Bounded gradient noise variance).** There exists $\sigma > 0$ such that for all $t, i$,
$$\begin{equation}
    \mathbb{E}\left[\| \xi_{t,i} \|^{\dagger 2} \right] \leq \sigma^2
\end{equation}$$
By norm equivalence in finite dimensions, there exists $\kappa_{\sigma} > 0$ such that,
$$\begin{equation} \mathbb{E}\left[ \| \xi_{t,i} \|_F^2 \right] \leq \kappa_{\sigma}^2 \sigma^2 =: \sigma_F^2 \end{equation}$$
where $\sigma_F := \kappa_{\sigma} \sigma$ and treat $\sigma_F$ as the gradient noise variance scale in the Frobenius norm.

> **Assumption 3 (L-smoothness of $f$ under $(\| \cdot \|, \| \cdot \|^{\dagger})$).** There exists $L > 0$ such that for all $X, Y \in \mathcal{W}$,
$$\begin{equation}
    \| \nabla f(Y) - \nabla f(X) \|^{\dagger} \leq L \| Y - X \|
\end{equation}$$
By norm equivalence, there exists $\kappa_L > 0$ such that,
$$\begin{equation}
    \| \nabla f(Y) - \nabla f(X) \|_F \leq \kappa_L L \| Y - X \|_F = L_F \| Y - X \|_F
\end{equation}$$
where $L_F := \kappa_L L$.

> **Assumption 4 (Local D-smoothness of $\| \cdot \|^{\dagger}$ in the noise region).** There exists a large enough $R > 0$ such that $\mathbb{P}(\| \xi_{t,i} \|^{\dagger}, \| \nabla f(W_t) - M_t \|^{\dagger}, \| \nabla f(W_t) - C_t \|^{\dagger} \leq R) = 1$ for all $t, i$. For minibatch size $b$, let,
$$\begin{align}
    K &:= \{ X^{\dagger} \in W^{\dagger} : \| X^{\dagger} \|^{\dagger} \leq bR \} \nonumber \\
    g(X^{\dagger}) &:= \frac{1}{2} \| X^{\dagger} \|^{\dagger 2} \quad \forall X^{\dagger} \in K \nonumber
\end{align}$$
Intuitively, $K$ is the region where the (accumulated) gradient noise and momentum errors lie almost surely. Then there exists $D > 0$ such that for all $X^{\dagger}, Y^{\dagger} \in K$,
$$\begin{equation}
    \| \nabla g(Y^{\dagger}) - \nabla g(X^{\dagger}) \| \leq D \| Y^{\dagger} - X^{\dagger} \|^{\dagger}
\end{equation}$$
> Note that if $\| \cdot \|^{\dagger}$ is induced by an inner product, then $D = 1$.

## 2. Convergence bound for steepest descent under arbitrary norms without weight decay

### 2.1. Gradient noise and momentum error bounds

We first control the variance of the mini-batch noise.

> **Lemma 5 (Minibatch gradient noise bounds).** Under Assumptions 1-2 and 4, for any minibatch size $b \geq 1$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \mathbb{E}\left[ \xi_{S_t} \right]
        &= 0 \label{eq:minibatchmean} \\
    \mathbb{E}\left[ \| \xi_{S_t} \|^{\dagger 2} \right]
        &\leq \frac{D\sigma^2}{b}
\end{align}$$

**Proof.** For the minibatch gradient noise mean, we have,
$$\begin{align}
    \mathbb{E}\left[ \xi_{S_t} \right]
        &= \mathbb{E}\left[ \nabla f(W_t) - \nabla f_{S_t}(W_t) \right] \nonumber \\
        &= \mathbb{E}\left[ \nabla f(W_t) - \frac{1}{b} \sum_{i=1}^{b} G_{\xi_{t,i}}(W_t) \right] \nonumber \\
        &= \mathbb{E}\left[ \frac{1}{b} \sum_{i=1}^{b} (\nabla f(W_t) - G_{\xi_{t,i}}(W_t)) \right] \nonumber \\
        &= \mathbb{E}\left[ \frac{1}{b} \sum_{i=1}^{b} \xi_{t,i} \right] \nonumber \\
        &= \frac{1}{b} \sum_{i=1}^{b} \mathbb{E}\left[ \xi_{t,i} \right] \nonumber \\
        &= 0 \nonumber
\end{align}$$

And for the minibatch gradient noise variance, we have,
$$\begin{align}
    \mathbb{E}\left[ \| \xi_{S_t} \|^{\dagger 2} \right]
        &= \mathbb{E}\left[ \left\| \nabla f(W_t) - \nabla f_{S_t}(W_t) \right\|^{\dagger 2} \right] \nonumber \\
        &= \mathbb{E}\left[ \left\| \nabla f(W_t) - \frac{1}{b} \sum_{i=1}^{b} G_{\xi_{t,i}}(W_t) \right\|^{\dagger 2} \right] \nonumber \\
        &= \mathbb{E}\left[ \left\| \frac{1}{b} \sum_{i=1}^{b} (\nabla f(W_t) - G_{\xi_{t,i}}(W_t)) \right\|^{\dagger 2} \right] \nonumber \\
        &= \frac{1}{b^2} \mathbb{E}\left[ \left\| \sum_{i=1}^{b} \xi_{t,i} \right\|^{\dagger 2} \right] \nonumber
\end{align}$$
From Assumption (4), we can apply the descent lemma on the running sum up to $b-1$ and $\xi_{t, b}$, then unroll the recurrence to get,
$$\begin{align}
    \mathbb{E}\left[ \| \xi_{S_t} \|^{\dagger 2} \right]
        &\leq \frac{1}{b^2} \left(
            \mathbb{E}\left[ \left\| \sum_{i=1}^{b-1} \xi_{t,i} \right\|^{\dagger 2} \right]
            + \cancel{\left\langle \nabla \left\|\sum_{i=1}^{b-1} \xi_{t,i} \right\|^{\dagger 2}, \mathbb{E}\left[ \xi_{t,b} \right] \right\rangle}
            + D \mathbb{E}\left[ \| \xi_{t,b} \|^{\dagger 2} \right]
        \right) \nonumber \\
        &\leq \frac{D}{b^2} \sum_{i=1}^{b} \mathbb{E}\left[ \left\| \xi_{t,i} \right\|^{\dagger 2} \right] \nonumber \\
        &\leq \frac{D\sigma^2}{b} \quad\blacksquare \nonumber
\end{align}$$

---

We then bound the average first and second moments of the momentum error term,
$$E_t := \nabla f(W_t) - M_t,$$
and later the Nesterov momentum error term $\nabla f(W_t) - C_t$.

> **Proposition 6 (Average first and second moments of the momentum error term).** Let $\beta \in (0, 1)$, and $M_0 = 0$. Under Assumptions 1-4, for any $T \geq 1$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|^{\dagger} \right]
        &\leq \left( \sqrt{\frac{1 + \beta}{2}} \right)^t \| \nabla f(W_0) \|^{\dagger}
            + \frac{2}{1 - \beta} L \eta
            + \sqrt{2 (1 - \beta)} D \frac{\sigma}{\sqrt{b}} \\
    \mathbb{E}\left[\| E_t \|^{\dagger 2} \right]
        &\leq \left( \frac{1 + \beta}{2} \right)^t \| \nabla f(W_0) \|^{\dagger 2}
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2
            + 2 (1 - \beta) D^2 \frac{\sigma^2}{b}
\end{align}$$
Moreover, averaging over $T$ iterations yields,
$$\begin{align}
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[ \| E_t \|^{\dagger} \right]
        &\leq \frac{2\sqrt{2}}{1 - \beta}\frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{2}{1 - \beta} L \eta
            + \sqrt{2 (1 - \beta)} D \frac{\sigma}{\sqrt{b}} \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|^{\dagger 2} \right]
        &\leq \frac{2}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) \|^{\dagger 2}
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2
            + 2 (1 - \beta) D^2 \frac{\sigma^2}{b}
\end{align}$$

**Proof.** Notice that,
$$\begin{align}
    E_t
        &= \nabla f(W_t) - M_t \nonumber \\
        &= \nabla f(W_t) - (\beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta (\nabla f(W_t) - M_{t-1}) + (1 - \beta)\xi_{S_t} \nonumber \\
        &= \beta (\nabla f(W_t) - \nabla f(W_{t-1}) + \nabla f(W_{t-1}) - M_{t-1}) + (1 - \beta)\xi_{S_t} \nonumber \\
        &= \beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1})) + (1 - \beta) \xi_{S_t} \nonumber
\end{align}$$

Applying the descent lemma on $g(\cdot) = \frac{1}{2}\| \cdot \|^{\dagger 2}$, taking expectations, and using Lemma (5) then gives,
$$\begin{align}
    g(E_t)
        &\leq g(\beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1}))) \nonumber \\
            &\quad+ \langle \nabla g(\beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1}))), (1 - \beta) \xi_{S_t} \rangle \nonumber \\
            &\quad+ \frac{D}{2} \| (1 - \beta) \xi_{S_t} \|^{\dagger 2} \nonumber \\
    \frac{1}{2} \| E_t \|^{\dagger 2}
        &\leq \frac{\beta^2}{2} \| E_{t-1} + (\nabla f(W_t) - \nabla f(W_{t-1})) \|^{\dagger 2} \nonumber \\
            &\quad+ (1 - \beta) \langle \nabla g(\beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1}))), \xi_{S_t} \rangle \nonumber \\
            &\quad+ \frac{D}{2} (1 - \beta)^2 \| \xi_{S_t} \|^{\dagger 2} \nonumber \\
    \mathbb{E}\left[ \| E_t \|^{\dagger 2} \right]
        &\leq \beta^2 \mathbb{E}\left[ \| E_{t-1} + \nabla f(W_t) - \nabla f(W_{t-1}) \|^{\dagger 2} \right] \nonumber \\
            &\quad+ \cancel{\beta(1 - \beta) \langle \nabla \| E_{t-1} + \nabla f(W_t) - \nabla f(W_{t-1}) \|^{\dagger 2},  \mathbb{E}\left[ \xi_{S_t} \right] \rangle} \nonumber \\
            &\quad+ (1 - \beta)^2 D \mathbb{E}\left[ \| \xi_{S_t} \|^{\dagger 2} \right] \nonumber \\
        &\leq \beta^2 \mathbb{E}\left[ \| E_{t-1} + \nabla f(W_t) - \nabla f(W_{t-1}) \|^{\dagger 2} \right]
            + (1 - \beta)^2 D^2 \frac{\sigma^2}{b} \nonumber
\end{align}$$

Using the Peter-Paul inequality, together with Assumption (3) then yields,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|^{\dagger 2} \right]
        &\leq \beta^2 (1 + \epsilon) \mathbb{E}\left[ \| E_{t-1} \|^{\dagger 2} \right]
            + \beta^2 \left(1 + \frac{1}{\epsilon}\right) \mathbb{E}\left[ \| \nabla f(W_t) - \nabla f(W_{t-1}) \|^{\dagger 2} \right] \nonumber \\
            &\quad+ (1 - \beta)^2 D^2 \frac{\sigma^2}{b} \nonumber \\
        &\leq \beta^2 (1 + \epsilon) \mathbb{E}\left[ \| E_{t-1} \|^{\dagger 2} \right]
            + \beta^2 \left(1 + \frac{1}{\epsilon}\right) L^2 \eta^2
            + (1 - \beta)^2 D^2 \frac{\sigma^2}{b} \nonumber
\end{align}$$
for any $\epsilon > 0$. Setting $\epsilon = \frac{1 - \beta}{2}$ and unrolling the recurrence then gives,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|^{\dagger 2} \right]
        &\leq \frac{\beta^2(3 - \beta)}{2} \mathbb{E}\left[ \| E_{t-1} \|^{\dagger 2} \right]
            + \frac{\beta^2 (3 - \beta)}{1 - \beta} L^2 \eta^2
            + (1 - \beta)^2 D^2 \frac{\sigma^2}{b} \nonumber \\
        &\leq \frac{1 + \beta}{2} \mathbb{E}\left[ \| E_{t-1} \|^{\dagger 2} \right]
            + \frac{2}{1 - \beta} L^2 \eta^2
            + (1 - \beta)^2 D^2 \frac{\sigma^2}{b} \label{eq:approxub} \\
        &\leq \left( \frac{1 + \beta}{2} \right)^t \mathbb{E}\left[ \| E_{0} \|^{\dagger 2} \right]
            + \left(\frac{2}{1 - \beta} L^2 \eta^2
            + (1 - \beta)^2 D^2 \frac{\sigma^2}{b}\right) \sum_{k=0}^{t-1} \left(\frac{1 + \beta}{2}\right)^k \nonumber \\
        &\leq \left( \frac{1 + \beta}{2} \right)^t \| E_{0} \|^{\dagger 2}
            + \left(\frac{2}{1 - \beta} L^2 \eta^2
            + (1 - \beta)^2 D^2 \frac{\sigma^2}{b}\right) \frac{2}{1 - \beta} \nonumber \\
        &\leq \left( \frac{1 + \beta}{2} \right)^t \| E_{0} \|^{\dagger 2}
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2
            + 2(1 - \beta) D^2 \frac{\sigma^2}{b} \nonumber \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|^{\dagger 2}\right]
        &\leq \frac{2}{1 - \beta} \frac{1}{T} \| E_{0} \|^{\dagger 2}
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2
            + 2 (1 - \beta) D^2 \frac{\sigma^2}{b} \nonumber
\end{align}$$
where we use the crude upper bound $\frac{\beta^2 (3 - \beta)}{2} \leq \frac{1 + \beta}{2} \leq 1$ in Equation $\eqref{eq:approxub}$ to simplify the algebra.

For the first moment, Jensen's inequality and $\sqrt{a + b + c} \leq \sqrt{a} + \sqrt{b} + \sqrt{c}$ for $a, b, c > 0$ yields,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|^{\dagger} \right]
        &\leq \sqrt{\mathbb{E}\left[ \| E_t \|^{\dagger 2} \right]} \nonumber \\
        &\leq \left( \sqrt{\frac{1 + \beta}{2}} \right)^t \| E_{0} \|^{\dagger}
            + \frac{2}{1 - \beta} L \eta
            + \sqrt{2 (1 - \beta)} D \frac{\sigma}{\sqrt{b}} \nonumber \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|^{\dagger}\right]
        &\leq \frac{2\sqrt{2}}{1 - \beta} \frac{1}{T} \| E_{0} \|^{\dagger}
            + \frac{2}{1 - \beta} L \eta
            + \sqrt{2 (1 - \beta)} D \frac{\sigma}{\sqrt{b}} \nonumber \\
\end{align}$$
Substituting $E_0 = \nabla f(W_0) - M_0 = \nabla f(W_0)$ completes the proof. $\quad\blacksquare$

---

We now bound the Nesterov momentum error term.

> **Corollary 7 (Average first and second moments of the Nesterov momentum error term).** Under the same assumptions as Proposition (6), for any $T \geq 1$ and any norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| \nabla f(W_t) - C_t \|^{\dagger} \right]
        &\leq \frac{2\sqrt{2}\beta}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{2 \beta}{1 - \beta} L \eta \nonumber \\
        &\quad+ \left(\sqrt{2 (1 - \beta)} D \beta + (1 - \beta)\right) \frac{\sigma}{\sqrt{b}} \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| \nabla f(W_t) - C_t \|^{\dagger 2}\right]
        &\leq \frac{2\beta}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) \|^{\dagger 2}
            + \frac{4\beta}{(1 - \beta)^2} L^2 \eta^2 \nonumber \\
        &\quad+ (2 D^2 \beta + 1) (1 - \beta) \frac{\sigma^2}{b}
\end{align}$$

**Proof.** We have,
$$\begin{align}
    \nabla f(W_t) - C_t
        &= \nabla f(W_t) - (\beta M_t + (1 - \beta) \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta (\nabla f(W_t) - M_t) + (1 - \beta) (\nabla f(W_t) - \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta E_t + (1 - \beta) \xi_{S_t} \nonumber
\end{align}$$
Since $x \mapsto \| x \|^{\dagger}$ and $x \mapsto \| x \|^{\dagger 2}$ are convex
$$\begin{align}
    \| \nabla f(W_t) - C_t \|^{\dagger 2}
        &\leq \beta \| E_t \|^{\dagger 2} + (1 - \beta) \| \xi_{S_t} \|^{\dagger 2} \nonumber \\
    \| \nabla f(W_t) - C_t \|^{\dagger}
        &\leq \beta \| E_t \|^{\dagger} + (1 - \beta) \| \xi_{S_t} \|^{\dagger} \nonumber
\end{align}$$
The result then follows from Lemma (5) and Proposition (6). $\quad\blacksquare$

---

### 2.2. Convergence bound without weight decay

> **Theorem 8 (Convergence bound without weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda = 0$ (i.e., weight decay is disabled) and step size $\eta > 0$. Then for any norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, there exist constants $X, Y, Z > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        \leq \frac{X}{T} + \frac{Y}{b} + Z
\end{equation}$$
where $T$ is the total number of time steps, $b$ is the batch size, and
$$Y = \frac{(2 \beta + 1 / D^2)(1 - \beta)}{2} \sigma^2.$$
If we instead choose to measure the gradient norm in the Frobenius norm, i.e., $\| \cdot \|_F$, then there exist constants $X_F, Y_F, Z_F > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F] \leq \frac{X_F}{T} + \frac{Y_F}{b} + Z_F
\end{equation}$$
and,
$$\begin{align*}
    X_F &\propto X \\
    Y_F &= \frac{(2 \beta + 1 / D^2)(1 - \beta)}{2} \sigma_F^2 \\
    Z_F &\propto Z
\end{align*}$$

**Proof.** Let us first disable weight decay, i.e., set $\lambda = 0$. Since $f$ is $L$-smooth, the descent lemma, Equation $\eqref{eq:lmo-inner-product}$, and Equation $\eqref{eq:lmo-norm}$ yields,
$$\begin{align}
    f(W_{t+1})
        &\leq f(W_t)
            + \langle \nabla f(W_t), W_{t+1} - W_t \rangle
            + \frac{L}{2} \| W_{t+1} - W_t \|^2 \label{eq:descentlemma} \\
        &\leq f(W_t)
            + \langle \nabla f(W_t), \eta A_t^* \rangle
            + \frac{L}{2} \| \eta A_t^* \|^2 \nonumber \\
        &\leq f(W_t)
            + \eta \langle \nabla f(W_t) - C_t + C_t, A_t^* \rangle
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            + \eta \langle C_t, A_t^* \rangle
            + \eta \langle \nabla f(W_t) - C_t, A_t^* \rangle
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \| C_t \|^{\dagger}
            + \eta \left(
                \frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac {1}{2 \epsilon} \| A_t^* \|^2\right)
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \left(
                \| \nabla f(W_t) \|^{\dagger}
                - \| \nabla f(W_t) - C_t \|^{\dagger}\right) \nonumber \\
            &\qquad+ \frac{\eta \epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac{(1/\epsilon + L\eta)\eta}{2} \nonumber \\
        &\leq f(W_t) - \eta \| \nabla f(W_t) \|^{\dagger}
            + \eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber \\
            &\qquad+ \frac{\eta \epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac{(1/\epsilon + L\eta)\eta}{2} \label{eq:descentlemma-final}
\end{align}$$
Note that the $\langle \cdot, \cdot \rangle$ operator in Equation $\eqref{eq:descentlemma}$ is *not* an inner product, but the canonical pairing between cotangent and tangent spaces ($\nabla f(W_t) \in T_{W_t}^* \mathcal{W}$ while $A_t^* \in T_{W_t}\mathcal{W}$). Under the standard basis of $\mathbb{R}^{m \times n}$, however, it *behaves like* the Frobenius inner product.

Rearranging Equation $\eqref{eq:descentlemma-final}$ then gives,
$$\| \nabla f(W_t) \|^{\dagger}
    \leq \frac{f(W_t) - f(W_{t+1})}{\eta}
        + \| \nabla f(W_t) - C_t \|^{\dagger}
        + \frac{\epsilon}{2} \| \nabla f(W_t) - C_t \|^{\dagger 2}
        + \frac{1/\epsilon + L\eta}{2}$$

**Approach 1: We measure the gradient norm with $\| \cdot \|^{\dagger}$.** Then we can set $\epsilon = \frac{1}{D^2}$. And after taking expectations, and averaging, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &\leq \frac{f(W_0) - f(W_T)}{\eta T}
            + \frac{D^2 + L\eta}{2} \nonumber \\
        &\quad+ \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|^{\dagger}]
            + \frac{1}{2 D^2 T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|^{\dagger 2}] \nonumber \\
        &\leq \frac{f(W_0) - f(W_T)}{\eta T} 
            + \frac{D^2 + L\eta}{2} \nonumber \\
        &\quad+  \left(\frac{2\sqrt{2}\beta}{1 - \beta}\frac{1}{T} \| E_0 \|^{\dagger}
            + \frac{2 \beta}{1 - \beta} L \eta
            + \left(\sqrt{2(1 - \beta)}D\beta + (1 - \beta) \right) \frac{\sigma}{\sqrt{b}} \right) \nonumber \\
        &\quad+ \frac{1}{2 D^2} \left(\frac{2\beta}{1 - \beta}\frac{1}{T} \| E_{0} \|^{\dagger 2}
            + \frac{4 \beta}{(1 - \beta)^2} L^2 \eta^2
            + (2 D^2 \beta + 1) (1 - \beta) \frac{\sigma^2}{b} \right) \nonumber \\
        &\leq \frac{X}{T} + \frac{Y}{b} + Z
\end{align}$$
where,
$$\begin{align}
    X
        &:= \frac{f(W_0) - f^*}{\eta}
            + \frac{2\sqrt{2}\beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta}{D^2 (1 - \beta)} \| \nabla f(W_0) \|^{\dagger 2} \nonumber \\
    Y
        &:= \frac{(2 \beta + 1 / D^2)(1 - \beta)}{2} \sigma^2 \nonumber \\
    Z
        &:= \frac{D^2 + L\eta}{2}
            + \frac{2 \beta}{1 - \beta} L \eta
            + \frac{2\beta}{D^2 (1 - \beta)^2} L^2 \eta^2
            + \left(\sqrt{2 (1 - \beta)} D \beta + (1 - \beta)\right) \frac{\sigma}{\sqrt{b}} \nonumber
\end{align}$$

**Approach 2: We measure the gradient norm with $\| \cdot \|_F$.** By norm equivalence, there exist constants $\kappa_1 > 0, \kappa_2 > 0$ such that for all $X \in \mathbb{R}^{m \times n}$,
$$ \kappa_1 \| X \|_F \leq \| X \|^{\dagger} \leq \kappa_2 \| X \|_F $$
For Muon, we have $\| X \|^{\dagger} = \| X \|_{\text{nuc}}$ (the nuclear norm), and so $\kappa_1 = 1, \kappa_2 \leq \sqrt{\min{(m, n)}}$.

We then set $\epsilon = \frac{\kappa_1}{\kappa_2^2 D^2}$ and substitute the norm equivalence bounds to obtain,
$$\| \nabla f(W_t) \|_F
    \leq \frac{f(W_t) - f(W_{t+1})}{\eta\kappa_1}
        + \frac{\kappa_2}{\kappa_1}\| \nabla f(W_t) - C_t \|_F
        + \frac{1}{2 D^2} \| \nabla f(W_t) - C_t \|_F^2
        + \frac{\kappa_2^2 D^2/\kappa_1 + L_F\eta}{2\kappa_1}$$

After taking expectations, averaging, and using Corollary (7), we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F]
        &\leq \frac{f(W_0) - f(W_T)}{\eta \kappa_1 T}
            + \frac{\kappa_2^2 D^2/\kappa_1 + L_F\eta}{2\kappa_1} \nonumber \\
        &\quad+ \frac{1}{T}\frac{\kappa_2}{\kappa_1}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|_F]
            + \frac{1}{2 D^2 T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|_F^2] \nonumber \\
        &\leq \frac{f(W_0) - f(W_T)}{\eta \kappa_1 T}
            + \frac{\kappa_2^2 D^2/\kappa_1 + L_F\eta}{2\kappa_1} \nonumber \\
        &\quad+ \frac{\kappa_2}{\kappa_1} \left(\frac{2\sqrt{2}\beta}{1 - \beta}\frac{1}{T} \| E_0 \|_F
            + \frac{2 \beta}{1 - \beta} L_F \eta
            + \left(\sqrt{2(1 - \beta)} D \beta + (1 - \beta) \right) \frac{\sigma_F}{\sqrt{b}} \right) \nonumber \\
        &\quad+ \frac{1}{2 D^2} \left(\frac{2\beta}{1 - \beta}\frac{1}{T} \| E_{0} \|_F^2
            + \frac{4 \beta}{(1 - \beta)^2} L_F^2 \eta^2
            + (2 D^2 \beta + 1) (1 - \beta) \frac{\sigma_F^2}{b} \right) \nonumber \\
        &\leq \frac{X_F}{T} + \frac{Y_F}{b} + Z_F
\end{align}$$
where,
$$\begin{align}
    X_F
        &:= \frac{f(W_0) - f^*}{\eta\kappa_1}
            + \frac{2\sqrt{2}\beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} \| \nabla f(W_0) \|_F
            + \frac{\beta}{D^2 (1 - \beta)} \| \nabla f(W_0) \|_F^2 \nonumber \\
    Y_F
        &:= \frac{(2 \beta + 1 / D^2)(1 - \beta)}{2} \sigma_F^2 \nonumber \\
    Z_F
        &:= \frac{\kappa_2^2 D^2/\kappa_1 + L_F\eta}{2\kappa_1}
            + \frac{2 \beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} L_F \eta
            + \frac{2\beta}{D^2 (1 - \beta)^2} L_F^2 \eta^2 \nonumber \\
            &\qquad+ \left(\sqrt{2 (1 - \beta)} D \beta + (1 - \beta)\right) \frac{\kappa_2}{\kappa_1} \frac{\sigma_F}{\sqrt{b}} \quad\blacksquare \nonumber
\end{align}$$

## 3. Convergence bound for steepest descent under arbitrary norms with weight decay

We now analyze the case $\lambda > 0$.

### 3.1. Weight, gradient, and momentum norm bounds

> **Proposition 9 (Weight and gradient bound).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda > 0$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$ and $\| W_0 \| \leq \frac{1}{\lambda}$. Additionally, suppose that there exists a minimizer $W^*$ with $\nabla f(W^*) = 0$ and $\| W^* \| \leq \frac{1}{\lambda}$. Then, for all $t \geq 0$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \| W_t \|
        &\leq \frac{1}{\lambda} \\
    \| \nabla f(W_t) \|^{\dagger}
        &\leq \frac{2L}{\lambda}
\end{align}$$

**Proof.** Let us unroll the recurrence in Equation $\eqref{eq:updateweightdecay}$,
$$\begin{align}
    W_t
        &= (1 - \lambda\eta) W_{t-1} + \eta A_{t-1}^* \nonumber \\
        &= (1 - \lambda\eta)^2 W_{t-2} + \eta (1 - \lambda\eta) A_{t-2}^* + \eta A_{t-1}^* \nonumber \\
        &\;\vdots \nonumber \\
        &= (1 - \lambda\eta)^t W_0 + \eta \sum_{i=0}^{t-1} (1 - \lambda\eta)^i A_{t-1-i}^* \nonumber
\end{align}$$
Taking norms and using the triangle inequality then gives,
$$\begin{align}
    \| W_t \|
        &\leq (1 - \lambda\eta)^t \| W_0 \| + \eta \sum_{i=0}^{t-1} (1 - \lambda\eta)^i \| A_{t-1-i}^* \| \nonumber \\
        &\leq (1 - \lambda\eta)^t \| W_0 \| + \eta \sum_{i=0}^{t-1} (1 - \lambda\eta)^i \nonumber \\
        &\leq (1 - \lambda\eta)^t \| W_0 \| + \frac{\eta}{\lambda\eta} (1 - (1 - \lambda\eta)^t) \nonumber \\
        &\leq \frac{1}{\lambda} \nonumber
\end{align}$$

For the gradient norm bound, we use the fact that $\nabla f(W^*) = 0$ at a minimizer $W^*$, together with the $L$-smoothness of $f$,
$$\begin{align}
    \| \nabla f(W_t) \|^{\dagger}
        &= \| \nabla f(W_t) - 0 \|^{\dagger} \nonumber \\
        &= \| \nabla f(W_t) - \nabla f(W^*) \|^{\dagger} \nonumber \\
        &\leq L \| W_t - W^* \| \nonumber \\
        &\leq L (\| W_t \| + \| W^* \|) \nonumber \\
        &\leq \frac{2L}{\lambda} \quad\blacksquare \nonumber
\end{align}$$

---

Next we bound the *variance* of gradients and momentum terms under weight decay.

> **Proposition 10 (Gradient and (Nesterov) momentum variance bound).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda > 0$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then, for all $t \geq 0$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|^{\dagger 2} \right]
        &\leq \frac{2D\sigma^2}{b} + \frac{8 L^2}{\lambda^2} \\
    \mathbb{E}\left[ \| M_t \|^{\dagger 2} \right]
        &\leq \frac{2D\sigma^2}{b} + \frac{8 L^2}{\lambda^2} \\
    \mathbb{E}\left[ \| C_t \|^{\dagger 2} \right]
        &\leq \frac{2D\sigma^2}{b} + \frac{8 L^2}{\lambda^2}
\end{align}$$

**Proof.** By the triangle inequality and Lemma (5), we have,
$$\begin{align}
    \| \nabla f_{S_t}(W_t) \|^{\dagger}
        &\leq \| \nabla f_{S_t}(W_t) - \nabla f(W_t) \|^{\dagger} + \| \nabla f(W_t) \|^{\dagger} \nonumber \\
    \| \nabla f_{S_t}(W_t) \|^{\dagger 2}
        &\leq 2 \| \nabla f_{S_t}(W_t) - \nabla f(W_t) \|^{\dagger 2} + 2 \| \nabla f(W_t) \|^{\dagger 2} \nonumber \\
    \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|^{\dagger 2} \right]
        &\leq 2 \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) - \nabla f(W_t) \|^{\dagger 2} \right]
            + 2 \| \nabla f(W_t) \|^{\dagger 2} \nonumber \\
        &\leq \frac{2D\sigma^2}{b} + \frac{8 L^2}{\lambda^2} \nonumber
\end{align}$$

Then, let us unroll the momentum recurrence,
$$\begin{align}
    \mathbb{E}\left[ \| M_t \|^{\dagger 2} \right]
        &= \mathbb{E}\left[ \| \beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t) \|^{\dagger 2} \right] \nonumber \\
        &\leq \beta \mathbb{E}\left[ \| M_{t-1} \|^{\dagger 2} \right]
            + (1 - \beta) \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|^{\dagger 2} \right] \nonumber \\
        &\leq \cancel{\beta^t \| M_0 \|^{\dagger 2}}
            + (1 - \beta) \sum_{i=0}^{t-1} \left( \frac{2 D\sigma^2}{b} + \frac{8 L^2}{\lambda^2} \right) \beta^i \nonumber \\
        &\leq \frac{2 D\sigma^2}{b} + \frac{8 L^2}{\lambda^2} \nonumber
\end{align}$$

As for the Nesterov momentum term, we have,
$$\begin{align}
    \mathbb{E}\left[ \| C_t \|^{\dagger 2} \right]
        &= \mathbb{E}\left[ \| \beta M_t + (1 - \beta) \nabla f_{S_t}(W_t) \|^{\dagger 2} \right] \nonumber \\
        &\leq \beta \mathbb{E}\left[ \| M_t \|^{\dagger 2} \right] + (1 - \beta) \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|^{\dagger 2} \right] \nonumber \\
        &\leq \frac{2 D\sigma^2}{b} + \frac{8 L^2}{\lambda^2} \quad\blacksquare \nonumber
\end{align}$$

---

### 3.2. Convergence bound with weight decay

> **Theorem 11 (Convergence bound with weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then for any norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, there exist constants $X, Y, Z > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \leq \frac{X}{T} + \frac{Y}{b} + Z
\end{equation}$$
where $T$ is the total number of time steps, $b$ is the batch size, and
$$Y = \frac{(2 \beta + 1 / D^2)(1 - \beta) + \lambda}{2} \sigma^2.$$
If we instead choose to measure the gradient norm in the Frobenius norm, i.e., $\| \cdot \|_F$, then there exist constants $X_F, Y_F, Z_F > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F] \leq \frac{X_F}{T} + \frac{Y_F}{b} + Z_F
\end{equation}$$
and,
$$\begin{align*}
    X_F &\propto X \\
    Y_F &= \frac{(2 \beta + 1 / D^2)(1 - \beta) + \lambda}{2} \sigma_F^2 \\
    Z_F &\propto Z
\end{align*}$$

**Proof.** We closely follow that of Theorem (8), with additional terms to account for weight decay.

$$\begin{align}
    f(W_{t+1})
        &\leq f(W_t)
            + \langle \nabla f(W_t), W_{t+1} - W_t \rangle
            + \frac{L}{2} \| W_{t+1} - W_t \|^2 \nonumber \\
        &\leq f(W_t)
            + \langle \nabla f(W_t), \eta A_t^* - \lambda\eta W_{t} \rangle
            + \frac{L}{2} \| \eta A_t^* - \lambda\eta W_{t} \|^2 \nonumber \\
        &\leq f(W_t)
            + \eta \langle \nabla f(W_t) - C_t + C_t, A_t^* - \lambda W_{t} \rangle
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            + \eta \langle C_t, A_t^* \rangle
            + \lambda\eta \langle C_t, -W_{t} \rangle
            + \eta \langle \nabla f(W_t) - C_t, A_t^* - \lambda W_{t} \rangle
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \| C_t \|^{\dagger}
            + \lambda\eta \left(
                \frac{\epsilon'}{2} \| C_t \|^{\dagger 2}
                + \frac{1}{2\epsilon'} \| -W_t \|^2 \right) \nonumber \\
        &\qquad+ \eta\left(
                \frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac {1}{2 \epsilon} \| A_t^* - \lambda W_{t} \|^2\right)
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \left(\| \nabla f(W_t) \|^{\dagger} - \| \nabla f(W_t) - C_t \|^{\dagger}\right)
            + \frac{\lambda\eta\epsilon'}{2} \| C_t \|^{\dagger 2}
            + \frac{\lambda\eta}{2\epsilon'} \| W_t \|^2
            \nonumber \\
        &\qquad+ \frac{\eta\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
            + \frac {\eta}{2 \epsilon} \left(2\| A_t^* \|^2 + 2\lambda^2 \| W_{t} \|^2 \right)
            + \frac{L\eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \| \nabla f(W_t) \|^{\dagger}
            + \eta \| \nabla f(W_t) - C_t \|^{\dagger}
            + \frac{\eta\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2} \nonumber \\
        &\qquad + \frac{\lambda\eta\epsilon'}{2} \| C_t \|^{\dagger 2}
            + \frac{\lambda\eta(2\lambda/\epsilon + 1/\epsilon')}{2} \| W_t \|^2
            + \frac{(2/\epsilon + L\eta)\eta}{2}
\end{align}$$

Rearranging then gives,
$$\begin{align}
    \| \nabla f(W_t) \|^{\dagger}
        &\leq \frac{f(W_t) - f(W_{t+1})}{\eta}
            + \| \nabla f(W_t) - C_t \|^{\dagger}
            + \frac{\epsilon}{2} \| \nabla f(W_t) - C_t \|^{\dagger 2} \nonumber \\
        &\quad
            + \frac{\lambda\epsilon'}{2} \| C_t \|^{\dagger 2}
            + \frac{\lambda(2\lambda/\epsilon + 1/\epsilon')}{2} \| W_t \|^2
            + \frac{2/\epsilon + L\eta}{2} \nonumber
\end{align}$$

**Approach 1: We measure the gradient norm with $\| \cdot \|^{\dagger}$.** Then we set $\epsilon = \frac{1}{D^2}$ and $\epsilon' = \frac{1}{2 D}$. Following the same strategy as in Theorem (8) with Proposition (9) and Proposition (10) then yields,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &\leq \frac{X}{T} + \frac{Y}{b} + Z
\end{align}$$
where,
$$\begin{align}
    X
        &:= \frac{f(W_0) - f^*}{\eta}
            + \frac{2\sqrt{2}\beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta}{D^2 (1 - \beta)} \| \nabla f(W_0) \|^{\dagger 2} \nonumber \\
    Y
        &:= \frac{(2 \beta + 1 / D^2)(1 - \beta) + \lambda}{2} \sigma^2 \nonumber \\
    Z
        &:= \frac{2 \beta}{1 - \beta} L \eta
            + \frac{2\beta}{D^2 (1 - \beta)^2} L^2 \eta^2
            + \frac{2 L^2}{\lambda D}
            + \frac{\lambda D^2 + D}{\lambda} \nonumber \\
        &\qquad
            + \frac{2 D^2 + L\eta}{2}
            + \left(\sqrt{2 (1 - \beta)} D \beta + (1 - \beta)\right) \frac{\sigma}{\sqrt{b}} \nonumber
\end{align}$$

**Approach 2: We measure the gradient norm with $\| \cdot \|_F$.** We set $\epsilon = \frac{\kappa_1}{\kappa_2^2 D^2}$ and $\epsilon' = \frac{\kappa_1}{2 \kappa_2^2 D}$, and substitute the norm equivalence bounds to obtain,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f(W_t) \|_F \right]
        &\leq \frac{f(W_t) - f(W_{t+1})}{\eta\kappa_1}
            + \frac{\kappa_2}{\kappa_1} \mathbb{E}\left[ \| \nabla f(W_t) - C_t \|_F \right]
            + \frac{1}{2 D^2} \mathbb{E}\left[ \| \nabla f(W_t) - C_t \|_F^2 \right] \nonumber \\
        &\quad
            + \frac{\lambda}{4 D} \left( \frac{2 D \sigma_F^2}{b} + \frac{8 L_F^2}{\lambda^2 \kappa_1} \right)
            + \frac{\kappa_2^2}{\kappa_1^2}\frac{\lambda D^2 + D}{\lambda}
            + \frac{2 \kappa_2^2 D^2/\kappa_1 + L_F\eta}{2\kappa_1} \nonumber \\
\end{align}$$

And after taking expectations and averaging, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F]
        &\leq \frac{X_F}{T} + \frac{Y_F}{b} + Z_F
\end{align}$$
where,
$$\begin{align}
    X_F
        &:= \frac{f(W_0) - f^*}{\eta\kappa_1}
            + \frac{2\sqrt{2}\beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} \| \nabla f(W_0) \|_F
            + \frac{\beta}{D^2 (1 - \beta)} \| \nabla f(W_0) \|_F^2 \nonumber \\
    Y_F
        &:= \frac{(2 \beta + 1 / D^2)(1 - \beta) + \lambda}{2} \sigma_F^2 \nonumber \\
    Z_F
        &:= \frac{2 \beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} L_F \eta
            + \frac{2\beta}{D^2 (1 - \beta)^2} L_F^2 \eta^2
            + \frac{2 L_F^2}{\lambda \kappa_1 D}
            + \frac{\kappa_2^2}{\kappa_1^2}\frac{\lambda D^2 + D}{\lambda} \nonumber \\
        &\qquad
            + \frac{2\kappa_2^2 D^2/\kappa_1 + L_F\eta}{2\kappa_1}
            + \left(\sqrt{2 (1 - \beta)} D \beta + (1 - \beta)\right) \frac{\kappa_2}{\kappa_1} \frac{\sigma_F}{\sqrt{b}} \quad\blacksquare \nonumber
\end{align}$$

---

## 4. Deriving the critical batch size

> **Theorem 12 (Critical batch size for steepest descent under arbitrary norms with (Nesterov) momentum and weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then for an arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, the critical batch size $b_{crit}$ that minimizes the total number of tokens processed to reach $\epsilon$-convergence according to the criterion in Equation $\eqref{eq:convergence-criterion}$ is given by,
$$\begin{equation}
    b_{crit} = \left( (2\beta + 1 / D^2)(1 - \beta) + \lambda \right) \frac{\sigma^2}{\epsilon'} \label{eq:critical-batch-size}
\end{equation}$$
where $\epsilon' := \epsilon - Z > 0$, for some constant $Z$ defined in Theorem (11).

**Proof.** We consider the steepest descent iteration process to have converged at time step $T$ when, for some $\epsilon > 0$,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \leq \frac{X}{T} + \frac{Y}{b} + Z \leq \epsilon \label{eq:convergence-criterion}
\end{equation}$$
Since $Z$ is a constant independent of $T$ and $b$, we can simply fold it into $\epsilon$ by defining $\epsilon' := \epsilon - Z > 0$. Simple algebra then yields the number of iterations to satisfy the convergence criterion in Equation $\eqref{eq:convergence-criterion}$ as,
$$\begin{align}
    \frac{X}{T} + \frac{Y}{b} + Z &\leq \epsilon \nonumber \\
    \frac{X}{T} + \frac{Y}{b} &\leq \epsilon - Z =: \epsilon' \nonumber \\
    \frac{Xb}{T} + Y &\leq \epsilon' b \nonumber \\
    \frac{Xb}{\epsilon' b - Y} &\leq T \nonumber \\
    \frac{Xb}{\epsilon' b - Y} &=: T(b)
\end{align}$$
Note that we also have to constrain $b > \frac{Y}{\epsilon'}$ to ensure that $T(b) > 0$. Taking the first and second derivatives then yields,
$$\begin{align}
    \frac{dT(b)}{db} &= -\frac{XY}{(\epsilon' b - Y)^2} \leq 0 \nonumber \\
    \frac{d^2T(b)}{db^2} &= \frac{2XY\epsilon'}{(\epsilon' b - Y)^3} \geq 0 \nonumber
\end{align}$$
Thus, $T(b)$ is a monotonically decreasing and convex function for $b > \frac{Y}{\epsilon'}$.

Now, the number of tokens we need to process to reach convergence is roughly proportional to,
$$b \cdot T(b) = \frac{Xb^2}{\epsilon' b - Y}$$
Taking the first and second derivatives again yields,
$$\begin{align}
    \frac{d(b \cdot T(b))}{db} &= \frac{Xb(\epsilon' b - 2Y)}{(\epsilon' b - Y)^2} \nonumber \\
    \frac{d^2(b \cdot T(b))}{db^2} &= \frac{2XY^2}{(\epsilon' b - Y)^3} \geq 0 \nonumber
\end{align}$$
Thus, $b \cdot T(b)$ is a convex function for $b > \frac{Y}{\epsilon'}$, with a minimizer $b^* = \frac{2Y}{\epsilon'}$. This gives us the critical batch size,
$$b_{crit} = \frac{2Y}{\epsilon'} = \left( (2\beta + 1 / D^2)(1 - \beta) + \lambda \right) \frac{\sigma^2}{\epsilon'}\quad\blacksquare$$

---

### 4.1. Estimating D-smoothness for various optimizers

Optimizers we use in practice can be viewed as performing steepest descent under different norms [(Bernstein et al., 2024)](https://arxiv.org/abs/2409.20325). We summarize the relevant norm choices and their corresponding $D$-smoothness constants below.

| Optimizer | Steepest descent norm              | Dual norm                  |
| --------- | ---------------------------------- | -------------------------- |
| SGD       | $\| \cdot \|_F$                    | $\| \cdot \|_F$            |
| SignSGD   | $\| \cdot \|_{\infty}$             | $\| \cdot \|_{1}$          |
| AdamW     | $\| \cdot \|_{\infty}$ (adaptive)  | $\| \cdot \|_{1}$          |
| Muon      | $\| \cdot \|_{2 \to 2}$            | $\| \cdot \|_{\text{nuc}}$ |
| SOAP      | $\| \cdot \|_{2 \to 2}$ (adaptive) | $\| \cdot \|_{\text{nuc}}$ |

We can then use the following JAX code to estimate the $D$-smoothness for steepest descent under various norms. This typically results in $D \approx 1$ for SignSGD/AdamW and Muon/SOAP. And thus, we can typically approximate the critical batch size formula as,
$$\begin{align}
    b_{crit} &\approx ((2\beta + 1)(1 - \beta) + \lambda) \frac{\sigma^2}{\epsilon'} \nonumber \\
\end{align}$$

```python
import jax
import jax.numpy as jnp

def lipschitz_estimate(grad_g, norm_fn, dual_norm_fn, lmo, key, shape, n_pairs=10000, radius=1.0):
    def one_ratio(key):
        k1, k2 = jax.random.split(key)
        # The LMO guarantees that W1, W2 are on the unit ball of the norm
        W1 = radius * lmo(jax.random.normal(k1, shape))
        W2 = radius * lmo(jax.random.normal(k2, shape))

        g1 = grad_g(W1)
        g2 = grad_g(W2)

        num = norm_fn((g1 - g2))
        denom = dual_norm_fn((W1 - W2))
        return num / denom

    keys = jax.random.split(key, n_pairs)
    ratios = jax.vmap(one_ratio)(keys)
    return jnp.max(ratios), jnp.mean(ratios)

def f_inf_norm(W):
    return jnp.max(jnp.abs(W))

def f1_norm(W):
    return jnp.sum(jnp.abs(W))

def spectral_norm(W):
    s = jnp.linalg.svd(W, compute_uv=False)
    return s.max()

def nuclear_norm(W):
    s = jnp.linalg.svd(W, compute_uv=False)
    return jnp.sum(s)

def orthogonalize(W):
    u, s, vh = jnp.linalg.svd(W, full_matrices=False)
    return u @ vh

def g(W, norm_fn):
    return 0.5 * norm_fn(W)**2

grad_g_f1 = jax.grad(lambda W: g(W, f1_norm))
grad_g_nuclear = jax.grad(lambda W: g(W, nuclear_norm))

key = jax.random.PRNGKey(0)
m, n = 128, 32
shape = (m, n)
n_pairs = 1000
print(m*n, float(lipschitz_estimate(grad_g_f1, f_inf_norm, f1_norm, jnp.sign, key, shape, n_pairs=n_pairs)[0]))
print(min(m, n), float(lipschitz_estimate(grad_g_nuclear, spectral_norm, nuclear_norm, orthogonalize, key, shape, n_pairs=n_pairs)[0]))
```

## 5. Discussion

The main result of this work is that the *shape* of the convergence bound:
$$\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] = \frac{X}{T} + \frac{Y}{b} + Z$$
is universal across all norms used for steepest descent, with only the constants $X$ and $Z$ being sensitive to the choice of norm. As a consequence, the critical batch size formula:
$$b_{crit} \approx \left( (2\beta + 1)(1 - \beta) + \lambda \right) \frac{\sigma^2}{\epsilon'}$$
also holds universally across all norms. This matches prior results by [Sato et al. (2025)](https://arxiv.org/abs/2507.01598). Intuitively, this means that first-order optimization does not ‘favor’ any particular norm in terms speed of convergence nor performance with respect to batch size scaling.

## Acknowledgements

Big thanks to Antonio Silveti-Falls and Volkan Cevher for providing helpful feedback on an earlier draft of this work.

## How to cite

```bibtex
@misc{cesista2025sdcbs,
  author = {Franz Louis Cesista},
  title = {Critical Batch Size for Steepest Descent Under Arbitrary Norms},
  year = {2025},
  month = {November},
  day = {22},
  url = {https://leloykun.github.io/ponder/steepest-descent-crit-bz/},
}
```

## References

1. Naoki Sato, Hiroki Naganuma, Hideaki Iiduka (2025). Convergence Bound and Critical Batch Size of Muon Optimizer. URL https://arxiv.org/abs/2507.01598
2. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
3. Jeremy Bernstein, Laker Newhouse (2024). Old Optimizer, New Norm: An Anthology. URL https://arxiv.org/abs/2409.20325
4. Simo Ryu (2025). Empirical observation that AdamW, Shampoo, and Muon follow the lr ~ sqrt(batch size) scaling rule on X/Twitter. URL https://x.com/cloneofsimo/status/1907731069878825400

## Appendix

### A1. How to scale learning rate with batch size

In practice, it is often best to scale the learning rate $\eta$ as $\eta \propto \sqrt{b}$ when increasing the batch size $b$. Here we provide a mathematical justification *why*. The crux is that increasing the batch size reduces the gradient noise variance, which in turn means that we can make larger weight updates without destabilizing training.

To see this, we first make the following assumption.

> **Assumption A1.13 (Local Lipschitzness of LMO).** Let $\texttt{LMO}_{\| \cdot \|}$ be the linear minimization oracle with respect to an arbitrary norm $\| \cdot \|$. Then there exists a constant $L_{\text{LMO}} > 0$ such that for $C_1, C_2$ denoting nesterov momentum terms, we have,
$$\begin{equation}
    \| \texttt{LMO}_{\| \cdot \|}(C_1) - \texttt{LMO}_{\| \cdot \|}(C_2) \| \leq L_{\text{LMO}} \| C_1 - C_2 \|^{\dagger}
\end{equation}$$

> **Proposition A1.14 (Gradient noise variance is proportional to $\eta^2/b$).** Let $\eta > 0$ be the learning rate and $b \geq 1$ be the batch size. Under Assumptions 1-3 and Assumption (A1.13), we have,
$$\begin{equation}
    \mathbb{E} \left[ \| \Delta W_t^{\text{noise}} \|^2 \right] \propto \frac{\eta^2}{b}
\end{equation}$$

**Proof.** We can decompose our weight update rule in Equation $\eqref{eq:updateweightdecay}$ into deterministic and stochastic components as follows,
$$\begin{equation}
    \nabla W_t = W_{t+1} - W_t =  \underbrace{-\lambda\eta W_t + \eta A_t^{\text{det}}}_{\Delta W_t^{\text{det}}} + \underbrace{\eta A_t^{\text{noise}}}_{\Delta W_t^{\text{noise}}}
\end{equation}$$
where $A_t^* = A_t^{\text{det}} + A_t^{\text{noise}}$ is the decomposition of the steepest descent direction into its deterministic and stochastic components.

Taking norms and expectations, and using Proposition (6) then yields,
$$\begin{align}
    \mathbb{E} \left[ \| \Delta W_t^{\text{noise}} \|^2 \right]
        &= \eta^2 \mathbb{E} \left[ \| A_t^{\text{noise}} \|^2 \right] \nonumber \\
        &= \eta^2 \mathbb{E} \left[ \| A_t^* - A_t^{\text{det}} \|^2 \right] \nonumber \\
        &\lesssim \eta^2 L_{\text{LMO}}^2 \mathbb{E} \left[ \| C_t - \nabla f(W_t) \|^{\dagger 2} \right] \nonumber \\
        &\lesssim 2 \eta^2 L_{\text{LMO}}^2 (1 - \beta) D^2 \frac{\sigma^2}{b} + O\left(\frac{1}{T} + 1 \right) \nonumber \\
        &\propto \frac{\eta^2}{b} \quad\blacksquare \nonumber
\end{align}$$

Now, if we already know that training is stable for some gradient noise variance level $\mathbb{E} \left[ \| \Delta W_t^{\text{noise}} \|^2 \right]$, then it is natural to preserve it as we scale the batch size $b$. Thus, we have,
$$\begin{align}
    \frac{\eta_{\text{new}}^2}{b_{\text{new}}}
        &= \frac{\eta_{\text{old}}^2}{b_{\text{old}}} = \text{constant} \nonumber \\
    \eta_{\text{new}}
        &= \eta_{\text{old}}\sqrt{\frac{b_{\text{new}}}{b_{\text{old}}}}. \nonumber
\end{align}$$
This means that, e.g., if we $4\times$ the batch size, then increasing the learning rate by a factor of $2$ preserves training stability. This matches empirical observations first reported by [Ryu (2025)](https://x.com/cloneofsimo/status/1907731069878825400).
