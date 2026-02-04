---
title: "Critical Batch Size for Steepest Descent Under Arbitrary Norms"
date: 2025-11-22
tags: ["Machine Learning", "Optimizers"]
author: ["Franz Louis Cesista", "Kaiyue Wen"]
description: "First-order optimization under arbitrary norms with Nesterov momentum (and decoupled weight decay) yields a universal critical batch size formula. The square root learning rate scaling rule with batch size also holds universally across all norms."
summary: "First-order optimization under arbitrary norms with Nesterov momentum (and decoupled weight decay) yields a universal critical batch size formula. The square root learning rate scaling rule with batch size also holds universally across all norms."
---

## 0. Abstract

This work generalizes prior results by [Sato et al. (2025)](https://arxiv.org/abs/2507.01598) on the critical batch size for the Muon optimizer [(Jordan et al., 2025)](https://kellerjordan.github.io/posts/muon/) to steepest descent under arbitrary norms with Nesterov momentum and weight decay. We show that (1) the same critical batch size formula, and (2) the square root learning rate scaling rule with batch size, holds universally across all norms. These results are useful for large-scale LLM training because they reduce the need for expensive hyperparameter tuning when switching between different optimizers and when scaling up batch sizes.

## 1. Introduction and preliminaries

We consider the following optimization problem:
$$\begin{equation}
    W_* = \arg\min_{W \in \mathcal{W}} f(W) \label{eq:opt-problem}
\end{equation}$$
where $f(\cdot): \mathcal{W} \to \mathbb{R}$ is a bounded from below and differentiable objective function, and $\mathcal{W}$ is a finite-dimensional vector space over $\mathbb{R}$, e.g., $\mathcal{W} = \mathbb{R}^{m \times n}$, equipped with an arbitrary norm $\| \cdot \|$ and its dual norm $\| \cdot \|^{\dagger}$.

More generally, we often take $\mathcal{W}$ to be a product of layers' weight spaces, e.g.,
$$\begin{equation}
    \mathcal{W} = \prod_{l=1}^{L} \mathbb{R}^{m_l \times n_l},
\end{equation}$$
for an $L$-layer neural network with weight matrices $(W^{(l)})_{l=1}^L$ where $W^{(l)} \in \mathbb{R}^{m_l \times n_l}$ for each layer $l$. Given layer-wise norms $\| \cdot \|_{(l)}$ and their duals $\| \cdot \|_{(l)}^{\dagger}$, we can then define the product norm and its dual as,
$$\begin{align}
    \| W \| &:= h\left( \| W^{(1)} \|_{(1)}, \| W^{(2)} \|_{(2)}, \ldots, \| W^{(L)} \|_{(L)} \right) \nonumber \\
    \| G \|^{\dagger} &:= h^{\dagger}\left( \| G^{(1)} \|_{(1)}^{\dagger}, \| G^{(2)} \|_{(2)}^{\dagger}, \ldots, \| G^{(L)} \|_{(L)}^{\dagger} \right) \nonumber
\end{align}$$
for some vector norm $h$ and its dual $h^{\dagger}$ on $\mathbb{R}^L$. Our results still hold under this more general setting.

Now, at iteration $t$, we sample an i.i.d. minibatch $S_t = \{ i_1, i_2, \ldots, i_b \}$ of size $b$ from the training dataset. For each data point $i$, we write the per-example stochastic gradient as,
$$\begin{equation}
    G_{\xi_{t, i}}(W_t) := \nabla f(W_t) - \xi_{t, i},
\end{equation}$$
where $\xi_{t,i}$ is the (additive) gradient noise at $(t, i)$. We then write the minibatch stochastic gradient and noise as,
$$\begin{align}
    \nabla f_{S_t}(W_t)
        &:= \frac{1}{b}\sum_{i=1}^{b} G_{\xi_{t,i}}(W_t) \label{eq:def_minibatch_gradient} \\
    \xi_{S_t}
        &:= \nabla f(W_t) - \nabla f_{S_t}(W_t)
\end{align}$$

### 1.1. First-order steepest descent with Nesterov momentum and (decoupled) weight decay

We solve Equation $\eqref{eq:opt-problem}$ by iteratively minimizing a first-order Taylor approximation of $f$ around the current weight $W_t$. From here, we can either impose a hard constraint on the step size under the norm $\| \cdot \|$, or add a quadratic regularization term.
$$\begin{align}
    \text{[Constrained]}^{(1)} \quad
    W_{t+1}
        &= \arg\min_{W \in \mathcal{W}} \left\{ f(W_t) + \left\langle \nabla f_{S_t}(W_t), W - W_t \right\rangle_F \right\}
        \quad \text{s.t.} \quad \| W - W_t \| \leq \eta\\
    \text{[Regularized]}^{(1)} \quad
    W_{t+1}
        &= \arg\min_{W \in \mathcal{W}} \left\{ f(W_t) + \left\langle \nabla f_{S_t}(W_t), W - W_t \right\rangle_F + \frac{1}{2\eta} \| W - W_t \|^2 \right\}
\end{align}$$

We can also reduce gradient variance by using Nesterov momentum defined as,
$$\begin{align}
    M_t &= \beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t) \\
    C_t &= \beta M_t + (1 - \beta) \nabla f_{S_t}(W_t) \\
\end{align}$$
where $\beta$ is the momentum hyperparameter, $M_t$ is the usual momentum accumulator, and $C_t$ is the Nesterov "look-ahead" gradient. We then have,
$$\begin{align}
    \text{[CSD]}^{(2)} \quad
    W_{t+1}
        &= \arg\min_{W \in \mathcal{W}} \left\{ f(W_t) + \left\langle C_t, W - W_t \right\rangle_F \right\}
        \quad \text{s.t.} \quad \| W - W_t \| \leq \eta\\
    \text{[RSD]}^{(2)} \quad
    W_{t+1}
        &= \arg\min_{W \in \mathcal{W}} \left\{ f(W_t) + \left\langle C_t, W - W_t \right\rangle_F + \frac{1}{2\eta} \| W - W_t \|^2 \right\}
\end{align}$$
To prevent the weights from blowing up, we can also add a decoupled weight decay term with coefficient $\lambda \geq 0$ by "shifting" the center of the constraint/regularization from $W_t$ to $(1 - \lambda\eta) W_t$ as follows,
$$\begin{align}
    \text{[CSD]}^{(3)} \quad
    W_{t+1}
        &= \arg\min_{W \in \mathcal{W}} \left\{ f(W_t) + \left\langle C_t, W - (1 - \lambda\eta) W_t \right\rangle_F \right\}
        \quad \text{s.t.} \quad \| W - (1 - \lambda\eta) W_t \| \leq \eta\\
    \text{[RSD]}^{(3)} \quad
    W_{t+1}
        &= \arg\min_{W \in \mathcal{W}} \left\{ f(W_t) + \left\langle C_t, W - (1 - \lambda\eta) W_t \right\rangle_F + \frac{1}{2\eta} \| W - (1 - \lambda\eta) W_t \|^2 \right\}
\end{align}$$

Solving the above problems then yields the following update rules,
$$\begin{align}
    \text{[CSD]}^{(3)} \quad
    W_{t+1}
        &= (1 - \lambda\eta) W_t + \eta \texttt{LMO}_{\| \cdot \|}(C_t) \label{eq:updateweightdecay} \\
    \text{[RSD]}^{(3)} \quad
    W_{t+1}
        &= (1 - \lambda\eta) W_t + \eta \| C_t \|^{\dagger} \texttt{LMO}_{\| \cdot \|}(C_t)
\end{align}$$
where $\texttt{LMO}_{\| \cdot \|}(\cdot)$ is the linear minimization oracle under the norm $\| \cdot \|$ defined as,
$$\begin{equation}
    A_t^*
        := \texttt{LMO}_{\| \cdot \|}(C_t)
        := \arg\min_{A \in \mathbb{R}^{m \times n}} \langle C_t, A \rangle_F \quad \text{ s.t. } \quad \| A \| \leq 1
\end{equation}$$
which has the following useful properties,
$$\begin{align}
    \| A_t^* \|
        &\leq 1 \label{eq:lmo-norm} \\
    \langle C_t, A_t^* \rangle_F
        &= \langle C_t, \texttt{LMO}_{\| \cdot \|}(C_t) \rangle_F \nonumber \\
        &= \arg\min_{A \leq 1} \langle C_t, A \rangle_F \nonumber \\
        &= -\arg\max_{A \leq 1} \langle C_t, A \rangle_F \nonumber \\
        &= - \| C_t \|^{\dagger}. \label{eq:lmo-inner-product}
\end{align}$$

For this work, we will focus on the constrained steepest descent with Nesterov momentum and decoupled weight decay ([CSD]$^{(3)}$) with update rule given by Equation $\eqref{eq:updateweightdecay}$.

### 1.2. Assumptions

> **Assumption 1 (Unbiased gradient noise, per sample).** At each time step $t$ and for each data point $i \in S_t$, the gradient noise satisfies,
$$\begin{equation} \mathbb{E}\left[ \xi_{t, i} | W_t \right] = 0, \end{equation}$$
and the samples $(\xi_{t,i})_{i=1}^b$ are conditionally independent given $W_t$. To simplify notation, we will often omit the conditioning on $W_t$ when it is clear from context.

> **Assumption 2 (Bounded gradient noise variance).** There exists $\sigma > 0$ such that for all $t, i$,
$$\begin{equation}
    \mathbb{E}\left[\| \xi_{t,i} \|^{\dagger 2} \right] \leq \sigma^2
\end{equation}$$

> **Assumption 3 (L-smoothness of $f$ under $(\| \cdot \|, \| \cdot \|^{\dagger})$).** There exists $L > 0$ such that for all $X, Y \in \mathcal{W}$,
$$\begin{equation}
    \| \nabla f(Y) - \nabla f(X) \|^{\dagger} \leq L \| Y - X \|
\end{equation}$$

> **Assumption 4 (Local D-smoothness of $g(\cdot) = \frac{1}{2}\| \cdot \|^{\dagger 2}$ in the noise region).** There exists a large enough $R > 0$ such that $\mathbb{P}(\| \xi_{t,i} \|^{\dagger} \leq R) = 1$ for all $t, i$. Let,
$$\begin{align}
    K &:= \{ X^{\dagger} \in \mathcal{W}^{\dagger} : \| X^{\dagger} \|^{\dagger} \leq R \} \\
    g(X^{\dagger}) &:= \frac{1}{2} \| X^{\dagger} \|^{\dagger 2} \quad \forall X^{\dagger} \in K
\end{align}$$
Intuitively, $K$ is the region where the gradient noise (and interpolations thereof) lie almost surely. Then there exists $D > 0$ such that for all $X^{\dagger}, Y^{\dagger} \in K$,
$$\begin{equation}
    \| \nabla g(Y^{\dagger}) - \nabla g(X^{\dagger}) \| \leq D \| Y^{\dagger} - X^{\dagger} \|^{\dagger}
\end{equation}$$
> Note that if $\| \cdot \|^{\dagger}$ is induced by an inner product, then $D = 1$.

## 2. Convergence bound for steepest descent under arbitrary norms without weight decay

### 2.1. Gradient noise and momentum error bounds

We first control the variance of the mini-batch noise.

> **Lemma 5 (Minibatch gradient noise bounds).** Under Assumptions (1), (2), and (4), for arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, and sequence of coefficients $(\alpha_i)_{i=1}^k$ with $\alpha_i \geq 0$ and $\sum_{i=0}^k \alpha_i \leq 1$, we have,
$$\begin{align}
    \mathbb{E}\left[ \left\| \sum_{i=0}^k \alpha_{i} \xi_{i} \right\|^{\dagger 2} \right]
        &\leq D \sigma^2 \sum_{i=0}^k \alpha_{i}^2
\end{align}$$
In particular, for minibatch size $b \geq 1$,
$$\begin{align}
    \mathbb{E}\left[ \| \xi_{S_t} \|^{\dagger 2} \right]
        &\leq \frac{D\sigma^2}{b} \label{eq:minibatchvariance}
\end{align}$$

**Proof.** Let $S_{k} = \sum_{i=1}^{k} \alpha_{i} \xi_{i}$ be the partial (weighted) sum of the first $k$ noise terms. Since $\sum_{i=1}^k \alpha_i \leq 1$, we know that $S_k \in K$ almost surely by Assumption (4). Applying the descent lemma on $g(\cdot) = \frac{1}{2}\| \cdot \|^{\dagger 2}$, taking expectations, and using Assumption (1) then gives,
$$\begin{align}
    g(S_{k})
        &\leq g(S_{k-1})
            + \langle \nabla g(S_{k-1}), \alpha_{k} \xi_{k} \rangle
            + \frac{D}{2} \| \alpha_{k} \xi_{k} \|^{\dagger 2} \nonumber \\
    \frac{1}{2} \| S_{k} \|^{\dagger 2}
        &\leq \frac{1}{2} \| S_{k-1} \|^{\dagger 2}
            + \alpha_{k} \langle \nabla g(S_{k-1}), \xi_{k} \rangle
            + \frac{D}{2} \alpha_{k}^2 \| \xi_{k} \|^{\dagger 2} \nonumber \\
    \mathbb{E}\left[ \| S_{k} \|^{\dagger 2} \right]
        &\leq \mathbb{E}\left[ \| S_{k-1} \|^{\dagger 2} \right]
            + \cancel{2 \alpha_{k} \left\langle \nabla g(S_{k-1}), \mathbb{E}\left[ \xi_{k} \right] \right\rangle}
            + D \alpha_{k}^2 \mathbb{E}\left[ \| \xi_{k} \|^{\dagger 2} \right] \nonumber \\
        &\leq \mathbb{E}\left[ \| S_{k-1} \|^{\dagger 2} \right]
            + D \alpha_{k}^2 \mathbb{E}\left[ \| \xi_{k} \|^{\dagger 2} \right] \nonumber
\end{align}$$
Unrolling the recurrence, and using Assumption (2) then gives,
$$\begin{align}
    \mathbb{E}[ \| S_{k} \|^{\dagger 2} ]
        &\leq D \sum_{i=1}^k \alpha_{i}^2 \mathbb{E}[ \| \xi_{i} \|^{\dagger 2} ]
        \leq D \sigma^2 \sum_{i=1}^k \alpha_{i}^2 \nonumber
\end{align}$$
Finally, setting $\alpha_{i} = \frac{1}{b}$ for all $i$ then gives Equation $\eqref{eq:minibatchvariance}. \quad\blacksquare$

---

We then bound the (first) momentum error and the Nesterov momentum error terms.

> **Proposition 6 (Expected momentum error bounds w/o weight decay).** Let the momentum parameter be $\beta \in [0, 1)$, learning rate $\eta > 0$, and initial momentum $M_0 = 0$. Then under Assumptions (1)-(4), arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, and $t \geq 0$,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f(W_t) - M_t \|^{\dagger} \right]
        &\leq \beta^t \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta}{1 - \beta} L \eta
            + \sqrt{\frac{1 - \beta}{1 + \beta}} \frac{\sqrt{D}\sigma}{\sqrt{b}}
\end{align}$$
Moreover, averaging over $T$ iterations yields,
$$\begin{align}
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[ \| \nabla f(W_t) - M_t \|^{\dagger} \right]
        &\leq \frac{1}{1 - \beta}\frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta}{1 - \beta} L \eta
            + \sqrt{\frac{1 - \beta}{1 + \beta}} \frac{\sqrt{D}\sigma}{\sqrt{b}}
\end{align}$$

**Proof.** Let us define the momentum error term at time $t$ as,
$$\begin{equation}
    E_t := \nabla f(W_t) - M_t
\end{equation}$$
Unrolling then gives,
$$\begin{align}
    E_t
        &= \nabla f(W_t) - M_t \nonumber \\
        &= \nabla f(W_t) - (\beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta (\nabla f(W_t) - M_{t-1}) + (1 - \beta)\xi_{S_t} \nonumber \\
        &= \beta (\nabla f(W_t) - \nabla f(W_{t-1}) + \nabla f(W_{t-1}) - M_{t-1}) + (1 - \beta)\xi_{S_t} \nonumber \\
        &= \beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1})) + (1 - \beta) \xi_{S_t} \nonumber \\
        &= \underbrace{\beta^t E_0
            + \sum_{k=1}^t \beta^{t-k+1} (\nabla f(W_k) - \nabla f(W_{k-1}))}_{E_t^{\text{drift}}}
            + \underbrace{\sum_{k=1}^t \beta^{t-k}(1 - \beta)\xi_{S_k}}_{E_t^{\text{noise}}} \nonumber \\
        &= E_t^{\text{drift}} + E_t^{\text{noise}} \label{eq:momentum-error-decomposition}
\end{align}$$
with $E_0^{\text{drift}} = \nabla f(W_0)$ and $E_0^{\text{noise}} = 0$.

Thus, from Assumption (3), the drift term can be bounded as,
$$\begin{align}
    \| E_t^{\text{drift}} \|^{\dagger}
        &\leq \beta^t \| E_0 \|^{\dagger}
            + \sum_{k=1}^t \beta^{t-k+1} \| \nabla f(W_k) - \nabla f(W_{k-1}) \|^{\dagger} \nonumber \\
        &\leq \beta^t \| E_0 \|^{\dagger}
            + L \sum_{k=1}^t \beta^{t-k+1} \| W_k - W_{k-1} \| \label{eq:prop6-branch} \\
        &\leq \beta^t \| E_0 \|^{\dagger}
            + L \sum_{k=1}^t \beta^{t-k+1} \| \eta A_k^* \| \nonumber \\
        &\leq \beta^t \| E_0 \|^{\dagger}
            + L \eta \sum_{k=1}^t \beta^{t-k+1} \nonumber \\
        &\leq \beta^t \| E_0 \|^{\dagger}
            + \frac{\beta}{1 - \beta} L \eta \nonumber
\end{align}$$
And for the noise term, we have from Lemma 5 (viewing the double sum over time and batch as a single sum over $t \times b$ independent noise terms),
$$\begin{align}
    \mathbb{E} \left[ \| E_t^{\text{noise}} \|^{\dagger 2} \right]
        &= \mathbb{E} \left[ \left\| \sum_{k=1}^t \sum_{i=1}^b \beta^{t-k}(1 - \beta)\frac{1}{b} \xi_{k,i} \right\|^{\dagger 2} \right] \nonumber \\
        &\leq D \sigma^2 \sum_{k=1}^t \sum_{i=1}^b \left( \frac{(1 - \beta) \beta^{t-k}}{b} \right)^2 \nonumber \\
        &\leq \frac{(1 - \beta)^2}{1 - \beta^2} \frac{D \sigma^2}{b} \nonumber \\
        &= \frac{1 - \beta}{1 + \beta} \frac{D \sigma^2}{b} \nonumber \\
    \mathbb{E} \left[ \| E_t^{\text{noise}} \|^{\dagger} \right]
        &\leq \sqrt{\mathbb{E} \left[ \| E_t^{\text{noise}} \|^{\dagger 2} \right]} \nonumber \\
        &\leq \sqrt{\frac{1 - \beta}{1 + \beta}} \frac{\sqrt{D} \sigma}{\sqrt{b}} \nonumber
\end{align}$$

Thus,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|^{\dagger} \right]
        &\leq \mathbb{E} \left[ \| E_t^{\text{drift}} \|^{\dagger} \right]
            + \mathbb{E} \left[ \| E_t^{\text{noise}} \|^{\dagger} \right] \nonumber \\
        &\leq \beta^t \| E_0 \|^{\dagger}
            + \frac{\beta}{1 - \beta} L \eta
            + \sqrt{\frac{1 - \beta}{1 + \beta}}  \frac{\sqrt{D}\sigma}{\sqrt{b}} \nonumber \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|^{\dagger}\right]
        &\leq \frac{1}{1 - \beta} \frac{1}{T} \| E_0 \|^{\dagger}
            + \frac{\beta}{1 - \beta} L \eta
            + \sqrt{\frac{1 - \beta}{1 + \beta}}  \frac{\sqrt{D}\sigma}{\sqrt{b}} \nonumber \\
\end{align}$$
Substituting $E_0 = \nabla f(W_0) - M_0 = \nabla f(W_0)$ completes the proof. $\quad\blacksquare$

---

We now bound the Nesterov momentum error term.

> **Corollary 7 (Expected Nesterov momentum error bounds w/o weight decay).** Under the same assumptions as Proposition 6, arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, and any $t \geq 0$,
$$\begin{align}
    &\mathbb{E}\left[\| \nabla f(W_t) - C_t \|^{\dagger} \right] \nonumber \\
        &\qquad\leq \beta^{t+1} \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta^2}{1 - \beta} L \eta
            + \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
\end{align}$$
Moreover, averaging over $T$ iterations yields,
$$\begin{align}
    &\frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| \nabla f(W_t) - C_t \|^{\dagger} \right] \nonumber \\
        &\qquad\leq \frac{\beta}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta^2}{1 - \beta} L \eta
            + \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
\end{align}$$

**Proof.** We have,
$$\begin{align}
    \nabla f(W_t) - C_t
        &= \nabla f(W_t) - (\beta M_t + (1 - \beta) \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta (\nabla f(W_t) - M_t) + (1 - \beta) (\nabla f(W_t) - \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta E_t + (1 - \beta) \xi_{S_t} \nonumber
\end{align}$$
And since $x \mapsto \| x \|^{\dagger}$ is convex,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f(W_t) - C_t \|^{\dagger} \right]
        &\leq \beta \mathbb{E}\left[ \| E_t \|^{\dagger} \right]
            + (1 - \beta) \mathbb{E}\left[ \| \xi_{S_t} \|^{\dagger} \right] \nonumber
\end{align}$$
The result then follows from Lemma 5 and Proposition 6. $\quad\blacksquare$

---

### 2.2. Convergence bound without weight decay

> **Theorem 8 (Generalized expected stationarity for steepest descent with Nesterov momentum without weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda = 0$ (i.e., weight decay is disabled), learning rate $\eta > 0$, momentum parameter $\beta \in [0, 1)$, and initial momentum $M_0 = 0$. Then under Assumptions (1)-(4), and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have constants $X, Y, Z > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        \leq \frac{X}{T} + \frac{Y}{\sqrt{b}} + Z
\end{equation}$$
where $T$ is the total number of time steps, $b$ is the batch size, and,
$$\begin{align}
    X
        &= \frac{f(W_0) - f^*}{\eta}
            + \frac{2 \beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger} \\
    Y
        &= 2 \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \sqrt{D}\sigma \\
    Z
        &= L \eta \left( \frac{2 \beta^2}{1 - \beta}  + \frac{1}{2} \right) \nonumber
\end{align}$$

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
            + \eta \| \nabla f(W_t) - C_t \|^{\dagger}\| A_t^* \|
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \left(
                \| \nabla f(W_t) \|^{\dagger}
                - \| \nabla f(W_t) - C_t \|^{\dagger}\right)
            + \eta \| \nabla f(W_t) - C_t \|^{\dagger}
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \| \nabla f(W_t) \|^{\dagger}
            + 2 \eta \| \nabla f(W_t) - C_t \|^{\dagger}
            + \frac{(L\eta)\eta}{2} \label{eq:descentlemma-final}
\end{align}$$
Note that the $\langle \cdot, \cdot \rangle$ operator in Equation $\eqref{eq:descentlemma}$ is *not* an inner product, but the canonical pairing between cotangent and tangent spaces ($\nabla f(W_t) \in T_{W_t}^* \mathcal{W}$ while $A_t^* \in T_{W_t}\mathcal{W}$). Under the standard basis of $\mathbb{R}^{m \times n}$, however, it *behaves like* the Frobenius inner product.

Rearranging Equation $\eqref{eq:descentlemma-final}$ then gives,
$$\| \nabla f(W_t) \|^{\dagger}
    \leq \frac{f(W_t) - f(W_{t+1})}{\eta}
        + 2 \| \nabla f(W_t) - C_t \|^{\dagger}
        + \frac{L\eta}{2}$$

Taking expectations, and averaging, we have, by Corollary 7,
$$\begin{align}
    &\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \nonumber \\
        &\qquad\leq \frac{f(W_0) - f(W_T)}{\eta T}
            + 2 \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|^{\dagger}]
            + \frac{L\eta}{2} \nonumber \\
        &\qquad\leq \frac{f(W_0) - f(W_T)}{\eta T} \nonumber \\
        &\qquad\quad+ 2 \left(
            \frac{\beta}{1 - \beta}\frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{\beta^2}{1 - \beta} L \eta
            + \left(\sqrt{\frac{1 - \beta}{1 + \beta}}\beta + (1 - \beta) \right) \frac{\sqrt{D}\sigma}{\sqrt{b}} \right) \nonumber \\
        &\qquad\quad+ \frac{L\eta}{2} \nonumber \\
        &\qquad\leq \frac{X}{T} + \frac{Y}{\sqrt{b}} + Z \nonumber
\end{align}$$
where,
$$\begin{align}
    X
        &:= \frac{f(W_0) - f^*}{\eta}
            + \frac{2 \beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger} \nonumber \\
    Y
        &:= 2 \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \sqrt{D}\sigma \nonumber \\
    Z
        &:= L \eta \left( \frac{2 \beta^2}{1 - \beta}  + \frac{1}{2} \right) \nonumber
\end{align}$$
and $f^*$ is the global minimum of $f$.

## 3. Convergence bound for steepest descent under arbitrary norms with weight decay for star-convex functions

We now analyze the case where $\lambda > 0$.

### 3.1. Weight, gradient, and momentum norm bounds

> **Proposition 9 (Weight, gradient, and update bounds w/ weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda > 0$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$ and $\| W_0 \| \leq \frac{1}{\lambda}$. Then, for all $t \geq 0$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \| W_t \|
        &\leq \frac{1}{\lambda} \\
    \| W_{t+1} - W_t \|
        &\leq 2\eta \label{eq:weight-update-bound}
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

As a consequence, we also have,
$$\begin{align}
    \| W_{t+1} - W_t \|
        &= \| -\lambda \eta W_t + \eta A_t^* \| \nonumber \\
        &\leq \lambda \eta \| W_t \| + \eta \| A_t^* \| \nonumber \\
        &\leq \lambda \eta \frac{1}{\lambda} + \eta \nonumber \\
        &= 2\eta \quad\blacksquare \nonumber
\end{align}$$

---

We then derive the corresponding bounds as in Proposition 6 and Corollary 7, but now with weight decay.

> **Corollary 10 (Expected momentum error bounds w/ weight decay).** Let the momentum parameter be $\beta \in [0, 1)$, learning rate $\eta > 0$ (such that $\lambda\eta < 1$), and initial momentum $M_0 = 0$. Under Assumptions (1)-(4), arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, and any $t \geq 0$,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f(W_t) - M_t \|^{\dagger} \right]
        &\leq \beta^t \| \nabla f(W_0) \|^{\dagger}
            + \frac{2 \beta}{1 - \beta} L \eta
            + \sqrt{\frac{1 - \beta}{1 + \beta}} \frac{\sqrt{D}\sigma}{\sqrt{b}}
\end{align}$$
Moreover, averaging over $T$ iterations yields,
$$\begin{align}
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[ \| \nabla f(W_t) - M_t \|^{\dagger} \right]
        &\leq \frac{1}{1 - \beta}\frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{2 \beta}{1 - \beta} L \eta
            + \sqrt{\frac{1 - \beta}{1 + \beta}} \frac{\sqrt{D}\sigma}{\sqrt{b}}
\end{align}$$

> **Corollary 11 (Expected Nesterov momentum error bounds w/ weight decay).** Under the same assumptions as Corollary 7, for arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, and any $t \geq 0$,
$$\begin{align}
    &\mathbb{E}\left[\| \nabla f(W_t) - C_t \|^{\dagger} \right] \nonumber \\
        &\qquad\leq \beta^{t+1} \frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{2 \beta^2}{1 - \beta} L \eta
            + \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
\end{align}$$
Moreover, averaging over $T$ iterations yields,
$$\begin{align}
    &\frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| \nabla f(W_t) - C_t \|^{\dagger} \right] \nonumber \\
        &\qquad\leq \frac{\beta}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) \|^{\dagger}
            + \frac{2 \beta^2}{1 - \beta} L \eta
            + \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
\end{align}$$

**Proof.** We branch off from the proof of Proposition 6 at Equation $\eqref{eq:prop6-branch}$, but now using Equation $\eqref{eq:weight-update-bound}$ to bound $\| W_{t+1} - W_t \| \leq 2\eta$. The rest of the proof then follows identically. $\quad\blacksquare$

### 3.2. Convergence bound with weight decay

For our results below to hold, we need to assume that the objective function $f$ is star-convex at a minimzer $W_*$.
> **Assumption 12 ($f$ is star-convexity at $W_*$).** For all $W \in \mathcal{W}$ and all $\alpha \in [0, 1]$,
$$\begin{equation}
    f((1 - \alpha) W + \alpha W_*) \leq (1 - \alpha) f(W) + \alpha f(W_*)
\end{equation}$$

And to ensure that $W_*$ can indeed be reached by our steepest descent algorithm, from Proposition 9, we also set $\lambda$ to be sufficiently small such that,
$$\begin{equation}
    \| W_* \| \leq \frac{1}{\lambda}
\end{equation}$$
Now let,
$$\begin{equation}
    X_t = (1 - \lambda\eta) W_t + \lambda\eta W_* \label{eq:wd-proof-xt}
\end{equation}$$
Then we have the following useful lemmas.

> **Lemma 13.** For Nesterov momentum terms $C_t$, weights $W_t$ and $W_{t+1}$, and $X_t$ defined in Equation \eqref{eq:wd-proof-xt}, we have the following inequalities,
$$\begin{align}
    \langle C_t, W_{t+1} - X_t \rangle \leq 0 \label{eq:lemma4-ineq-1} \\
    \| W_{t} - X_t \| \leq 2\eta \\
    \| W_{t+1} - X_t \| \leq 2\eta
\end{align}$$

**Proof.** For Inequality \eqref{eq:lemma4-ineq-1}, we have,
$$\begin{align}
    \langle C_t, W_{t+1} \rangle
        &= \langle C_t, (1 - \lambda\eta) W_{t} + \eta A_t^* \rangle \nonumber \\
        &\leq \langle C_t, (1 - \lambda\eta) W_{t} + \eta A \rangle \quad \forall A : \| A \| \leq 1 \nonumber \\
        &= \langle C_t, X_t \rangle \nonumber \\
    \langle C_t, W_{t+1} - X_t \rangle
        &\leq 0 \nonumber
\end{align}$$

The other two inequalities follow from the triangle inequality and the update rule,
$$\begin{align}
    \| W_t - X_t \|
        &= \| W_t - ((1 - \lambda\eta) W_t + \lambda\eta W_*) \| \nonumber \\
        &= \lambda\eta \| W_t - W_* \| \nonumber \\
        &\leq \lambda\eta \left( \| W_t \| + \| W_* \| \right) \nonumber \\
        &\leq 2\eta \nonumber \\
    \| W_{t+1} - X_t \|
        &= \| ((1 - \lambda\eta) W_t + \eta A_t^*) - ((1 - \lambda\eta) W_t + \lambda\eta W_*) \| \nonumber \\
        &= \| \eta A_t^* - \lambda\eta W_* \| \nonumber \\
        &\leq \eta \| A_t^* \| + \lambda\eta \| W_* \| \nonumber \\
        &\leq 2\eta \qquad\blacksquare \nonumber
\end{align}$$

---

> **Theorem 14 (Expected suboptimality for steepest descent with Nesterov momentum and decoupled weight decay).** Let $\eta > 0$ be the learning rate, weight decay parameter $\lambda > 0$ (such that $\lambda\eta \leq 1$), Nesterov momentum parameter $\beta \in [0, 1)$, and initial momentum $M_0 = 0$. Then, under Assumptions (1)-(4), star-convexity of $f$ at $W_*$, and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have constants $X, Y, Z > 0$ such that,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W_*) \right]
        &\leq (1 - \lambda\eta)^T X + \frac{Y}{\sqrt{b}} + Z
\end{align}$$
where $T$ is the total number of steps, $b$ is the batch size, and,
$$\begin{align}
    X
        &= f(W_0) - f(W_*) \\
    Y
        &= \frac{2}{\lambda} \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \sqrt{D} \sigma \\
    Z
        &= \left[
            \frac{4L}{\lambda} \left(1 + \frac{\beta^2}{1 - \beta} \right)
            + \frac{2\beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger}
        \right] \eta
\end{align}$$

**Proof.** From the descent lemma, we have,

$$\begin{align}
    f(W_{t+1})
        &\leq f(W_t) + \langle \nabla f(W_t), W_{t+1} - W_t \rangle
            + \frac{L}{2} \| W_{t+1} - W_t \|^2 \nonumber \\
        &\leq f(W_t) + \left( \langle C_t, W_{t+1} - W_t \rangle + \langle \nabla f(W_t) - C_t, W_{t+1} - W_t \rangle \right)
            + \frac{L(2\eta)^2}{2} \nonumber \\
        &= f(W_t) + \left(\underbrace{\langle C_t, W_{t+1} - X \rangle}_{\leq 0} + \langle C_t, X - W_t \rangle\right) + 2L\eta^2 \nonumber \\
        &\quad+ \left(
            \langle \nabla f(W_t) - C_t, W_{t+1} - X \rangle
            + \langle \nabla f(W_t) - C_t, X - W_{t} \rangle \right) \nonumber \\
        &= f(W_t) + \langle \nabla f(W_t), X - W_t \rangle + 2L\eta^2 + \langle \nabla f(W_t) - C_t, W_{t+1} - X \rangle \nonumber \\
        &\leq \left(f(X) + \frac{L}{2} {\underbrace{\| X - W_t \|}_{\leq 2\eta}}^2 \right) + 2L\eta^2 + \| \nabla f(W_t) - C_t \|^{\dagger} \underbrace{\| W_{t+1} - X \|}_{\leq 2\eta} \label{eq:wd-proof-ineq-3} \\
        &\leq f(X) + 4L\eta^2 + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \label{eq:wd-proof-ineq-4}
\end{align}$$
where Inequality \eqref{eq:wd-proof-ineq-3} follows from the $L$-smoothness of $f$,
$$\begin{align}
    f(W_t)
        &\leq f(X)
            + \langle \nabla f(W_t), W_t - X \rangle
            + \frac{L}{2} \| W_t - X \|^2 \nonumber \\
        &\leq f(X)
            - \langle \nabla f(W_t), X - W_t \rangle
            + \frac{L}{2} \| X - W_t \|^2 \nonumber \\
    f(W_t) + \langle \nabla f(W_t), X - W_t \rangle
        &\leq f(X) + \frac{L}{2} \| X - W_t \|^2. \nonumber
\end{align}$$

Applying star-convexity of $f$ at $W_*$ on Inequality \eqref{eq:wd-proof-ineq-4} yields,
$$\begin{align}
    f(W_{t+1})
        &\leq f( (1 - \lambda\eta)W_t + \lambda\eta W_*)
            + 4L\eta^2
            + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber \\
        &\leq \left( (1 - \lambda\eta)f(W_t) + \lambda\eta f(W_*) \right)
            + 4L\eta^2
            + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber \\
    f(W_{t+1}) - f(W_*)
        &\leq (1 - \lambda\eta)(f(W_t) - f(W_*))
            + 4L\eta^2
            + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber
\end{align}$$

Taking expectations and applying Corollary 11, we have,
$$\begin{align}
    \mathbb{E}\left[ f(W_{t+1}) - f(W_*) \right]
        &\leq (1 - \lambda\eta)\mathbb{E}\left[ f(W_t) - f(W_*) \right]
            + 4L\eta^2
            + 2\eta \mathbb{E}\left[ \| \nabla f(W_t) - C_t \|^{\dagger} \right] \nonumber \\
        &\leq (1 - \lambda\eta)\mathbb{E}\left[ f(W_t) - f(W_*) \right]
            + 4L\eta^2 \nonumber \\
        &\quad+ 2\eta\left(
                \beta^{t+1} \| \nabla f(W_0) \|^{\dagger}
                + \frac{2 \beta^2}{1 - \beta} L \eta
                + \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
            \right) \nonumber \\
        &\leq (1 - \lambda\eta)\mathbb{E}\left[ f(W_t) - f(W_*) \right]
            + 4 \left(1 + \frac{\beta^2}{1 - \beta} \right) L \eta^2
            + 2\eta\beta^{t+1} \| \nabla f(W_0) \|^{\dagger} \nonumber \\
        &\quad+ 2\eta \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \nonumber
\end{align}$$

Unrolling the recurrence then yields,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W_*) \right]
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W_*)) \nonumber \\
            &\quad+ 4 \left(1 + \frac{\beta^2}{1 - \beta} \right) L \eta^2 \sum_{t=0}^{T-1} (1 - \lambda\eta)^{T-1-t} \nonumber \\
            &\quad+ 2\eta \| \nabla f(W_0) \|^{\dagger} \sum_{t=0}^{T-1} \beta^{t+1} (\underbrace{1 - \lambda\eta}_{\leq 1})^{T-1-t} \nonumber \\
            &\quad+ 2\eta \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \sum_{t=0}^{T-1} (1 - \lambda\eta)^{T-1-t} \nonumber \\
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W_*))
            + \frac{4}{\lambda} \left(1 + \frac{\beta^2}{1 - \beta} \right) L \eta
            + \frac{2\eta\beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger} \nonumber \\
        &\quad+ \frac{2}{\lambda} \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \nonumber \\
        &\leq (1 - \lambda\eta)^T X + \frac{Y}{\sqrt{b}} + Z \nonumber
\end{align}$$
where,
$$\begin{align}
    X
        &:= f(W_0) - f(W_*) \nonumber \\
    Y
        &:= \frac{2}{\lambda} \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \sqrt{D} \sigma \nonumber \\
    Z
        &:= \left[
            \frac{4 L}{\lambda} \left(1 + \frac{\beta^2}{1 - \beta} \right)
            + \frac{2 \beta}{1 - \beta} \| \nabla f(W_0) \|^{\dagger}
        \right] \eta \nonumber
\end{align}$$

---

## 4. Deriving the critical batch size

### 4.1. Critical batch size for steepest descent without weight decay

> **Theorem 15 (Critical batch size for steepest descent under arbitrary norms with Nesterov momentum without weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda = 0$. Then for an arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, the critical batch size $b_{crit}$ that minimizes the total number of tokens processed to reach $\epsilon$-convergence in terms of generalized expected stationarity is given by,
$$\begin{align}
    b_{crit}
        &= \mathcal{O}\left( (1 - \beta) \frac{D\sigma^2}{\epsilon'} \right)
\end{align}$$
where $\epsilon' := (\epsilon - Z)^2 > 0$.

**Proof.** We consider the steepest descent iteration process to have $\epsilon$-converged at time step $T$ in terms of generalized expected stationarity when, for some $\epsilon > 0$,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \leq \frac{X}{T} + \frac{Y}{\sqrt{b}} + Z \leq \epsilon \label{eq:convergence-criterion}
\end{equation}$$

Since $Z$ is a constant independent of $T$ and $b$, we can simply fold it into $\epsilon$ by defining $\epsilon' := (\epsilon - Z)^2 > 0$. Simple algebra then yields the number of iterations to satisfy the convergence criterion in Equation $\eqref{eq:convergence-criterion}$ as,
$$\begin{align}
    \frac{X}{T} + \frac{Y}{\sqrt{b}} + Z &\leq \epsilon \nonumber \\
    \frac{X}{T} + \frac{Y}{\sqrt{b}} &\leq \epsilon - Z =: \sqrt{\epsilon'} \nonumber \\
    \frac{X\sqrt{b}}{T} + Y &\leq \sqrt{\epsilon' b} \nonumber \\
    \frac{X\sqrt{b}}{\sqrt{\epsilon' b} - Y} &\leq T \nonumber \\
    \frac{X\sqrt{b}}{\sqrt{\epsilon' b} - Y} &=: T(b)
\end{align}$$
Note that we also have to constrain $b > \frac{Y^2}{\epsilon'}$ to ensure that $T(b) > 0$. Taking the first and second derivatives then yields,
$$\begin{align}
    T'(b) &= -\frac{XY}{2 \sqrt{b} (\sqrt{\epsilon' b} - Y)^2} \leq 0 \nonumber \\
    T''(b) &= \frac{XY(3\sqrt{\epsilon' b} - Y)}{4b^{3/2}(\sqrt{\epsilon' b} - Y)^3} \geq 0 \nonumber
\end{align}$$
Thus, $T(b)$ is a monotonically decreasing and convex function for $b > \frac{Y^2}{\epsilon'}$.

Now, the number of tokens we need to process to reach $\epsilon$-convergence is roughly proportional to,
$$\text{SFO}(b) := b \cdot T(b) = \frac{Xb^{3/2}}{\sqrt{\epsilon' b} - Y}$$
Taking the first and second derivatives again yields,
$$\begin{align}
    \text{SFO}'(b) &= \frac{X\sqrt{b}(2\sqrt{\epsilon' b} - 3Y)}{2(\sqrt{\epsilon' b} - Y)^2} \nonumber \\
    \text{SFO}''(b) &= \frac{XY (3Y - \sqrt{\epsilon' b})}{4\sqrt{b}(\sqrt{\epsilon' b} - Y)^3} \geq 0 \nonumber
\end{align}$$
Thus, $b \cdot T(b)$ is a convex function for $b > \frac{Y^2}{\epsilon'}$, with a minimizer $b^* = \frac{9Y^2}{4\epsilon'}$. This gives us the critical batch size,
$$\begin{align}
    b_{crit}
        &= 9 \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right)^2 \frac{D \sigma^2}{\epsilon'} \nonumber \\
        &= \mathcal{O} \left( (1 - \beta) \frac{D \sigma^2}{\epsilon'} \right) \qquad\blacksquare \nonumber
\end{align}$$

### 4.2. Critical batch size for steepest descent with decoupled weight decay

> **Theorem 16 (Critical batch size for steepest descent under arbitrary norms with Nesterov momentum and decoupled weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then for an arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, the critical batch size $b_{crit}$ that minimizes the total number of tokens processed to reach $\epsilon$-convergence in terms expected suboptimality is given by,
$$\begin{align}
    b_{crit}
        &= \mathcal{O}\left( \frac{1 - \beta}{\lambda^2} \frac{D\sigma^2}{\epsilon'} \right)
\end{align}$$
where $\epsilon' := (\epsilon - Z)^2 > 0$.

**Proof.** We consider the steepest descent iteration process to have $\epsilon$-converged at time step $T$ in terms of expected suboptimality when, for some $\epsilon > 0$,
$$\begin{equation}
    \mathbb{E}\left[ f(W_T) - f(W_*) \right]
        \leq (1 - \lambda\eta)^T X + \frac{Y}{\sqrt{b}} + Z
        \leq e^{-\lambda\eta T} X + \frac{Y}{\sqrt{b}} + Z
        \leq \epsilon \label{eq:convergence-criterion-wd}
\end{equation}$$

As before, we can fold $Z$ into $\epsilon$ by defining $\epsilon' := (\epsilon - Z)^2 > 0$. Simple algebra then yields the number of iterations to satisfy the convergence criterion in Equation $\eqref{eq:convergence-criterion-wd}$ as,
$$\begin{align}
    e^{-\lambda\eta T} X + \frac{Y}{\sqrt{b}} + Z
        &\leq \epsilon \nonumber \\
    e^{-\lambda\eta T} X + \frac{Y}{\sqrt{b}}
        &\leq \epsilon - Z =: \sqrt{\epsilon'} \nonumber \\
    \frac{1}{\lambda\eta} \ln \left( \frac{X}{\sqrt{\epsilon'} - \frac{Y}{\sqrt{b}}} \right)
        &\leq T := T(b) \nonumber
\end{align}$$
Note that we have to constrain $b > \frac{Y^2}{\epsilon'}$ to ensure that $T(b) > 0$. Now define,
$$h(x) := \frac{1}{\lambda\eta} \ln\frac{X}{x} \qquad g(b) := \sqrt{\epsilon'} - \frac{Y}{\sqrt{b}}$$
such that,
$$T(b) = h(g(b))$$

Taking first and second derivatives yields,
$$\begin{align}
    g'(b)
        &= \frac{Y b^{-3/2}}{2} > 0 \nonumber \\
    g''(b)
        &= -\frac{3 Y b^{-5/2}}{4} < 0 \nonumber \\
    h'(x)
        &= -\frac{1}{\lambda\eta x} < 0 \nonumber \\
    h''(x)
        &= \frac{1}{\lambda\eta x^2} > 0 \nonumber \\
    T'(b)
        &= \underbrace{h'(g(b))}_{< 0} \underbrace{g'(b)}_{> 0} < 0 \nonumber \\
    T''(b)
        &= \underbrace{h''(g(b))}_{> 0} \underbrace{(g'(b))^2}_{> 0} + \underbrace{h'(g(b))}_{< 0} \underbrace{g''(b)}_{< 0} > 0 \nonumber
\end{align}$$
Thus, $T(b)$ is a monotonically decreasing and convex function for $b > \frac{Y^2}{\epsilon'}$.

Now, the number of tokens we need to process to reach $\epsilon$-convergence is roughly proportional to,
$$\text{SFO}(b) := b \cdot T(b) = \frac{b}{\lambda\eta} \ln \left( \frac{X}{\sqrt{\epsilon'} - \frac{Y}{\sqrt{b}}} \right)$$
Minimizing this is equivalent to minimizing,
$$\phi(s) = s^2 \ln \left( \frac{X}{\sqrt{\epsilon'} - \frac{Y}{s}} \right)$$
Taking the first and second derivatives yields,
$$\begin{align}
    \phi'(s)
        &= 2s \ln \left( \frac{Xs}{\sqrt{\epsilon'}s - Y} \right) - \frac{Y s}{\sqrt{\epsilon'} s - Y} \nonumber \\
    \phi''(s)
        &= 2 \ln \left( \frac{Xs}{\sqrt{\epsilon'}s - Y} \right) + \frac{Y (3Y - 2\sqrt{\epsilon'}s)}{(\sqrt{\epsilon'} s - Y)^2} > 0 \nonumber
\end{align}$$
Thus, $\phi(s)$ is a convex function for $s > \frac{Y}{\sqrt{\epsilon'}}$ (and thus so is $\text{SFO}(b)$ for $b > \frac{Y^2}{\epsilon'}$). To get the minimizer, we set $\phi'(s) = 0$ and rearrange to get,
$$\begin{equation}
    2 \ln \left( \frac{Xs}{\sqrt{\epsilon'}s - Y} \right) = \frac{Y}{\sqrt{\epsilon'} s - Y} \label{eq:wd-crit-bz-deriv-eq}
\end{equation}$$
Now, let $u = \frac{Y}{\sqrt{\epsilon'}s - Y}$. Then, rearranging Equation $\eqref{eq:wd-crit-bz-deriv-eq}$ gives,
$$\begin{equation}
    u = 2 \ln\left( \frac{X}{\sqrt{\epsilon'}} (u+1) \right)
\end{equation}$$
which has a solution via the Lambert $W$ function,
$$\begin{equation}
    u^* = -2 W_{-1} \left( -\frac{\sqrt{\epsilon'}}{2X}e^{-1/2} \right) - 1 > 1
\end{equation}$$

From the definition of $u$, solving for $s$ then yields,
$$\begin{align}
    u^*
        &= \frac{Y}{\sqrt{\epsilon'}s - Y} \nonumber \\
    s_{crit}
        &= \frac{Y}{\sqrt{\epsilon'}}\left( 1 + \frac{1}{u^*} \right) \nonumber \\
    b_{crit}
        &= s_{crit}^2 = \frac{Y^2}{\epsilon'} \underbrace{\left( 1 + \frac{1}{u^*} \right)^2}_{> 1} \nonumber \\
        &= \frac{4}{\lambda^2} \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right)^2 \frac{D \sigma^2}{\epsilon'} \left( 1 + \frac{1}{u^*} \right)^2 \nonumber \\
        &= \mathcal{O}\left( \frac{1 - \beta}{\lambda^2} \frac{D\sigma^2}{\epsilon'} \right) \qquad\blacksquare \nonumber
\end{align}$$


### 4.3. Estimating D-smoothness for various optimizers

Optimizers we use in practice can be viewed as performing steepest descent under different norms [(Bernstein et al., 2024)](https://arxiv.org/abs/2409.20325). We summarize the relevant norm choices and their corresponding (empirical) local $D$-smoothness constants below.

| Optimizer     | Steepest descent norm   | Dual norm                  | $D$         |
| ------------- | ----------------------- | -------------------------- | ----------- |
| SGD           | $\| \cdot \|_F$         | $\| \cdot \|_F$            | $1$         |
| SignSGD/AdamW | $\| \cdot \|_{\infty}$  | $\| \cdot \|_{1}$          | $\approx 1$ |
| Muon/SOAP     | $\| \cdot \|_{2 \to 2}$ | $\| \cdot \|_{\text{nuc}}$ | $\approx 1$ |

See [Appendix A1](#a1-jax-code-to-estimate-d-smoothness) for the JAX code to estimate $D$-smoothness for steepest descent under various norms. We also take into account the fact that gradients in large-scale LLM training naturally have low stable rank structure. Empirically, $D \approx 1$ for SignSGD/AdamW and Muon/SOAP even for high-dimensional weight matrices, indicating that the critical batch size do not depend on the width and chosen norm. Thus, we can further reduce the critical batch size formulas to,
$$\begin{align}
    b_{crit}
        &= \mathcal{O} \left( (1 - \beta) \frac{\sigma^2}{\epsilon'} \right) \quad\text{(without weight decay)} \\
    b_{crit}
        &= \mathcal{O} \left( \frac{1 - \beta}{\lambda^2} \frac{\sigma^2}{\epsilon'} \right) \quad\text{(with decoupled weight decay)}
\end{align}$$

## 5. Learning rate scaling with batch size

In practice, it is often best to scale the learning rate $\eta$ as $\eta \propto \sqrt{b}$ when increasing the batch size $b$, regardless of the optimizer used. Here we provide a mathematical justification *why*. The crux is that increasing the batch size reduces the gradient noise variance, which in turn means that we can make larger weight updates without destabilizing training.

To see this, we first make the following assumption.

> **Assumption 17 (Local Lipschitzness of LMO).** Let $\texttt{LMO}_{\| \cdot \|}$ be the linear minimization oracle with respect to an arbitrary norm pair $\| \cdot \|$ (with dual norm $\| \cdot \|^{\dagger}$). Then there exists a constant $L_{\text{LMO}} > 0$ such that for $C_1, C_2 \in \mathcal{W}^\dagger$ denoting Nesterov momentum terms, we have,
$$\begin{equation}
    \| \texttt{LMO}_{\| \cdot \|}(C_1) - \texttt{LMO}_{\| \cdot \|}(C_2) \| \leq L_{\text{LMO}} \| C_1 - C_2 \|^{\dagger}
\end{equation}$$

Then, we have the following result.

> **Proposition 18 (Weight update noise variance is proportional to $\eta^2/b$).** Let $\eta > 0$ be the learning rate and $b \geq 1$ be the batch size. Under Assumptions 1-4 and Assumption (17) and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have,
$$\begin{equation}
    \mathbb{E} \left[ \| \Delta W_t^{\text{noise}} \|^2 \right] \propto \frac{\eta^2}{b}
\end{equation}$$

**Proof.** We can decompose our weight update rule in Equation $\eqref{eq:updateweightdecay}$ into deterministic and stochastic components as follows,
$$\begin{equation}
    \nabla W_t = W_{t+1} - W_t =  \underbrace{-\lambda\eta W_t + \eta A_t^{\text{det}}}_{\Delta W_t^{\text{det}}} + \underbrace{\eta A_t^{\text{noise}}}_{\Delta W_t^{\text{noise}}}
\end{equation}$$
where $A_t^* = A_t^{\text{det}} + A_t^{\text{noise}}$ is the decomposition of the steepest descent direction into its deterministic and stochastic components.

Taking norms and expectations, and using Corollary 11 then yields,
$$\begin{align}
    \mathbb{E} \left[ \| \Delta W_t^{\text{noise}} \|^2 \right]
        &= \eta^2 \mathbb{E} \left[ \| A_t^{\text{noise}} \|^2 \right] \nonumber \\
        &= \eta^2 \mathbb{E} \left[ \| A_t^* - A_t^{\text{det}} \|^2 \right] \nonumber \\
        &\lesssim \eta^2 L_{\text{LMO}}^2 \mathbb{E} \left[ \| C_t - \nabla f(W_t) \|^{\dagger 2} \right] \nonumber \\
        &\lesssim \eta^2 L_{\text{LMO}}^2 \frac{(3 \beta + 1) (1 - \beta)}{1 + \beta} \frac{D \sigma^2}{b} + O\left(\frac{1}{T} + 1 \right) \nonumber \\
        &\propto \frac{\eta^2}{b} \quad\blacksquare \nonumber
\end{align}$$

Now, if we already know that training is fast and stable for some gradient noise variance level $\mathbb{E} \left[ \| \Delta W_t^{\text{noise}} \|^2 \right]$, then it is natural to preserve it as we scale the batch size $b$. Thus, we have,
$$\begin{align}
    \frac{\eta_{\text{new}}^2}{b_{\text{new}}}
        &= \frac{\eta_{\text{old}}^2}{b_{\text{old}}} = \text{constant} \nonumber \\
    \eta_{\text{new}}
        &= \eta_{\text{old}}\sqrt{\frac{b_{\text{new}}}{b_{\text{old}}}}. \label{eq:lr-bz-scaling}
\end{align}$$
This means that, e.g., if we $4\times$ the batch size, then increasing the learning rate by a factor of $2$ preserves training stability. This is consistent with prior work ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162); [Malladi et al., 2024](https://arxiv.org/abs/2205.10287); [Ryu (2025)](https://x.com/cloneofsimo/status/1907731069878825400)).

## 6. Experiments

### 6.1. AdamW and Muon have the same critical batch size

![](crit-bz-muon-vs-adamw.jpg#center)

Here we train a 130M parameter Llama-based Transformer model using both AdamW and Muon optimizers for 1 Chinchilla. We sweep over batch sizes from $2^{18}$ to $2^{22}$ tokens, and for each batch size, we scale the learning rate $\eta$ as $\eta = \eta_0 \sqrt{b / b_0}$ (Equation $\eqref{eq:lr-bz-scaling}$), where $b_0 \approx 2^{19}$ and $\eta_0$ is the optimal learning rate found for $b_0$ for each optimizer ([Wen et al., 2025](https://arxiv.org/abs/2509.02046v1)). We then plot the validation loss against the batch size in the figure above.

We see that both AdamW and Muon reach the same loss for batch sizes up to $2^{19}$ tokens, after which both optimizers start to degrade in performance. This provides empirical evidence that AdamW and Muon have the same critical batch size, consistent with our theoretical results. Interestingly, we also see that Muon is more stable at larger batch sizes, which is consistent with prior work ([Essential AI Team, 2025](https://arxiv.org/abs/2505.02222); [Ahn et al., 2025](https://arxiv.org/abs/2504.05295); [Pethick et al., 2025](https://arxiv.org/abs/2502.07529)). This will be an interesting direction for future work.

### 6.2. Square Root Learning Rate Scaling is Effective

![](lr-bz-scaling.png#center)

Here we show that the square root learning rate scaling rule as in Equation $\eqref{eq:lr-bz-scaling}$ is effective for both AdamW and Muon optimizers. We train a 130M parameter Llama-based Transformer model using both optimizers for 8 Chinchilla, sweeping over learning rates and batch sizes. We then plot the validation loss against the learning rate & batch size in the figure above. Notice that the optimal $(\eta, \sqrt{b})$ pair remains roughly constant for both optimizers, confirming the effectiveness of the square root learning rate scaling rule.

## 7. Discussion

The main result of this work is that the *shape* of the convergence bounds in terms of generalized expected stationarity and expected suboptimality:
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &= \frac{X(\eta, \beta)}{T} + \frac{Y(\beta, D)}{\sqrt{b}} + Z(\eta, \beta) \\
    \mathbb{E}[f(W_t) - f(W_*)]
        &= (1 - \lambda\eta)^T X + \frac{Y(\beta, \lambda, D)}{\sqrt{b}} + Z(\eta, \beta, \lambda)
\end{align}$$
are universal across all norms used for steepest descent. In fact, for preconditioned steepest descent or steepest descent under norms induced by inner products, $D = 1$ and thus the bounds are exactly the same. And for non-inner-product norms like $\| \cdot \|_{\infty}$ and $\| \cdot \|_{2 \to 2}$, $D \approx 1$ empirically makes the bounds approximately the same as well.

As a consequence, the critical batch size formulas for with and without weight decay:
$$\begin{align}
    b_{crit} &= \mathcal{O}\left( (1 - \beta) \frac{\sigma^2}{\epsilon'} \right) \quad \text{w/o weight decay} \\
    b_{crit} &= \mathcal{O}\left( \frac{1 - \beta}{\lambda^2} \frac{\sigma^2}{\epsilon'} \right) \quad \text{w/ decoupled weight decay}
\end{align}$$
also hold universally across all norms. We have also provided empirical evidence that AdamW and Muon have the same critical batch size in practice, consistent with our theoretical results. This matches prior results by [Sato et al. (2025)](https://arxiv.org/abs/2507.01598) that the critical batch size formula transfers between AdamW and Muon, but now we have shown that it potentially transfers to *all* first-order optimizers that can be interpreted as performing steepest descent under some norm.

Also notice that $b_{crit} \to 0$ as $\beta \to 1$, which is expected since high momentum increases the effective batch size (or the "lifetime" of gradient estimates). Lastly, there is also a "phase transition" when adding weight decay. And with weight decay, the critical batch size scales with the square of the 'effective constraint radius' ($\frac{1}{\lambda}$) of the weights.

## Acknowledgements

Big thanks to the [Marin Community](https://marin.community/) and especially Kaiyue Wen for helping me run experiments for this work. Also big thanks to Antonio Silveti-Falls and Volkan Cevher for providing helpful feedback on an earlier draft of this work. All remaining errors are my own.

## How to cite

```bibtex
@misc{cesista2025sdcbs,
  author = {Franz Louis Cesista, Kaiyue Wen},
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
4. Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team. An Empirical Model of Large-Batch Training. URL https://arxiv.org/abs/1812.06162
5. Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, Sanjeev Arora (2024). On the SDEs and Scaling Rules for Adaptive Gradient Algorithms. URL https://arxiv.org/abs/2205.10287
6. Simo Ryu (2025). Empirical observation that AdamW, Shampoo, and Muon follow the lr ~ sqrt(batch size) scaling rule on X/Twitter. URL https://x.com/cloneofsimo/status/1907731069878825400
7. Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang (2025). Fantastic Pretraining Optimizers and Where to Find Them. URL https://arxiv.org/abs/2509.02046v1
8. Essential AI: Ishaan Shah, Anthony M. Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, Khoi Nguyen, Kurt Smith, Michael Callahan, Michael Pust, Mohit Parmar, Peter Rushton, Platon Mazarakis, Ritvik Kapila, Saurabh Srivastava, Somanshu Singla, Tim Romanski, Yash Vanjani, Ashish Vaswani (2025). Practical Efficiency of Muon for Pretraining. URL https://arxiv.org/abs/2505.02222
9. Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, John Langford (2025). Dion: Distributed Orthonormalized Updates. URL https://arxiv.org/abs/2504.05295
10. Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher (2025). Training Deep Learning Models with Norm-Constrained LMOs. URL https://arxiv.org/abs/2502.07529

## Appendix

### A.1. JAX code to estimate D-smoothness

```python
import jax
import jax.numpy as jnp

def lipschitz_estimate(grad_g, norm_fn, dual_norm_fn, generate_low_rank_matrix, key, shape, n_pairs=10000, radius=1.0):
    def one_ratio(key):
        k1, k2 = jax.random.split(key)
        G1 = generate_low_rank_matrix(k1, shape, radius)
        G2 = generate_low_rank_matrix(k2, shape, radius)
        return norm_fn(grad_g(G1) - grad_g(G2)) / dual_norm_fn(G1 - G2)

    keys = jax.random.split(key, n_pairs)
    ratios = jax.vmap(one_ratio)(keys)
    return jnp.max(ratios), jnp.mean(ratios)


def f_inf_norm(W):
    return jnp.linalg.norm(W, ord=jnp.inf)

def f1_norm(W):
    return jnp.linalg.norm(W, ord=1)


def spectral_norm(W):
    return jnp.linalg.matrix_norm(W, ord=2)

def nuclear_norm(W):
    return jnp.linalg.norm(W, ord='nuc')


def frobenius_norm(W):
    return jnp.linalg.norm(W, ord='fro')

def frobenius_lmo(W, radius=1.):
    return radius * W / frobenius_norm(W)


def g(W, norm_fn):
    return 0.5 * norm_fn(W)**2

grad_g_f1 = jax.grad(lambda W: g(W, f1_norm))
grad_g_nuclear = jax.grad(lambda W: g(W, nuclear_norm))
grad_g_fro = jax.grad(lambda W: g(W, frobenius_norm))


def generate_low_rank_matrix(key, shape, r=12):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (shape[0], r))
    B = jax.random.normal(k2, (r, shape[1]))
    return A @ B

def f1_generate_low_rank_matrix(key, shape, radius=1.0, r=12):
    G = generate_low_rank_matrix(key, shape, r=r)
    return G / f1_norm(G) * radius

def nuclear_generate_low_rank_matrix(key, shape, radius=1.0, r=12):
    G = generate_low_rank_matrix(key, shape, r=r)
    return G / nuclear_norm(G) * radius

def frobenius_generate_low_rank_matrix(key, shape, radius=1.0, r=12):
    G = generate_low_rank_matrix(key, shape, r=r)
    return G / frobenius_norm(G) * radius


key = jax.random.PRNGKey(0)
m, n = 768, 768
shape = (m, n)
n_pairs = 100
radius = 1.
print("(L_2, L2)", float(lipschitz_estimate(grad_g_fro, frobenius_norm, frobenius_norm, frobenius_generate_low_rank_matrix, key, shape, n_pairs=n_pairs, radius=radius)[0]))
print("(L_inf, L_1)", float(lipschitz_estimate(grad_g_f1, f_inf_norm, f1_norm, f1_generate_low_rank_matrix, key, shape, n_pairs=n_pairs, radius=radius)[0]))
print("(spec, nuc)", float(lipschitz_estimate(grad_g_nuclear, spectral_norm, nuclear_norm, nuclear_generate_low_rank_matrix, key, shape, n_pairs=n_pairs, radius=radius)[0]))
```
