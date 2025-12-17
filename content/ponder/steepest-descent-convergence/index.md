---
title: "Convergence Bounds for Steepest Descent Under Arbitrary Norms"
date: 2025-12-11
tags: ["Machine Learning", "Optimizers"]
author: ["Franz Louis Cesista"]
description: "First-order optimization under arbitrary norms with Nesterov momentum (and decoupled weight decay) yields a universal convergence bound. Our results generalize to norms not induced by inner products, and also considers batch size."
summary: "First-order optimization under arbitrary norms with Nesterov momentum (and decoupled weight decay) yields a universal convergence bound. Our results generalize to norms not induced by inner products, and also considers batch size."
---

## 1. Introduction

This work improves on [Kovalev's (2025)](https://arxiv.org/abs/2503.12645) prior work on convergence bounds for (stochastic) steepest descent under arbitrary norms by:
1. Incorporating Nesterov momentum,
2. Incorporating *decoupled* weight decay,
3. Incorporating batch size,
4. Computing gradient noise variance directly using the dual norm (instead of using Euclidean norm as a proxy), and
5. Eliminating assumptions (e.g., $\eta \geq \lambda \{ \| W_0 \|, \| W^* \| \}$).

## 2. Convergence bound for steepest descent under arbitrary norms with Nesterov momentum without weight decay

From Theorem 8 in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), we have the following bound on the average expected gradient norm when using steepest descent under an arbitrary norm $\| \cdot \|$ with Nesterov momentum without weight decay.

> **Theorem 1.** Let $W_t$ be the weight at time step $t$, learning rate $\eta > 0$, momentum parameter $\beta \in [0, 1)$, and initial momentum $M_0 = 0$. Then under Assumptions (1)-(4) in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &\leq \frac{1}{T} \left( \frac{\Delta_0}{\eta}
            + \frac{4 \beta}{1 - \beta} G_0 \right) \nonumber \\
        &\quad+ 2 \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D}\sigma}{\sqrt{b}} \nonumber \\
        &\quad+ \left( \frac{4 \beta^2}{1 - \beta} + \frac{1}{2} \right) L \eta \label{eq:theorem1-bound}
\end{align}$$
where $\Delta_0 = f(W_0) - f^*$, $G_0 = \| \nabla f(W_0) \|^{\dagger}$, and $\rho$ is some upper bound on the anti-alignment of nesterov momentum terms $C_t$ and the weights $W_t$.

For a given stationary tolerance $\epsilon > 0$, we want to determine bounds on the step size $\eta$, momentum parameter $\beta$, and total number of time steps $T$ such that,
$$\begin{equation}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \leq \epsilon
\end{equation}$$

Now, note that as $T \to \infty$, only the (first) term involving $\frac{1}{T}$ vanishes. The other two terms involve $\beta$ and $\eta$, which we can tune accordingly.

> **Corollary 2.** For some generalized stationary tolerance $\epsilon > 0$, to ensure that, $\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \leq \epsilon$, via steepest descent under arbitrary norms with Nesterov momentum without weight decay, it suffices to set,
$$\begin{align}
    \theta &= 1 - \beta = \mathcal{O}\left(\min{\left\{1, \frac{b\epsilon^2}{D\sigma^2}\right\}}\right) \\
    \eta &= \mathcal{O}\left(\min{\left\{\frac{\epsilon}{L}, \frac{b\epsilon^3}{D\sigma^2L}\right\}}\right) \\
    T &= \Omega\left(\max{\left\{
        \frac{G_0}{\epsilon},
        \frac{L \Delta_0}{\epsilon^2},
        \frac{D \sigma^2 G_0}{b \epsilon^3},
        \frac{D \sigma^2 L \Delta_0}{b \epsilon^4}
    \right\}}\right)
\end{align}$$

**Proof.** Reparametrizing $\theta = 1 - \beta$ and using $\beta < 1$, we can simplify the bound in Theorem 1 as follows,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &\lesssim \frac{1}{T} \left( \frac{\Delta_0}{\eta}
            + \frac{2}{\theta} G_0 \right)
            + 2\sqrt{2} \sqrt{\theta} \frac{\sqrt{D}\sigma}{\sqrt{b}}
            + \left( \frac{4}{\theta} + \frac{1}{2} \right) L \eta
        \leq \epsilon
\end{align}$$

It then suffices to force each term to be $\mathcal{O}(\epsilon)$.

### 2.1. Bounding θ

For some small constant $c_1 \leq 1/5$, we want to bound the term solely involving $\theta$ by $\epsilon$ as follows,
$$\begin{align}
    2\sqrt{2}\sqrt{\theta} \frac{\sqrt{D}\sigma}{\sqrt{b}} \leq c_1 \epsilon
        &\implies \theta \leq \frac{c_1^2}{8} \frac{b\epsilon^2}{D\sigma^2}, \nonumber
\end{align}$$
so we set,
$$\begin{equation}
    \theta = \mathcal{O}\left(\min{\left\{1, \frac{b\epsilon^2}{D\sigma^2}\right\}}\right) \label{eq:theta-bound}
\end{equation}$$

### 2.2. Bounding η

For small constants $c_2, c_3 \leq 1/5$, we then bound the terms involving $\eta$ by $\epsilon$ as follows,
$$\begin{align}
    \frac{4}{\theta} L \eta \leq c_2 \epsilon
        &\implies \eta \leq \frac{c_2}{4} \frac{\theta \epsilon}{L} \nonumber \\
    \frac{1}{2} L \eta \leq c_3 \epsilon
        &\implies \eta \leq 2 c_3 \frac{\epsilon}{L} \nonumber
\end{align}$$
For $0 < \theta < 1$, the first condition is the most restrictive, so we set,
$$\eta = \mathcal{O}\left( \frac{\theta \epsilon}{L} \right)$$
Substituting the bound on $\theta$ from Equation \eqref{eq:theta-bound}, we have,
$$\begin{equation}
    \eta = \mathcal{O}\left(\min{\left\{\frac{\epsilon}{L}, \frac{b\epsilon^3}{D\sigma^2L}\right\}}\right) \label{eq:eta-bound}
\end{equation}$$

### 2.3. Bounding T

For small constants $c_4, c_5 < 1/5$, we then bound the terms involving $T$ by $\epsilon$ as follows,
$$\begin{align}
    \frac{1}{T} \frac{\Delta_0}{\eta} \leq c_4 \epsilon
        &\implies T \geq \frac{1}{c_4} \frac{\Delta_0}{\eta \epsilon} \label{eq:T-bound-1} \\
    \frac{1}{T} \frac{2}{\theta} G_0 \leq c_5 \epsilon
        &\implies T \geq \frac{2}{c_5} \frac{G_0}{\theta \epsilon} \label{eq:T-bound-2}
\end{align}$$

Substituting the bound on $\eta$ from Equation \eqref{eq:eta-bound} into Equation \eqref{eq:T-bound-1}, we have,
$$\begin{equation}
    T = \Omega\left(\max{\left\{
        \frac{L \Delta_0}{\epsilon^2},
        \frac{D \sigma^2 L \Delta_0}{b \epsilon^4}
    \right\}}\right) \label{eq:T-bound-1-final}
\end{equation}$$

Likewise, substituting the bound on $\theta$ from Equation \eqref{eq:theta-bound} into Equation \eqref{eq:T-bound-2}, we have,
$$\begin{align}
    T &= \Omega\left(\max{\left\{
        \frac{G_0}{\epsilon},
        \frac{D \sigma^2 G_0}{b \epsilon^3}
    \right\}}\right) \label{eq:T-bound-2-final}
\end{align}$$

Thus, combining Equations \eqref{eq:T-bound-1-final} and \eqref{eq:T-bound-2-final}, we have,
$$\begin{align}
    T &= \Omega\left(\max{\left\{
        \frac{G_0}{\epsilon},
        \frac{L \Delta_0}{\epsilon^2},
        \frac{D \sigma^2 G_0}{b \epsilon^3},
        \frac{D \sigma^2 L \Delta_0}{b \epsilon^4}
    \right\}}\right) \qquad\blacksquare \label{eq:T-bound-final}
\end{align}$$

## 3. Convergence bound for steepest descent under arbitrary norms with Nesterov momentum with decoupled weight decay for star-convex functions

For our results below to hold, we need to assume that the objective function $f$ is star-convex. For that, we need to pick a minimizer $W^*$ of $f$ within the subspace of "reachable" weights in $\mathcal{W}$ when using decoupled weight decay. From Proposition 9 in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), we can pick,
$$\begin{equation}
    W^* := \arg\min_{W \in \mathcal{W}} f(W) \quad \text{ such that } \quad \| W \| \leq \frac{1}{\lambda}
\end{equation}$$

> **Assumption 3 ($f$ is star-convexity at $W^*$).** For all $W \in \mathcal{W}$ and all $\alpha \in [0, 1]$,
$$\begin{equation}
    f((1 - \alpha) W + \alpha W^*) \leq (1 - \alpha) f(W) + \alpha f(W^*)
\end{equation}$$

Now let,
$$\begin{equation}
    X_t = (1 - \lambda\eta) W_t + \lambda\eta W^* \label{eq:wd-proof-xt}
\end{equation}$$
Then we have the following useful lemmas.

> **Lemma 4.** For Nesterov momentum terms $C_t$, weights $W_t$ and $W_{t+1}$, and $X_t$ defined in Equation \eqref{eq:wd-proof-xt}, we have the following inequalities,
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
        &= \| W_t - ((1 - \lambda\eta) W_t + \lambda\eta W^*) \| \nonumber \\
        &= \lambda\eta \| W_t - W^* \| \nonumber \\
        &\leq \lambda\eta \left( \| W_t \| + \| W^* \| \right) \nonumber \\
        &\leq 2\eta \nonumber \\
    \| W_{t+1} - X_t \|
        &= \| ((1 - \lambda\eta) W_t + \eta A_t^*) - ((1 - \lambda\eta) W_t + \lambda\eta W^*) \| \nonumber \\
        &= \| \eta A_t^* - \lambda\eta W^* \| \nonumber \\
        &\leq \eta \| A_t^* \| + \lambda\eta \| W^* \| \nonumber \\
        &\leq 2\eta \qquad\blacksquare \nonumber
\end{align}$$

---

> **Theorem 5.** Let $\eta > 0$ be the learning rate, weight decay parameter $\lambda > 0$ (such that $\lambda\eta \leq 1$), Nesterov momentum parameter $\beta \in [0, 1)$, and initial momentum $M_0 = 0$. Then, under Assumptions (1)-(4) in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), star-convexity of $f$ at $W^*$, and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W^*) \right]
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W^*))
            + \frac{4}{\lambda} \left(1 + \frac{2 \beta^2}{1 - \beta} \right) L \eta \nonumber \\
        &\quad+ \frac{4\eta\beta}{1 - \beta} \| G_0 \|^{\dagger}
            + \frac{2}{\lambda} \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
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

Applying star-convexity of $f$ at $W^*$ on Inequality \eqref{eq:wd-proof-ineq-4} yields,
$$\begin{align}
    f(W_{t+1})
        &\leq f( (1 - \lambda\eta)W_t + \lambda\eta W^*)
            + 4L\eta^2
            + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber \\
        &\leq \left( (1 - \lambda\eta)f(W_t) + \lambda\eta f(W^*) \right)
            + 4L\eta^2
            + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber \\
    f(W_{t+1}) - f(W^*)
        &\leq (1 - \lambda\eta)(f(W_t) - f(W^*))
            + 4L\eta^2
            + 2\eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber
\end{align}$$

Taking expectations and applying Corollary 11 from [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), we have,
$$\begin{align}
    \mathbb{E}\left[ f(W_{t+1}) - f(W^*) \right]
        &\leq (1 - \lambda\eta)\mathbb{E}\left[ f(W_t) - f(W^*) \right]
            + 4L\eta^2
            + 2\eta \mathbb{E}\left[ \| \nabla f(W_t) - C_t \|^{\dagger} \right] \nonumber \\
        &\leq (1 - \lambda\eta)\mathbb{E}\left[ f(W_t) - f(W^*) \right]
            + 4L\eta^2 \nonumber \\
        &\quad+ 2\eta\left(
                2\beta^{t+1} \| G_0 \|^{\dagger}
                + \frac{4 \beta^2}{1 - \beta} L \eta
                + \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}}
            \right) \nonumber \\
        &\leq (1 - \lambda\eta)\mathbb{E}\left[ f(W_t) - f(W^*) \right]
            + 4 \left(1 + \frac{2 \beta^2}{1 - \beta} \right) L \eta^2
            + 4\eta\beta^{t+1} \| G_0 \|^{\dagger} \nonumber \\
        &\quad+ 2\eta \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \nonumber
\end{align}$$

Unrolling the recurrence then yields,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W^*) \right]
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W^*)) \nonumber \\
            &\quad+ 4 \left(1 + \frac{2 \beta^2}{1 - \beta} \right) L \eta^2 \sum_{t=0}^{T-1} (1 - \lambda\eta)^{t} \nonumber \\
            &\quad+ 4\eta \| G_0 \|^{\dagger} \sum_{t=0}^{T-1} \beta^{t+1} (\underbrace{1 - \lambda\eta}_{\leq 1})^{T-1-t} \nonumber \\
            &\quad+ 2\eta \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \sum_{t=0}^{T-1} (1 - \lambda\eta)^{t} \nonumber \\
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W^*))
            + \frac{4}{\lambda} \left(1 + \frac{2 \beta^2}{1 - \beta} \right) L \eta
            + \frac{4\eta\beta}{1 - \beta} \| G_0 \|^{\dagger} \nonumber \\
        &\quad+ \frac{2}{\lambda} \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \qquad\blacksquare \nonumber
\end{align}$$

We can then use this theorem to derive convergence bounds as follows.

> **Corollary 6.** For some expected suboptimality tolerance $\epsilon > 0$, to ensure that, $\mathbb{E}\left[ f(W_T) - f(W^*) \right] \leq \epsilon$, via steepest descent under arbitrary norms with Nesterov momentum *with* decoupled weight decay, it suffices to set,
$$\begin{align}
    \theta &= 1 - \beta = \mathcal{O}\left(\min{\left\{1, \frac{\lambda^2 b \epsilon^2}{D \sigma^2}\right\}}\right) \\
    \eta &= \mathcal{O}\left(\min{\left\{\frac{\lambda\epsilon}{L}, \frac{\epsilon}{\| G_0 \|^{\dagger}},
        \frac{\lambda^3 b \epsilon^3}{D \sigma^2 L},
        \frac{\lambda^2 b \epsilon^3}{D \sigma^2 \| G_0 \|^{\dagger}}
    \right\}}\right) \\
    T &= \Omega\left(\max{\left\{
        \frac{L}{\lambda^2 \epsilon},
        \frac{\| G_0 \|^{\dagger}}{\lambda \epsilon},
        \frac{D \sigma^2 L}{\lambda^3 b \epsilon^3},
        \frac{D \sigma^2 \| G_0 \|^{\dagger}}{\lambda^2 b \epsilon^3}
    \right\}}\right)
\end{align}$$

**Proof.** As in the previous section, reparametrizing $\theta = 1 - \beta$ and using $\beta < 1$, we can simplify the bound in Theorem 5 as follows,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W^*) \right]
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W^*))
            + \frac{12}{\lambda\theta} L \eta
            + \frac{4\eta}{\theta} \| G_0 \|^{\dagger}
            + \frac{2\sqrt{2}}{\lambda} \sqrt{\theta} \frac{\sqrt{D} \sigma}{\sqrt{b}}
\end{align}$$

### 3.1. Bounding θ

For some small constant $c_1 \leq 1/4$, we want to bound the term involving $\theta$ by $\epsilon$ as follows,
$$\begin{align}
    \frac{2\sqrt{2}}{\lambda} \sqrt{\theta} \frac{\sqrt{D} \sigma}{\sqrt{b}} \leq c_1 \epsilon
        &\implies \theta \leq \frac{c_1^2}{8} \frac{\lambda^2 b \epsilon^2}{D \sigma^2}, \nonumber
\end{align}$$
so we set,
$$\begin{equation}
    \theta = \mathcal{O}\left(\min{\left\{1, \frac{\lambda^2 b \epsilon^2}{D \sigma^2}\right\}}\right) \label{eq:theta-bound-wd}
\end{equation}$$

### 3.2. Bounding η

For small constants $c_2, c_3 \leq 1/4$, we then bound the terms involving $\eta$ by $\epsilon$ as follows,
$$\begin{align}
    \frac{12}{\lambda\theta} L \eta \leq c_2 \epsilon
        &\implies \eta \leq \frac{c_2}{12} \frac{\lambda\theta\epsilon}{L} \nonumber \\
    \frac{4\eta}{\theta} \| G_0 \|^{\dagger} \leq c_3 \epsilon
        &\implies \eta \leq \frac{c_3}{4} \frac{\theta\epsilon}{\| G_0 \|^{\dagger}} \nonumber
\end{align}$$
so we set,
$$\begin{equation}
    \eta = \mathcal{O}\left( \min{\left\{ \frac{\lambda\theta\epsilon}{L}, \frac{\theta\epsilon}{\| G_0 \|^{\dagger}} \right\}} \right) \nonumber
\end{equation}$$
Substituting the bound on $\theta$ from above, we have,
$$\begin{equation}
    \eta = \mathcal{O}\left(\min{\left\{\frac{\lambda\epsilon}{L}, \frac{\epsilon}{\| G_0 \|^{\dagger}},
        \frac{\lambda^3 b \epsilon^3}{D \sigma^2 L},
        \frac{\lambda^2 b \epsilon^3}{D \sigma^2 \| G_0 \|^{\dagger}}
    \right\}}\right) \label{eq:eta-bound-wd}
\end{equation}$$

### 3.3. Bounding T

For some small constant $c_4 < 1/4$, we then bound the term involving $T$ by $\epsilon$ as follows,

$$\begin{align}
    (1 - \lambda\eta)^T \leq e^{- \lambda\eta T} \leq c_4 \epsilon
        &\implies T \geq \frac{1}{\lambda\eta} \log{\left(\frac{1}{c_4 \epsilon}\right)} \nonumber
\end{align}$$

Substituting the bound on $\eta$ from above, we have,
$$\begin{align}
    T &= \Omega\left(\max{\left\{
        \frac{L}{\lambda^2 \epsilon},
        \frac{\| G_0 \|^{\dagger}}{\lambda \epsilon},
        \frac{D \sigma^2 L}{\lambda^3 b \epsilon^3},
        \frac{D \sigma^2 \| G_0 \|^{\dagger}}{\lambda^2 b \epsilon^3}
    \right\}}\right) \label{eq:T-bound-final-wd}
\end{align}$$

## 4. Discussion

Here we have proven that steepest descent under arbitrary norms with Nesterov momentum with or without decoupled weight decay converges, with a universal convergence bound that holds for any norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$. We have also derived (universal) iteration complexity bounds for both cases, in terms of generalized expected stationarity and expected suboptimality, respectively.

For generalized expected stationarity, in the case without weight decay, the iteration complexity in Equation \eqref{eq:T-bound-final} is proportional to $1/\epsilon^4$ in the worst case, which is consistent with prior state-of-the-art results ([Ghadimi and Lan, 2013](https://doi.org/10.1137/120880811); [Cutkosky and Mehta, 2020](https://proceedings.mlr.press/v119/cutkosky20b.html); [Sun et al., 2023](https://proceedings.mlr.press/v202/sun23l.html); [Kovalev, 2025](https://arxiv.org/abs/2503.12645)) and cannot be improved further without additional assumptions ([Arjevani et al., 2022](https://doi.org/10.1007/s10107-022-01822-7)). For the expected suboptimality, in the case with decoupled weight decay, the iteration complexity is proportional to $1/\epsilon^3$ in the worst case, which matches [Kovalev's (2025)](https://arxiv.org/abs/2503.12645) prior result.

Interestingly, from the bounds in Equations \eqref{eq:eta-bound} and \eqref{eq:eta-bound-wd}, there seems to be a batch size threshold $b^*$ such that, up to which, increasing the batch size allows us to increase the learning rate, thereby reducing the number of iterations required to reach the desired stationary tolerance $\epsilon$. But beyond $b^*$, increasing the batch size no longer helps reduce the iteration complexity, as the bounds on $\eta$ and $\theta$ become independent of $b$, and $T$ starts to scale as $\Omega(1/\epsilon^2)$ (without weight decay) or $\Omega(1/\epsilon)$ (with decoupled weight decay).

## How to cite

```bibtex
@misc{cesista2025sdconvergence,
  author = {Franz Louis Cesista},
  title = {Convergence Bounds for Steepest Descent Under Arbitrary Norms},
  year = {2025},
  month = {December},
  day = {11},
  url = {https://leloykun.github.io/ponder/steepest-descent-convergence/},
}
```

## References

1. Dmitry Kovalev (2025). Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization. URL https://arxiv.org/abs/2503.12645
2. Saeed Ghadimi, Guanghui Lan (2013). Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming. URL https://doi.org/10.1137/120880811
3. Ashok Cutkosky, Harsh Mehta (2020). Momentum Improves Normalized SGD. URL https://proceedings.mlr.press/v119/cutkosky20b.html
4. Tao Sun, Qingsong Wang, Dongsheng Li, Bao Wang (2023). Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions. URL https://proceedings.mlr.press/v202/sun23l.html
5. Yossi Arjevani, Yair Carmon, John C. Duchi, Dylan J. Foster, Nathan Srebro, Blake Woodworth. Lower bounds for non-convex stochastic optimization. Math. Program. 199, 165–214 (2023). https://doi.org/10.1007/s10107-022-01822-7
