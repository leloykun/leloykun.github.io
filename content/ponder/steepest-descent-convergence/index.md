---
title: "Convergence Bounds for Steepest Descent Under Arbitrary Norms"
date: 2025-12-11
tags: ["Machine Learning", "Optimizers"]
author: ["Franz Louis Cesista"]
description: "First-order optimization under arbitrary norms with Nesterov momentum (and decoupled weight decay) yields a universal convergence bound. Our results generalize to norms not induced by inner products, and also considers batch size."
summary: "First-order optimization under arbitrary norms with Nesterov momentum (and decoupled weight decay) yields a universal convergence bound. Our results generalize to norms not induced by inner products, and also considers batch size."
editPost:
    URL: "https://github.com/leloykun/steepest-descent-lean"
    Text: "Lean Formalization"
---

## 1. Introduction

This work improves on [Kovalev's (2025)](https://arxiv.org/abs/2503.12645) prior work on convergence bounds for (stochastic) steepest descent under arbitrary norms by:
1. Incorporating Nesterov momentum,
2. Incorporating *decoupled* weight decay,
3. Incorporating batch size,
4. Computing gradient noise variance directly using the dual norm (instead of using Euclidean norm as a proxy), and
5. Eliminating assumptions (e.g., $\eta \geq \lambda \{ \| W_0 \|, \| W_* \| \}$).

## 2. Convergence bound for steepest descent under arbitrary norms with Nesterov momentum without weight decay

From Theorem 8 in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), we have the following bound on the average expected gradient norm when using steepest descent under an arbitrary norm $\| \cdot \|$ with Nesterov momentum without weight decay.

> **Theorem 1 (Generalized expected stationarity for steepest descent with Nesterov momentum without weight decay).** Let $W_t$ be the weight at time step $t$, learning rate $\eta > 0$, momentum parameter $\beta \in [0, 1)$, and initial momentum $M_0 = 0$. Then under Assumptions (1)-(4) in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &\leq \frac{1}{T} \left( \frac{\Delta_0}{\eta}
            + \frac{2 \beta}{1 - \beta} G_0 \right) \nonumber \\
        &\quad+ 2 \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D}\sigma}{\sqrt{b}} \nonumber \\
        &\quad+ \left( \frac{2 \beta^2}{1 - \beta} + \frac{1}{2} \right) L \eta \label{eq:theorem1-bound}
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
            + \left( \frac{2}{\theta} + \frac{1}{2} \right) L \eta
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
    \frac{2}{\theta} L \eta \leq c_2 \epsilon
        &\implies \eta \leq \frac{c_2}{2} \frac{\theta \epsilon}{L} \nonumber \\
    \frac{1}{2} L \eta \leq c_3 \epsilon
        &\implies \eta \leq 2 c_3 \frac{\epsilon}{L} \nonumber
\end{align}$$
For $0 < \theta \leq 1$, the first condition is the most restrictive, so we set,
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

## 3. Convergence bound for steepest descent under arbitrary norms with Nesterov momentum and decoupled weight decay for star-convex functions

From Theorem 14 in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), we have the following bound on the expected suboptimality when using steepest descent under an arbitrary norm $\| \cdot \|$ with Nesterov momentum and decoupled weight decay.

> **Theorem 3 (Expected suboptimality for steepest descent with Nesterov momentum and decoupled weight decay).** Let $\eta > 0$ be the learning rate, weight decay parameter $\lambda > 0$ (such that $\lambda\eta \leq 1$), Nesterov momentum parameter $\beta \in [0, 1)$, and initial momentum $M_0 = 0$. Then, under Assumptions (1)-(4) in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), star-convexity of $f$ at $W_*$, the bounded-weight conditions $\| W_0 \| \leq \frac{1}{\lambda}$ and $\| W_* \| \leq \frac{1}{\lambda}$, and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W_*) \right]
        &\leq (1 - \lambda\eta)^T \Delta_0 \nonumber \\
        &\quad+ \frac{2}{\lambda} \left(\sqrt{\frac{1 - \beta}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D} \sigma}{\sqrt{b}} \nonumber \\
        &\quad+ \left[
            \frac{4 L}{\lambda} \left(1 + \frac{\beta^2}{1 - \beta} \right)
            + \frac{2 \beta}{1 - \beta} G_0
        \right] \eta \label{eq:theorem2-bound}
\end{align}$$
where $\Delta_0 = f(W_0) - f(W_*)$ and $G_0 = \| \nabla f(W_0) \|^{\dagger}$.

We can then use this theorem to derive convergence bounds as follows.

> **Corollary 4.** For some expected suboptimality tolerance $0 < \epsilon < \Delta_0$, to ensure that, $\mathbb{E}\left[ f(W_T) - f(W_*) \right] \leq \epsilon$, via steepest descent under arbitrary norms with Nesterov momentum *with* decoupled weight decay, it suffices to set,
$$\begin{align}
    \theta &= 1 - \beta = \mathcal{O}\left(\min{\left\{1, \frac{\lambda^2 b \epsilon^2}{D \sigma^2}\right\}}\right) \\
    \eta &= \mathcal{O}\left(\min{\left\{\frac{\lambda\epsilon}{L}, \frac{\epsilon}{G_0},
        \frac{\lambda^3 b \epsilon^3}{D \sigma^2 L},
        \frac{\lambda^2 b \epsilon^3}{D \sigma^2 G_0}
    \right\}}\right) \\
    T &= \Omega\left(\max{\left\{
        \frac{L}{\lambda^2 \epsilon},
        \frac{G_0}{\lambda \epsilon},
        \frac{D \sigma^2 L}{\lambda^4 b \epsilon^3},
        \frac{D \sigma^2 G_0}{\lambda^3 b \epsilon^3}
    \right\}} \log\left(\frac{\Delta_0}{\epsilon}\right)\right)
\end{align}$$

**Proof.** As in the previous section, reparametrizing $\theta = 1 - \beta$ and using $\beta < 1$, we can simplify the bound in Theorem 3 as follows,
$$\begin{align}
    \mathbb{E}\left[ f(W_T) - f(W_*) \right]
        &\leq (1 - \lambda\eta)^T (f(W_0) - f(W_*))
            + \frac{8}{\lambda\theta} L \eta
            + \frac{2\eta}{\theta} G_0
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
    \frac{8}{\lambda\theta} L \eta \leq c_2 \epsilon
        &\implies \eta \leq \frac{c_2}{8} \frac{\lambda\theta\epsilon}{L} \nonumber \\
    \frac{2\eta}{\theta} G_0 \leq c_3 \epsilon
        &\implies \eta \leq \frac{c_3}{2} \frac{\theta\epsilon}{G_0} \nonumber
\end{align}$$

Combining these with the constraints $\lambda \eta \leq 1$ and $\| W_* \| \leq \frac{1}{\lambda}$, we set,
$$\begin{equation}
    \eta = \mathcal{O}\left( \min{\left\{ \| W_* \|, \frac{\lambda\theta\epsilon}{L}, \frac{\theta\epsilon}{G_0} \right\}} \right) \nonumber
\end{equation}$$
Substituting the bound on $\theta$ from above, we have,
$$\begin{equation}
    \eta = \mathcal{O}\left(\min{\left\{
        \| W_* \|,
        \frac{\lambda\epsilon}{L},
        \frac{\epsilon}{G_0},
        \frac{\lambda^3 b \epsilon^3}{D \sigma^2 L},
        \frac{\lambda^2 b \epsilon^3}{D \sigma^2 G_0}
    \right\}}\right) \label{eq:eta-bound-wd}
\end{equation}$$

### 3.3. Bounding T

For some small constant $c_4 < 1/4$, we then bound the term involving $T$ by $\epsilon$ as follows,

$$\begin{align}
    (1 - \lambda\eta)^T \Delta_0 \leq e^{- \lambda\eta T} \Delta_0 \leq c_4 \epsilon
        &\implies T \geq \frac{1}{\lambda\eta} \log{\left(\frac{\Delta_0}{c_4 \epsilon}\right)} \nonumber
\end{align}$$

Substituting the bound on $\eta$ from above, we have,
$$\begin{align}
    T &= \Omega\left(\max{\left\{
        \frac{L}{\lambda^2 \epsilon},
        \frac{G_0}{\lambda \epsilon},
        \frac{D \sigma^2 L}{\lambda^4 b \epsilon^3},
        \frac{D \sigma^2 G_0}{\lambda^3 b \epsilon^3}
    \right\}} \log\left(\frac{\Delta_0}{\epsilon}\right)\right) \label{eq:T-bound-final-wd}
\end{align}$$

## 4. Discussion

Here we have proven that steepest descent under arbitrary norms with Nesterov momentum with or without decoupled weight decay converges, with a universal convergence bound that holds for any norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$. We have also derived (universal) iteration complexity bounds for both cases, in terms of generalized expected stationarity and expected suboptimality, respectively.

For generalized expected stationarity, in the case without weight decay, the iteration complexity in Equation \eqref{eq:T-bound-final} is proportional to $1/\epsilon^4$ in the worst case, which is consistent with prior state-of-the-art results ([Ghadimi and Lan, 2013](https://doi.org/10.1137/120880811); [Cutkosky and Mehta, 2020](https://proceedings.mlr.press/v119/cutkosky20b.html); [Sun et al., 2023](https://proceedings.mlr.press/v202/sun23l.html); [Kovalev, 2025](https://arxiv.org/abs/2503.12645)) and cannot be improved further without additional assumptions ([Arjevani et al., 2022](https://doi.org/10.1007/s10107-022-01822-7)). For the expected suboptimality, in the case with decoupled weight decay, the iteration complexity is $\widetilde{\mathcal{O}}(1/\epsilon^3)$ in the worst case, where the logarithmic factor comes from $\log(\Delta_0 / \epsilon)$.

Interestingly, from the bounds in Equations \eqref{eq:eta-bound} and \eqref{eq:eta-bound-wd}, there seems to be a batch size threshold $b^*$ such that, up to which, increasing the batch size allows us to increase the learning rate, thereby reducing the number of iterations required to reach the desired stationary tolerance $\epsilon$. But beyond $b^*$, increasing the batch size no longer helps reduce the iteration complexity, as the bounds on $\eta$ and $\theta$ become independent of $b$, and $T$ starts to scale as $\Omega(1/\epsilon^2)$ (without weight decay) or $\widetilde{\Omega}(1/\epsilon)$ (with decoupled weight decay).

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
