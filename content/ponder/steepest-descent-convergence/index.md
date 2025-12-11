---
title: "Convergence Bounds for Steepest Descent Under Arbitrary Norms"
date: 2025-12-11
tags: ["Machine Learning", "Optimizers"]
author: ["Franz Louis Cesista"]
description: "First-order optimization under arbitrary norms with Nesterov momentum (and weight decay) yields a universal convergence bound. Our results generalize to norms not induced by inner products, and also considers batch size."
summary: "First-order optimization under arbitrary norms with Nesterov momentum (and weight decay) yields a universal convergence bound. Our results generalize to norms not induced by inner products, and also considers batch size."
---

## 1. Introduction

From Theorem 11 in [Ponder: Critical Batch Size for Steepest Descent Under Arbitrary Norms](../steepest-descent-crit-bz/), we have the following bound on the average expected gradient norm when using steepest descent under an arbitrary norm $\| \cdot \|$ with Nesterov momentum and weight decay.

> Let $W_t$ be the weight at time step $t$, $\lambda$ be the weight decay parameter, and $\eta > 0$ be the step size such that $\lambda \eta \leq 1$. Assume that the initial weight norm satisfies $\| W_0 \| \leq \frac{1}{\lambda}$, and the initial momentum $M_0 = 0$. Finally, let $D$ be the Lipschitz constant of $g(\cdot) = \frac{1}{2}\| \cdot \|^{\dagger 2}$ (with $D = 1$ if $\| \cdot \|^{\dagger}$ is induced by an inner product). Then for any norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}]
        &\leq \frac{1}{T} \left( \frac{\Delta_0}{\eta}
            + \frac{2 \beta}{1 - \beta} G_0
            + \frac{2 \beta}{D (1 - \beta^2)} G_0^2 \right) \nonumber \\
        &\quad+ \frac{(3 \beta + 1)(1 - \beta)}{2(1 + \beta)} \frac{\sigma^2}{b} \nonumber \\
        &\quad+ \left(\sqrt{\frac{2 (1 - \beta)}{1 + \beta}} \beta + (1 - \beta)\right) \frac{\sqrt{D}\sigma}{\sqrt{b}} \nonumber \\
        &\quad+ \frac{2 \beta^2}{1 - \beta} L \eta
            + \frac{2 \beta^3}{D (1 - \beta)^2} L^2 \eta^2
            + \frac{L\eta}{2} \nonumber \\
        &\quad+ \frac{\lambda}{2}\frac{\sigma^2}{b}
            + \frac{2 L^2}{\lambda D}
            + \frac{D}{\lambda}
            + 2D
\end{align}$$
where $\Delta_0 = f(W_0) - f^*$, and $G_0 = \| \nabla f(W_0) \|^{\dagger}$.

For a given stationary tolerance $\epsilon > 0$, we want to determine bounds on the step size $\eta$, momentum parameter $\beta$, and total number of time steps $T$ such that,
$$\begin{equation}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \text{ - [noise floor]} \leq \epsilon
\end{equation}$$

## 2. Convergence analysis

First, note that as $T \to \infty$, only the (first) term involving $\frac{1}{T}$ vanishes. This term depends on hyperparameters $\eta$ and $\beta$, so we will need to set bounds on these hyperparameters to ensure convergence to the noise floor. We then fold the terms depending only on $\lambda, \sigma, b, L, D$ into the noise floor.

Now, for $\beta \approx 1$, and reparametrizing $\theta = 1 - \beta$, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|^{\dagger}] \text{ - [noise floor]}
        &\lesssim \frac{1}{T} \left( \frac{\Delta_0}{\eta}
            + \frac{2}{\theta} G_0
            + \frac{2}{D \theta} G_0^2 \right) \nonumber \\
        &\quad+ \theta \frac{\sigma^2}{b}
            + 2\sqrt{\theta} \frac{\sqrt{D}\sigma}{\sqrt{b}} \nonumber \\
        &\quad+ \frac{2}{\theta} L \eta
            + \frac{2}{D \theta^2} L^2 \eta^2
            + \frac{L\eta}{2} \nonumber \\
        &\leq \epsilon \nonumber
\end{align}$$

It then suffices to force each term to be $\mathcal{O}(\epsilon)$.

### 2.1. Bounding θ

For small constants $c_1$ and $c_2$, say $c_1, c_2 \leq 1/8$, we want to bound the terms involving $\theta$ by $\epsilon$ as follows,
$$\begin{align}
    \theta \frac{\sigma^2}{b} \leq c_1 \epsilon
        &\implies \theta \leq c_1 \frac{b\epsilon}{\sigma^2} \nonumber \\
    2\sqrt{\theta} \frac{\sqrt{D}\sigma}{\sqrt{b}} \leq \epsilon
        &\implies \theta \leq \frac{c_2^2}{4} \frac{b\epsilon^2}{D\sigma^2} \nonumber
\end{align}$$
For small $\epsilon$, the second condition is more restrictive, so we set,
$$\begin{equation}
    \theta = \mathcal{O}\left(\min{\left\{1, \frac{b\epsilon^2}{D\sigma^2}\right\}}\right) \label{eq:theta-bound}
\end{equation}$$

### 2.2. Bounding η

For small constants $c_3, c_4, c_5$, we then bound the terms involving $\eta$ by $\epsilon$ as follows,
$$\begin{align}
    \frac{2}{\theta} L \eta \leq c_3 \epsilon
        &\implies \eta \leq \frac{c_3}{2} \frac{\theta \epsilon}{L} \nonumber \\
    \frac{2}{D \theta^2} L^2 \eta^2 \leq c_4 \epsilon
        &\implies \eta \leq \frac{\sqrt{c_4}}{\sqrt{2}} \frac{\theta \sqrt{D \epsilon}}{L} \nonumber \\
    \frac{L\eta}{2} \leq c_5 \epsilon
        &\implies \eta \leq 2 c_5 \frac{\epsilon}{L} \nonumber
\end{align}$$
For $0 \leq \theta \leq 1$ and small $\epsilon$, the first condition is the most restrictive, so we set,
$$\eta = \mathcal{O}\left( \frac{\theta \epsilon}{L} \right)$$
Substituting the bound on $\theta$ from Equation \eqref{eq:theta-bound}, we have,
$$\begin{equation}
    \eta = \mathcal{O}\left(\min{\left\{\frac{\epsilon}{L}, \frac{b\epsilon^3}{D\sigma^2L}\right\}}\right) \label{eq:eta-bound}
\end{equation}$$

### 2.3. Bounding T

For small constants $c_6, c_7, c_8$, we then bound the terms involving $T$ by $\epsilon$ as follows,
$$\begin{align}
    \frac{1}{T} \frac{\Delta_0}{\eta} \leq c_6 \epsilon
        &\implies T \geq \frac{1}{c_6} \frac{\Delta_0}{\eta \epsilon} \label{eq:T-bound-1} \\
    \frac{1}{T} \frac{2}{\theta} G_0 \leq c_7 \epsilon
        &\implies T \geq \frac{2}{c_7} \frac{G_0}{\theta \epsilon} \label{eq:T-bound-2} \\
    \frac{1}{T} \frac{2}{D \theta} G_0^2 \leq c_8 \epsilon
        &\implies T \geq \frac{2}{c_8} \frac{G_0^2}{D \theta \epsilon}  \label{eq:T-bound-3}
\end{align}$$

Substituting the bound on $\eta$ from Equation \eqref{eq:eta-bound} into Equation \eqref{eq:T-bound-1}, we have,
$$\begin{equation}
    T = \Omega\left(\max{\left\{
        \frac{L \Delta_0}{\epsilon^2},
        \frac{D \sigma^2 L \Delta_0}{b \epsilon^4}
    \right\}}\right) \nonumber
\end{equation}$$

Likewise, substituting the bound on $\theta$ from Equation \eqref{eq:theta-bound} into Equations \eqref{eq:T-bound-2} and \eqref{eq:T-bound-3}, we have,
$$\begin{align}
    T &= \Omega\left(\max{\left\{
        \frac{G_0}{\epsilon},
        \frac{G_0^2}{D \epsilon},
        \frac{D \sigma^2 G_0}{b \epsilon^3},
        \frac{\sigma^2 G_0^2}{b \epsilon^3}
    \right\}}\right) \nonumber
\end{align}$$

Thus,
$$\begin{align}
    T &= \Omega\left(\max{\left\{
        \frac{G_0}{\epsilon},
        \frac{G_0^2}{D \epsilon},
        \frac{L \Delta_0}{\epsilon^2},
        \frac{D \sigma^2 G_0}{b \epsilon^3},
        \frac{\sigma^2 G_0^2}{b \epsilon^3},
        \frac{D \sigma^2 L \Delta_0}{b \epsilon^4}
    \right\}}\right) \label{eq:T-bound-final}
\end{align}$$

## 3. Discussion

The iteration complexity in Equation \eqref{eq:T-bound-final} is proportional to $1/\epsilon^4$ in the worst case, which is consistent with prior state-of-the-art results ([Ghadimi and Lan, 2013](https://doi.org/10.1137/120880811); [Cutkosky and Mehta, 2020](https://proceedings.mlr.press/v119/cutkosky20b.html); [Sun et al., 2023](https://proceedings.mlr.press/v202/sun23l.html); [Kovalev, 2025](https://arxiv.org/abs/2503.12645)) and cannot be improved further without additional assumptions ([Arjevani et al., 2022](https://doi.org/10.1007/s10107-022-01822-7)).

Interestingly, from the bounds in Equations \eqref{eq:theta-bound} and \eqref{eq:eta-bound}, there seems to be a batch size threshold $b^*$ such that, up to which, increasing the batch size allows us to increase the learning rate and let $\beta \to 1$, thereby reducing the number of iterations required to reach the desired stationary tolerance $\epsilon$. But beyond $b^*$, increasing the batch size no longer helps reduce the iteration complexity, as the bounds on $\eta$ and $\theta$ become independent of $b$, and $T$ starts to scale as $\Omega(1/\epsilon^2)$.

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
