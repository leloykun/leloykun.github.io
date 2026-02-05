---
title: "Shampoo-PRISM: Kronecker-Factored Optimization via Anisotropic Spectral Shaping"
date: 2026-02-04
tags: ["Machine Learning", "Optimizers", "Muon"]
author: ["Franz Louis Cesista"]
description: "A novel optimizer that combines Shampoo-style preconditioning with PRISM's anisotropic spectral shaping to adaptively suppress noisy gradient directions while maximally descending under the spectral norm trust-region constraint."
summary: "A novel optimizer that combines Shampoo-style preconditioning with PRISM's anisotropic spectral shaping to adaptively suppress noisy gradient directions while maximally descending under the spectral norm trust-region constraint."
---

## 1. Introduction

Consider the optimization problem,
$$\begin{align}
    W_* &= \arg\min_{W \in \mathcal{M}} f(W), \label{eq:original_problem}
\end{align}$$
where $f: \mathcal{M} \to \mathbb{R}$ is a differentiable and bounded-below objective function defined on a vector space $\mathcal{M} \subseteq \mathbb{R}^{m \times n}$. Expanding via Taylor's approximation around the current iterate $W_t$, and replacing the higher-order terms with a trust-region constraint yields the optimization subproblem,
$$\begin{align}
    W_{t+1}
        &= \arg\min_{W \in \mathcal{M}} f(W_t) + \langle G_t, W - W_t \rangle \quad \text{ s.t. } \quad \|W - W_t\| \leq \eta, \label{eq:taylor_expansion} \\
        &= \arg\min_{W \in \mathcal{M}} \langle G_t, W - W_t \rangle \quad \text{ s.t. } \quad \|W - W_t\| \leq \eta, \label{eq:taylor_expansion_2}
\end{align}$$
where $\eta > 0$ is the learning rate hyperparameter, and $G_t := \nabla f(W_t)$.

In the stochastic setting, we can replace the full gradient $G_t$ with its first-order moment estimate $M_t := \mathbb{E}[G_t]$. This leads to the stochastic optimization subproblem,
$$\begin{align}
    W_{t+1}
        &= \arg\min_{W \in \mathcal{M}} \langle M_t, W - W_t \rangle \quad \text{ s.t. } \quad \|W - W_t\| \leq \eta. \label{eq:stochastic_update}
\end{align}$$

And to prevent weights from blowing up, we can also introduce a decoupled weight decay term $\lambda > 0$ by 'shifting' the center of the constraint from $W_t$ to $(1 - \eta\lambda) W_t$ at each iteration, yielding the final optimization subproblem,
$$\begin{align}
    W_{t+1}
        &= \arg\min_{W \in \mathcal{M}} \langle M_t, W - (1 - \eta\lambda) W_t \rangle \quad \text{ s.t. } \quad \|W - (1 - \eta\lambda) W_t\| \leq \eta, \label{eq:final_update}
\end{align}$$
which has a solution, via the Linear Minimization Oracle (LMO) of the norm $\| \cdot \|$,
$$\begin{align}
    W_{t+1}
        &= (1 - \eta\lambda) W_t - \eta \texttt{LMO}_{\| \cdot \|}(M_t). \label{eq:lmo_solution}
\end{align}$$

Specializing to the spectral norm $\| \cdot \|_{2 \to 2}$ then yields the Muon optimizer ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/)),
$$\begin{align}
    W_{t+1}
        &= (1 - \eta\lambda) W_t - \eta \cdot \texttt{msign}(M_t) \label{eq:muon_update} \\
        &= (1 - \eta\lambda) W_t - \eta M_t \underbrace{(M_t^T M_t)^{-1/2}}_{P_t}, \label{eq:muon_update_2}
\end{align}$$
where $\texttt{msign}$ is the matrix sign operator.

But as discussed by [Yang (2026)](https://arxiv.org/abs/2602.03096), the preconditioner implicitly used by Muon, $P_t = (M_t^T M_t)^{-1/2} = (\mathbb{E}[G_t]^T \mathbb{E}[G_t])^{-1/2}$, only takes into account the first moment of the gradients, omitting second-order information that may be useful in stabilizing training in high-variance settings. E.g., Muon could take overly-aggressive steps along directions where the gradient variance is high, leading to suboptimal convergence behavior. They propose PRISM, which instead uses a covariance-aware preconditioner,
$$\begin{align}
    P_t
        &= (M_t^T M_t + \gamma^2 D_t^T D_t)^{-1/2} \approx (\mathbb{E}[G_t]^T \mathbb{E}[G_t] + \text{Cov}(G_t))^{-1/2}, \label{eq:prism_preconditioner}
\end{align}$$
where $D_t := G_t - M_t$ is called the 'momentum-based prediction', and $\gamma \geq 0$ is a hyperparameter controlling the strength of the covariance correction.

### 1.1. Shampoo-PRISM

Notice that, since PRISM only applies the preconditioner on the right side of $M_t$ in equation \eqref{eq:muon_update_2}, it only 'shapes' the updates using the geometry of the *column space* of $M_t$. I.e., it 'whitens' along the input-feature directions, but not the output-feature directions. And gradient noise can be anisotropic in either direction. A simple fix then is to apply PRISM-style anisotropic shaping on both sides, yielding the following Shampoo-style update rule ([Anil et al., 2020](https://arxiv.org/abs/2002.09018); [Gupta et al., 2018](https://arxiv.org/abs/1802.09568)),
$$\begin{align}
    W_{t+1}
        &= (1 - \eta\lambda) W_t - \eta \Delta W_t, \\
    \Delta W_t
        &= L_t M_t R_t,  \label{eq:shampoo_prism_update}
\end{align}$$
where the left and right preconditioners $L_t$ and $R_t$ are defined as,
$$\begin{align}
    L_t
        &= \widetilde{L}_t^{-1/4} && \widetilde{L}_t = M_t M_t^T + \gamma_L^2 D_t D_t^T, \label{eq:shampoo_prism_left} \\
    R_t
        &= \widetilde{R}_t^{-1/4} && \widetilde{R}_t = M_t^T M_t + \gamma_R^2 D_t^T D_t, \label{eq:shampoo_prism_right}
\end{align}$$
for some $\gamma_L, \gamma_R \geq 0$.

As to *why* the use of the $-1/4$ roots, firstly so that we recover Muon's update rule in Equation $\eqref{eq:muon_update}$ when $\gamma_L = \gamma_R = 0$; and secondly, because this makes the bidirectional anisotropic spectral shaping the geometric mean of the left- and right-sided one-sided PRISM shaping, as we will see in the next section.

## 2. Anisotropic spectral shaping of Shampoo-PRISM

Let $M_t = \sum_k \sigma_k u_k v_k^T$ be the singular value decomposition of $M_t$, where $\sigma_k \geq 0$ are the singular values, and $\{u_k\} \subset \mathbb{R}^m$ and $\{v_k\} \subset \mathbb{R}^n$ are the left and right singular vectors, respectively. We want to find coefficients $\rho_k^{\text{bi}} \in \mathbb{R}^{+}$ such that,
$$\begin{align}
    \Delta W_t
        &\approx \sum_k \rho_k^{\text{bi}} u_k v_k^T, \label{eq:shampoo_prism_svd}
\end{align}$$
where $\rho_k^{\text{bi}}$ modulates the magnitude along the direction $u_k v_k^T$ based on the signal-to-noise ratio (SNR) of the gradient along that direction. If the SNR is high, then we want $\rho_k^{\text{bi}} = 1$ as in Muon, and if the SNR is low, then we want to attenuate the update along that direction, i.e., $\rho_k^{\text{bi}} \ll 1$.

To ensure that we get a *scalar* $\rho_k^{\text{bi}}$, we assume that the modes $(u_k, v_k)$ are approximately also modes of the gram matrices, $\widetilde{L}_t$ and $\widetilde{R}_t$, i.e.,
$$\begin{align}
    \widetilde{L}_t u_k \approx \alpha_k u_k, \qquad \widetilde{R}_t v_k \approx \beta_k v_k, \label{eq:preconditioner_eigenvectors}
\end{align}$$
implying that,
$$\begin{align}
    L_t u_k
        &\approx \alpha_k^{-1/4} u_k, \qquad R_t v_k \approx \beta_k^{-1/4} v_k. \label{eq:preconditioner_eigenvalues}
\end{align}$$

We then have,
$$\begin{align}
    \alpha_k
        &= u_k^T \widetilde{L}_t u_k \nonumber \\
        &= u_k^T (M_t M_t^T + \gamma_L^2 D_t D_t^T) u_k, \nonumber \\
        &= \| M_t^T u_k \|_2^2 + \gamma_L^2 \| D_t^T u_k \|_2^2, \nonumber \\
        &= \| \sigma_k v_k \|_2^2 + \gamma_L^2 \| D_t^T u_k \|_2^2, \nonumber \\
        &= \sigma_k^2 + \gamma_L^2 \| D_t^T u_k \|_2^2, \\
\end{align}$$
and likewise,
$$\begin{align}
    \beta_k
        &= \sigma_k^2 + \gamma_R^2 \| D_t v_k \|_2^2. \\
\end{align}$$

Thus,
$$\begin{align}
    \rho_k^{\text{bi}}
        &= u_k^T L_t M_t R_t v_k \nonumber \\
        &\approx u_k^T (\alpha_k^{-1/4} M_t \beta_k^{-1/4}) v_k \nonumber \\
        &= \frac{\sigma_k}{(\alpha_k \beta_k)^{1/4}} \nonumber \\
        &= \frac{\sigma_k}{\sqrt[4]{(\sigma_k^2 + \gamma_L^2 \| D_t^T u_k \|_2^2)(\sigma_k^2 + \gamma_R^2 \| D_t v_k \|_2^2)}}
\end{align}$$
and defining the left- and right-sided SNRs as,
$$\begin{align}
    \text{SNR}_{L,k}
        &= \frac{\sigma_k}{\gamma_L \| D_t^T u_k \|_2} \qquad &&
    \text{SNR}_{R,k}
        = \frac{\sigma_k}{\gamma_R \| D_t v_k \|_2},
\end{align}$$
we can rewrite $\rho_k^{\text{bi}}$ as,
$$\begin{align}
    \rho_k^{\text{bi}}
        &= \frac{1}{\sqrt[4]{(1 + 1 / \text{SNR}_{L,k}^2)(1 + 1 / \text{SNR}_{R,k}^2)}}. \label{eq:shampoo_prism_rho}
\end{align}$$

Alternatively, we can also write $\rho_k^{\text{bi}}$ as the geometric mean of the one-sided PRISM coefficients,
$$\begin{align}
    \rho_k^{\text{bi}}
        &= \sqrt{\rho_k^{\text{left}} \cdot \rho_k^{\text{right}}} \label{eq:shampoo_prism_rho_geom}
\end{align}$$
Thus, if the signal-to-noise ratio is high on *both* sides, then $\rho_k^{\text{left}} \approx 1, \rho_k^{\text{right}} \approx 1 \implies \rho_k^{\text{bi}} \approx 1$, and we take full steps as in Muon; otherwise, if the SNR is low on *either* side, then $\rho_k^{\text{bi}} \ll 1$, and we take smaller to no steps along that direction.

### 2.1. Shampoo-PRISM follows the spectral norm trust-region constraint

Note that $0 \leq \frac{1}{\sqrt{1 + 1/x^2}} \leq 1$ for all $x \in \mathbb{R}$. Thus, both $\rho_k^{\text{left}}$ and $\rho_k^{\text{right}}$ lie in $[0, 1]$, and so does their geometric mean $\rho_k^{\text{bi}}$. Hence, $\| \eta \Delta W \|_{2 \to 2} = \eta \cdot \max_k \rho_k^{\text{bi}} \leq \eta$.

## 3. GPU/TPU-friendly implementation

From [Efficient Calculation of Matrix Square Root and Inverse Square Root](https://kexue.fm/archives/11158) and [Efficient Calculation of Matrix r-th Roots and Inverse r-th Roots](https://rohin-garg.github.io/kexue-en/translations/translation_11175.html), we can compute products of the form,
$$\begin{align}
    G P^{-s/r} \qquad \text{ and } \qquad Q^{-s/r} G P^{-s/r},
\end{align}$$
for $s, r \in \mathbb{Z}^+$ and SPD matrices $Q \in \mathbb{R}^{m \times m}$ and $P \in \mathbb{R}^{n \times n}$, using only matrix multiplications, additions, and scalar operations, which are efficient on GPUs and TPUs. This allows us to efficiently compute Equation \eqref{eq:shampoo_prism_update} directly as follows.

```python
import jax
import jax.numpy as jnp

# Coefficients taken from https://kexue.fm/archives/11175
coefs = [
    None,  # r = 0
    None,  # r = 1, omitted
    [      # r = 2
        (7.42487, -18.3958, 12.8967),
        (3.48773, -2.33004, 0.440469),
        (2.77661, -2.07064, 0.463023),
        (1.99131, -1.37394, 0.387593),
        (15 / 8, -5 / 4, 3 / 8),
    ],
    None,  # r = 3, omitted
    [      # r = 4
        (3.85003, -10.8539, 8.61893),
        (1.80992, -0.587778, 0.0647852),
        (1.50394, -0.594516, 0.121161),
        (45 / 32, -9 / 16, 5 / 32),
    ],
]

def abc(r=1, steps=None, scale=1):
    w, steps = coefs[r], steps or len(coefs[r])
    for a, b, c in w[:steps] + w[-1:] * max(steps - len(w), 0):
        yield a / scale, b / scale**(r + 1), c / scale**(2 * r + 1)

def matmul_invroot(G: jax.Array, P: jax.Array, r: int, s=1, steps=None, eps=1e-5, scale: float=1.001):
    # Computes G @ P^(-s/r)
    I = jnp.eye(P.shape[0], dtype=P.dtype)
    P = P / (t := (P * P.mT).sum()**0.5) + eps * I
    for a, b, c in abc(r, steps, scale=scale):
        W = a * I + b * P + c * P @ P
        W1, W2 = jnp.linalg.matrix_power(W, s), jnp.linalg.matrix_power(W, r)
        G, P = G @ W1, P @ W2
    return G * t**(-s/r)

def double_sided_matmul_invroot(Q: jax.Array, G: jax.Array, P: jax.Array, *, r: int, s=1, steps=None, eps: float=1e-5, scale: float=1.001):
    # Computes Q^(-s/r) @ G @ P^(-s/r)
    I_m, I_n = jnp.eye(G.shape[0], dtype=Q.dtype), jnp.eye(G.shape[1], dtype=P.dtype)
    Q = Q / (tQ := jnp.sum(Q * Q.T)**0.5) + eps * I_m
    P = P / (tP := jnp.sum(P * P.T)**0.5) + eps * I_n
    for a, b, c in abc(r, steps, scale=scale):
        WQ = a * I_m + b * Q + c * Q @ Q
        WP = a * I_n + b * P + c * P @ P
        WQ1, WQ2 = jnp.linalg.matrix_power(WQ, s), jnp.linalg.matrix_power(WQ, r)
        WP1, WP2 = jnp.linalg.matrix_power(WP, s), jnp.linalg.matrix_power(WP, r)
        Q, G, P = Q @ WQ2, WQ1 @ G @ WP1, P @ WP2
    G = G * tQ**(-s/r) * tP**(-s/r)
    return G

def shampoo_prism(M: jax.Array, D: jax.Array, *, gamma_L=0.0, gamma_R=0.0, eps_gram=1e-6, inv_steps=8, inv_eps=1e-5, inv_scale=1.001):
    H_L = M @ M.T + gamma_L**2 * D @ D.T + eps_gram * jnp.eye(M.shape[0], dtype=M.dtype)
    H_R = M.T @ M + gamma_R**2 * D.T @ D + eps_gram * jnp.eye(M.shape[1], dtype=M.dtype)
    O = double_sided_matmul_invroot(H_L, M, H_R, r=4, steps=inv_steps, eps=inv_eps, scale=inv_scale)
    # Alternatively,
    # MR = matmul_invroot(M, H_R, r=4, steps=inv_steps, eps=inv_eps)
    # O  = matmul_invroot(MR.T, H_L, r=4, steps=inv_steps, eps=inv_eps).T
    return O
```

## How to cite

```bibtex
@misc{cesista2026shampooprism,
  author = {Franz Louis Cesista},
  title = {{Shampoo-PRISM}: {K}ronecker-Factored Optimization via Anisotropic Spectral Shaping},
  year = {2026},
  month = {February},
  day = {4},
  url = {https://leloykun.github.io/ponder/shampoo-prism/},
}
```

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
2. Yujie Yang (2026). PRISM: Structured Optimization via Anisotropic Spectral Shaping. Available at: https://arxiv.org/abs/2602.03096
3. Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer (2020). Scalable second order optimization for deep learning. URL https://arxiv.org/abs/2002.09018
4. Vineet Gupta, Tomer Koren, Yoram Singer (2018). Shampoo: Preconditioned Stochastic Tensor Optimization. URL https://arxiv.org/abs/1802.09568
5. Jianlin Su (2025). Efficient Calculation of Matrix Square Root and Inverse Square Root. URL https://kexue.fm/archives/11158
6. Jianlin Su (2025). Efficient Calculation of Matrix r-th Roots and Inverse r-th Roots. URL https://rohin-garg.github.io/kexue-en/translations/translation_11175.html

## Appendix

### A1. Optimized PRISM

In the original PRISM paper, we need to construct the $2m \times n$ matrix $\widetilde{M}_t$ and then apply the orthogonalization operator to this larger matrix. This wastes both GPU memory and compute. Instead, we can directly compute the preconditioner $P_t$ in Equation \eqref{eq:prism_preconditioner}, and $M_t P_t^{-1/2}$ using the matrix multiply-with-inverse-root discussed in [Section 3](#3-gputpu-friendly-implementation) above, as shown below.

```python
def prism_v2(M: jax.Array, D: jax.Array, *, gamma=0.0, eps_gram=1e-6, inv_steps=8, inv_eps=1e-5, inv_scale=1.001):
    H_R = M.T @ M + gamma**2 * D.T @ D + eps_gram * jnp.eye(M.shape[1], dtype=M.dtype)
    return matmul_invroot(M, H_R, r=2, steps=inv_steps, eps=inv_eps, scale=inv_scale)
```

This only costs $\mathcal{O}(n^2)$ in extra memory, instead of $\mathcal{O}(2mn)$. And for $T$ iterations, it only costs $\mathcal{O}((2+T)mn^2 + 3Tn^3)$ flops vs. $\mathcal{O}(4Tmn^2 + Tn^3)$ flops in the original implementation. For $4n \times n$ weight matrices commonly found in up- and down-projections in MLPs in Llama-like models, this results in a $8\times$ memory saving and $2.125\times$ speedup.
