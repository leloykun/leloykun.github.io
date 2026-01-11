---
title: "Steepest Descent on Affine-Conic Representable Manifolds with Boundary via Dual Ascent"
date: 2026-01-09
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "Novel optimizers for maximally descending on the loss landscape while satisfying strict weight constraints."
summary: "Novel optimizers for maximally descending on the loss landscape while satisfying strict weight constraints."
---

## 1. Introduction

Consider the optimization problem,
$$\begin{align}
    W^* = \arg\min_{W \in \mathcal{M}} f(W),
\end{align}$$
where $f: \mathcal{M} \to \mathbb{R}$ is a differentiable and bounded-below objective function defined on a normed manifold or [manifold with boundary/corners](https://ncatlab.org/nlab/show/manifold+with+boundary) $\mathcal{M}$. There are practical considerations on whether or not to include the boundary of $\mathcal{M}$: first, we often only have access to retraction maps that map to the boundary from 'outside' the manifold (e.g. the [PSD cone](https://en.wikipedia.org/wiki/Definite_matrix) and the [Spectral Ball](../steepest-descent-finsler-dual-ascent/#33-special-case-1-steepest-descent-on-the-spectral-ball-under-the--norm) of radius $R$); second, our update rules have to differ when we are at the boundaries and failure to account for this may lead to suboptimal solutions or divergence.

In [previous](../steepest-descent-finsler-dual-ascent/) [blog](../steepest-descent-crit-bz/) [posts](../steepest-descent-doubly-stochastic/), we discussed manifolds where the tangent space (at interior points) or tangent cone (at boundary points) $T_{W}\mathcal{M}$ at any point $W \in \mathcal{M}$ can be represented in the affine-conic form,
$$\begin{align}
    T_{W}\mathcal{M} = \{ A \in \mathbb{R}^{m \times n} : L_W(A) + b_W \in -K \},
\end{align}$$
where $L_W: \mathbb{R}^{m \times n} \to \mathcal{Y}$ is a linear map (possibly point-dependent), $b_W \in \mathcal{Y}$ is a constant offset (often $b_W = 0$), and $K \subseteq \mathcal{Y}$ is a closed convex cone. First-order optimization on such manifolds can then be done as,
$$\begin{align}
W_{t+1}
        &= \texttt{retract}_{\mathcal{M}}(W_{t} + A^*_t),
\end{align}$$
where $\texttt{retract}_{\mathcal{M}}$ is a retraction map that maps points back to the manifold $\mathcal{M}$, and $A_t^*$ is the solution to either a constrained $\eqref{eq:constrained_tangent_update}$ or regularized $\eqref{eq:regularized_tangent_update}$ linearized subproblem,
$$\begin{align}
    A_t^*
        &= \arg\min_{A \in T_{W_t}\mathcal{M}} f(W_t) + \langle G_t, A \rangle \quad \text{ s.t. } \quad \| A \|_{W_t} \leq \eta \label{eq:constrained_tangent_update}\tag{C1} \\
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle G_t, A \rangle \quad \text{ s.t. } \quad \| A \|_{W_t} \leq \eta, \quad A \in T_{W_t}\mathcal{M} \\
    A_t^*
        &= \arg\min_{A \in T_{W_t}\mathcal{M}} f(W_t) + \langle G_t, A \rangle + \frac{1}{2\eta} \| A \|_{W_t}^2 \label{eq:regularized_tangent_update}\tag{R1} \\
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle G_t, A \rangle +  \frac{1}{2\eta} \| A \|_{W_t}^2 \quad \text{ s.t. } \quad A \in T_{W_t}\mathcal{M} \\
\end{align}$$
where $G_t \in \mathbb{R}^{m \times n}$ is the Riemannian gradient (or differential) of $f$ at $W_t$ (computed via backpropagation), $\|\cdot\|_{W_t}$ is the chosen norm at point $W_t$, and $\eta > 0$ is the learning rate hyperparameter.

The problem with this approach is that the 'boundary-aware' constraints only activate *at* the boundaries. So we could be infinitesimally close to the boundary, but still ignore the possibility of crossing over it. In this blog post, we present an alternative approach where we directly constrain $W_{t} + A_t^*$ to lie in $\mathcal{M}$, or at least be as close as possible to $\mathcal{M}$.

## 2. Optimization on affine-conic representable manifolds with boundary

Let $\mathcal{M}$ be a manifold with boundary that can be represented in the affine-conic form,
$$\begin{align}
    \mathcal{M} &= \{ W \in \mathbb{R}^{m \times n} : L(W) + b \in -K \},
\end{align}$$
where $L: \mathbb{R}^{m \times n} \to \mathcal{Y}$ is a linear map, $b \in \mathcal{Y}$ is a constant offset (often $b = 0$), and $K \subseteq \mathcal{Y}$ is a closed convex cone. Some manifolds such as the [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold) and the [Spectral Band](../steepest-descent-finsler-dual-ascent/#32-steepest-descent-on-the-spectral-band-under-the--norm) can no longer be represented in this form, but some important ones can. Concrete examples:
1. The Spectral Ball of radius $R$:
$$\begin{align}
    \mathcal{M}
        &= \{ W \in \mathbb{R}^{m \times n} : \| W \|_{2 \to 2} \leq R \} \nonumber \\
        &= \Bigl\{ W \in \mathbb{R}^{m \times n} : \begin{bmatrix}
            RI_m & W \\
            W^T  & RI_n
        \end{bmatrix} \succeq \mathbf{0} \Bigr\}, \nonumber \\
        &= \Bigl\{ W \in \mathbb{R}^{m \times n} : \begin{bmatrix}
            \mathbf{0} & W \\
            W^T        & \mathbf{0}
        \end{bmatrix} + \begin{bmatrix}
            RI_m       & \mathbf{0} \\
            \mathbf{0} & RI_n
        \end{bmatrix} \in -\mathbb{S}_{-}^{m+n} \Bigr\},
\end{align}$$
2. The Birkhoff Polytope:
$$\begin{align}
    \mathcal{M}
        &= \{ W \in \mathbb{R}^{m \times n} : W \mathbf{1}_n = \mathbf{1}_m, W^T \mathbf{1}_m = \mathbf{1}_n, W_{ij} \geq 0\} \nonumber \\
        &= \Bigl\{ W \in \mathbb{R}^{m \times n} : (W \mathbf{1}_n, W^T \mathbf{1}_m, W) + (-\mathbf{1}_m, -\mathbf{1}_n, \mathbf{0}) \in -(\{\mathbf{0}\}, \{\mathbf{0}\}, R_{-}^{m \times n}) \Bigr\},
\end{align}$$

$\blacksquare$ We can then either solve the constrained $\eqref{eq:constrained_update}$ or regularized $\eqref{eq:regularized_update}$ linearized subproblem,

$$\begin{align}
    W_{t+1}
        &= \arg\min_{W \in \mathcal{M}} f(W_t) + \langle G_t, W - W_t \rangle \quad \text{ s.t. } \quad \| W - W_t \|_{W_t} \leq \eta \label{eq:constrained_update}\tag{C2} \\
        &= \arg\min_{W \in \mathbb{R}^{m \times n}} \langle G_t, W - W_t \rangle \quad \text{ s.t. } \quad \| W - W_t \|_{W_t} \leq \eta, \quad W \in \mathcal{M} \label{eq:constrained_update_explicit} \\
    W_{t+1}
        &= \arg\min_{W \in \mathcal{M}} f(W_t) + \langle G_t, W - W_t \rangle + \frac{1}{2\eta} \| W - W_t \|_{W_t}^2 \label{eq:regularized_update}\tag{R2} \\
        &= \arg\min_{W \in \mathbb{R}^{m \times n}} \langle G_t, W - W_t \rangle +  \frac{1}{2\eta} \| W - W_t \|_{W_t}^2 \quad \text{ s.t. } \quad W \in \mathcal{M}
\end{align}$$
where $\| \cdot \|_{W_t}$ is the chosen norm at point $W_t$. For now, we will focus on solving the constrained problem $\eqref{eq:constrained_update}$. The regularized problem $\eqref{eq:regularized_update}$ can be solved in a similar manner. To simplify notation, we drop the subscript $W_t$ from $\|\cdot\|_{W_t}$ in the rest of this section, but keep in mind that it could be point-dependent.

Now let $A = W - W_t$. Then, we can rewrite Equation $\eqref{eq:constrained_update_explicit}$ as,
$$\begin{align}
    A_t^*
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle G_t, A \rangle \quad \text{ s.t. } \quad \| A \| \leq \eta, \quad L(W_t + A) + b \in -K \label{eq:optimaldescent} \\
    W_{t+1}
        &= W_t + A_t^*.
\end{align}$$
Let $\mathcal{Y}^*$ be the dual space of $\mathcal{Y}$, then the adjoint of $L$, $L^*: \mathcal{Y}^* \to \mathbb{R}^{m \times n}$, is defined as the unique linear map satisfying,
$$\langle L(A), Y \rangle = \langle A, L^*(Y) \rangle, \quad \forall A \in \mathbb{R}^{m \times n}, Y \in \mathcal{Y}^*.$$

Restricting $Y$ to the dual space $K^* \subseteq \mathcal{Y}^*$ then yields the Lagrangian of Equation $\eqref{eq:optimaldescent}$,
$$\begin{align}
    \mathcal{L}(A, Y) &= \langle G_t, A \rangle + \mathcal{i}_{\| \cdot \| \leq \eta}(A) + \langle Y, L(W_t + A) + b \rangle \nonumber \\
        &= \mathcal{i}_{\| \cdot \| \leq \eta}(A) + \langle G_t + L^*(Y), A \rangle + \langle Y, L(W_t) + b \rangle
\end{align}$$
where $\mathcal{i}_S$ is the indicator function of set $S$ defined as,
$$\mathcal{i}_S(X) = \begin{cases}
    0 & X \in S \\
    +\infty & X \notin S
\end{cases}.$$

One can then check that,
$$A^*_t = \arg\min_{A \in \mathbb{R}^{m \times n}} \left[ \max_{Y \in K^*} \mathcal{L}(A, Y) \right]$$
which, by Sion's minimax theorem, we can solve by iteratively switching the order of minimization and maximization,
$$ \min_{A \in \mathbb{R}^{m \times n}} \left[ \max_{Y \in K^*} \mathcal{L}(A, Y) \right] = \max_{Y \in K^*} \left[ \underbrace{\min_{A \in \mathbb{R}^{m \times n}} \mathcal{L}(A, Y)}_{\text{minimizer: } A^*(Y)} \right]$$

First, let us consider the primal minimizer,
$$\begin{align}
    A^*(Y)
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \mathcal{L}(A, Y) \nonumber \\
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \mathcal{i}_{\| \cdot \| \leq \eta}(A) + \langle G_t + L^*(Y), A \rangle + \cancel{\langle Y, L(W_t) + b \rangle} \nonumber \\
        &= \arg\min_{\| A \| \leq \eta} \langle G_t + L^*(Y), A \rangle \nonumber \\
        &= -\eta\cdot\texttt{LMO}_{\| \cdot \|}(G_t + L^*(Y)),
\end{align}$$
where $\texttt{LMO}_{\| \cdot \|}(Z) = \arg\max_{\| A \| \leq 1} \langle Z, A \rangle$ is the Linear Minimization Oracle under norm $\| \cdot \|$.

Substituting $A^*(Y)$ back into the Lagrangian then yields the dual problem,
$$\begin{align}
    h(Y)
        &= \max_{Y \in K^*} \mathcal{L}(A^*(Y), Y) \nonumber \\
        &= \max_{Y \in K^*} \langle G_t + L^*(Y), -\eta\cdot\texttt{LMO}_{\| \cdot \|}(G_t + L^*(Y)) \rangle + \langle Y, L(W_t) + b \rangle \nonumber \\
        &= -\eta \| G_t + L^*(Y) \|^\dagger + \langle Y, L(W_t) + b \rangle
\end{align}$$
where $\| \cdot \|^\dagger$ is the dual norm of $\| \cdot \|$. And by chain rule, the dual problem above has *a* supergradient,
$$\begin{align}
    \nabla_{Y} h(Y)
        &\ni -\eta\cdot L\left(\texttt{LMO}_{\| \cdot \|}(G_t + L^*(Y))\right) + L(W_t) + b \nonumber \\
        &= L(A^*(Y)) + L(W_t) + b \nonumber \\
        &= L(W_t + A^*(Y)) + b
\end{align}$$
which we can use to do gradient ascent on the dual variable $Y$. And finally, to maintain $Y \in K^*$, we project the updated dual variable back to $K^*$ after each ascent step.

$\blacksquare$ Putting everything together, we have the following update rule for the primal and dual variables $A^j_t$ and $Y^{j+1}_t$,
$$\begin{align}
    A^j_t
        &= -\eta\cdot\texttt{LMO}_{\| \cdot \|}(G_t + L^*(Y^{j}_t)) \\
    Y^{j+1}_t
        &= \texttt{proj}_{K^*} \left(Y^{j}_t + \sigma_j \left( L( W_t + A^j_t ) + b \right)\right)
\end{align}$$
or equivalently,
$$\begin{align}
    W^j_{t+1}
        &= W_t - \eta\cdot\texttt{LMO}_{\| \cdot \|}(G_t + L^*(Y^{j}_t)) \\
    Y^{j+1}_t
        &= \texttt{proj}_{K^*} \left(Y^{j}_t + \sigma_j \left( L( W_{t+1}^j ) + b \right)\right),
\end{align}$$
where $\sigma_j > 0$ is the dual ascent learning rate, and $\texttt{proj}_{K^*}$ is the orthogonal projection onto the dual cone $K^*$. Literature on dual ascent typically recommend using a learning rate schedule of $\sigma_j = \sigma_{0}/\sqrt{j+1}$. And if $K = \{ 0 \}$, the projection is simply the identity map. At convergence, we have $W^j_{t+1} \to W_{t+1}$.

See [Appendix A1](#a1-jax-implementation-of-the-dual-ascent-optimizer) for implementation in JAX.

## How to cite

```bibtex
@misc{cesista2026steepestdescentaffineconic,
  author = {Franz Louis Cesista},
  title = {{S}teepest Descent on Affine-Conic Representable Manifolds with Boundary via Dual Ascent},
  year = {2026},
  month = {January},
  day = {9},
  url = {https://leloykun.github.io/ponder/steepest-descent-affine-conic/},
}
```

## Appendix

### A1. JAX implementation of the dual ascent optimizer

```python
def dual_ascent_faithful(
    W: jax.Array,  # R^(m x n)
    G: jax.Array,  # R^(m x n)
    eta: float,  # learning rate
    B: Tuple[jax.Array],  # K_dual
    L_primal: Callable[[jax.Array], Tuple[jax.Array]],  # R^(m x n) -> K_dual
    L_dual:  Callable[[Tuple[jax.Array]], jax.Array],  # K_dual -> R^(m x n)
    proj_K_dual: Callable[[Tuple[jax.Array]], Tuple[jax.Array]],  # K_dual -> K_dual
    norm_K_dual: Callable[[Tuple[jax.Array]], float],  # K_dual -> R
    lmo: Callable[[jax.Array], jax.Array],  # R^(m x n) -> R^(m x n)
    *,
    max_steps: int=128, sigma: float=1.0,
    rtol: float=1e-3, atol: float=1e-6,
):
    S_init = proj_K_dual(jax.tree_util.tree_map(lambda s, b: s + b, L_primal(W), B))
    # S_init = jax.tree_util.tree_map(lambda s: jnp.zeros_like(s), S_init)

    def cond_fn(state):
        S, k, res = state
        return jnp.logical_and(k < max_steps, jnp.logical_and(res > atol, res > rtol * norm_K_dual(S)))

    def body_fn(state):
        S, k, _ = state
        W_next = W - eta * lmo(G + L_dual(S))
        grad_S = jax.tree_util.tree_map(lambda pre_grad_s, b: pre_grad_s + b, L_primal(W_next), B)
        S_new = proj_K_dual(jax.tree_util.tree_map(lambda s, g: s + sigma / jnp.sqrt(k+1) * g, S, grad_S))
        res = norm_K_dual(grad_S)
        return S_new, k+1, res

    S_final, n_iters, final_res = jax.lax.while_loop(cond_fn, body_fn, (S_init, 0, jnp.inf))
    A_final = -eta * lmo(G + L_dual(S_final))
    return A_final
```
