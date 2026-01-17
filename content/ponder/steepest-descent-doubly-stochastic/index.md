---
title: "Steepest Descent on the Birkhoff Polytope Equipped with the Spectral Norm"
date: 2026-01-04
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "We derive an optimizer that performs steepest descent on the Birkhoff polytope equipped with the spectral norm via dual ascent. We show that it yields larger effective weight updates than naive LMO-based optimizers."
summary: "We derive an optimizer that performs steepest descent on the Birkhoff polytope equipped with the spectral norm via dual ascent. We show that it yields larger effective weight updates than naive LMO-based optimizers."
---

## 1. Introduction

Deepseek recently published a paper titled [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) where they fix training instabilities introduced by the [Hyper-Connections](https://arxiv.org/abs/2409.19606) paper by constraining the weight matrices to be doubly stochastic, i.e., elements of the Birkhoff polytope. The crux is that, to prevent the activations and gradients from blowing up, the residual transform $A_l$ in the following residual block for Hyper-Connections has to be non-expansive.
$$\begin{equation}
    x_{l+1} = A_l x_l + B_l^T f(C_l x_l, W_l)
\end{equation}$$
The instability problem becomes clearer when we recursively extend hyper-connections to multiple layers,
$$\begin{equation}
    x_{L}
        = \left(\prod_{i=1}^{L-l} A_{L-i}\right) x_l
            + \sum_{i=l}^{L-1} \left( \prod_{j=1}^{L-1-i} A_{L-j} \right) B_i^T f(C_i x_i, W_i)
\end{equation}$$
where $L$ and $l$ are indices for a deeper and a shallower layer, respectively. If $\| A_l \|_{2 \to 2} > 1$, then the product $\| \prod_{i=1}^{L-l} A_{L-i} \|_{2 \to 2}$ explodes.

The obvious fix is to simply constrain $A_l$ such that $\| A_l \|_{2 \to 2} \leq 1$. Any subset of the spectral ball of radius 1 works so long as we can form at least a semigroup under matrix multiplication. We could, for example, constrain $A_l$ to be orthogonal, or cap the eigenvalues by 1 as in Section 2 of [Ponder: Rethinking Maximal Update Parametrization: Steepest Descent on the Spectral Ball](rethinking-mup-spectral-ball/#2-eigenvalue-clipping). Deepseek chose to constrain $A_l$ to be a doubly stochastic matrix, which guarantees $\| A_l \|_{2 \to 2} \leq 1$ by the [Perron-Frobenius theorem](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem) (but some direction(s) may be contractive).

Oddly enough, despite having "manifold" in the title, they do not actually perform optimization on the Birkhoff polytope nor is it even a manifold (in the classical sense). This polytope has "boundaries" and "corners" where we no longer have tangent spaces, but rather tangent *cones*. They do prevent $A_l$ from landing on the boundaries by exponentiating the entries before projecting onto the Birkhoff polytope using the [Sinkhorn-Knopp operator](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem#Sinkhorn%E2%80%93Knopp_algorithm)--and the interior of the Birkhoff polytope is indeed a manifold. But even then, they do not use any properties of this manifold!

Here, we derive an optimizer that actually performs steepest descent on the Birkhoff polytope equipped with the spectral norm, using the dual ascent framework from [Ponder: Rethinking Maximal Update Parametrization: Steepest Descent on Finsler-Structured (Matrix) Geometries via Dual Ascent](../steepest-descent-finsler-dual-ascent).

## 2. Method

### 2.1. Constrained first-order optimization on cone geometries

Let $f : \mathcal{M} \to \mathbb{R}$ be a differentiable and bounded-below objective function defined on a normed cone geometry $\mathcal{M}$, or a constraint set $\mathcal{M} \subseteq \mathbb{R}^{m \times n}$ equipped with a norm $\|\cdot\|$ on each tangent set, $T_{W_t}\mathcal{M}$, at each point $W_t \in \mathcal{M}$. First-order optimization on such geometries goes as follows:

1. Let $W_t \in \mathcal{M}$ be the 'weight' parameter at time step $t$. Compute the "raw gradient" $G_t = \nabla f(W_t)$ via e.g. backpropagation.
2. Compute an 'optimal' descent direction $A^*_t \in T_{W_t} \mathcal{M}$ constrained as,
$$\begin{align}
    A^*_t
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} f(W_t) + \langle G_t, A \rangle \quad \text{ s.t. } \quad \| A \| \leq \eta,\quad A \in T_{W_t}\mathcal{M} \nonumber \\
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle G_t, A \rangle \quad \text{ s.t. } \quad \| A \| \leq \eta,\quad A \in T_{W_t}\mathcal{M}, \label{eq:optimaldescent}
\end{align}$$
where $\eta > 0$ is the learning rate hyperparameter.
1. Update the weight in the direction of $A^*_t$ and retract the result back to the manifold via metric projection, $\texttt{retract}_{\mathcal{M}}: \mathbb{R}^{m \times n} \to \mathcal{M}$, $$W_{t+1} \leftarrow \texttt{retract}_{\mathcal{M}}(W_t + A^*_t).$$ 

Note that both constraints on $A$ in Equation $\eqref{eq:optimaldescent}$ are membership constraints to closed convex sets, and so it is simply a convex optimization problem.

### 2.2. Finding the optimal descent direction via dual ascent

Here, we set $\mathcal{M} = \mathcal{B}_n$, the Birkhoff polytope of $n \times n$ doubly stochastic matrices,
$$\begin{align}
    \mathcal{B}_n
        &= \{ W \in \mathbb{R}^{n \times n} \mid W \mathbf{1} = \mathbf{1}, W^\top \mathbf{1} = \mathbf{1}, W \geq 0 \}
\end{align}$$
and $\|\cdot\| = \|\cdot\|_{2 \to 2}$, the spectral norm.

$\blacksquare$ To compute the optimal descent direction $A^*_t$ in Equation $\eqref{eq:optimaldescent}$, we need to derive the tangent spaces/cones $T_{W} \mathcal{B}_n$ first. At internal points $W \in \mathcal{B}_n^{+}$ (where all entries are strictly positive), we take the derivative of the first two equality constraints to get the tangent space,
$$\begin{align}
    T_{W} \mathcal{B}_n^{+}
        &= \{ A \in \mathbb{R}^{n \times n} \mid A \mathbf{1} = \mathbf{0}, A^\top \mathbf{1} = \mathbf{0} \},
\end{align}$$
which are simply the matrices with zero row and column sums. More generally, where $W$ may have some zero entries, making it a boundary point of the Birkhoff polytope, we get the tangent cone,
$$\begin{align}
    T_{W} \mathcal{B}_n
        &= \{ A \in \mathbb{R}^{n \times n} \mid A \mathbf{1} = \mathbf{0}, A^\top \mathbf{1} = \mathbf{0}, A_{ij} \geq 0 \forall (i,j) \text{ such that } W_{ij} = 0 \}
\end{align}$$
Intuitively, if $W_{ij}$ is already $0$, then we can only move "inward" into the polytope along that dimension, i.e., $A_{ij} \geq 0$. Otherwise, we can move in either direction.

Now, we can represent $T_{W} \mathcal{B}_n$ in the standard form discussed in [Ponder: Rethinking Maximal Update Parametrization: Steepest Descent on Finsler-Structured (Matrix) Geometries via Dual Ascent](../steepest-descent-finsler-dual-ascent) as follows:
$$\begin{align}
    T_{W} \mathcal{B}_n
        &= \{ A \in \mathbb{R}^{n \times n} \mid L(A) + b \in -K \}
\end{align}$$
where,
$$\begin{align}
    L(A)
        &:= (A \mathbf{1}, A^\top \mathbf{1}, A \odot M) \nonumber \\
    b
        &:= (\mathbf{0}, \mathbf{0}, \mathbf{0}) \nonumber \\
    K
        &:= \mathbf{0} \times \mathbf{0} \times \mathbb{R}_{-}^{|\{(i,j) \mid W_{ij} = 0\}|}
            \qquad \text{s.t.} \qquad -K = \mathbf{0} \times \mathbf{0} \times \mathbb{R}_{+}^{|\{(i,j) \mid W_{ij} = 0\}|}
            \nonumber \\
    M_{ij}
        &:= \begin{cases}
            1, & W_{ij} = 0 \\
            0, & \text{otherwise}
        \end{cases} \nonumber
\end{align}$$
The dual of the negative orthant cone is itself, and so,
$$\begin{align}
    K^{\dagger}
        &:= \mathbf{0} \times \mathbf{0} \times \mathbb{R}_{-}^{|\{(i,j) \mid W_{ij} = 0\}|} \nonumber \\
\end{align}$$
The projection onto $K^{\dagger}$ and the adjoint operator $L^{\dagger}$ are then given by,
$$\begin{align}
    \text{proj}_{K^{\dagger}}(S_1, S_2, S_3)
        &:= (S_1, S_2, \min(S_3 \odot M, 0)) \nonumber \\
    L^{\dagger}(S_1, S_2, S_3)
        &:= S_1 \mathbf{1}^\top + \mathbf{1} S_2^\top + S_3 \odot M \nonumber
\end{align}$$

And finally, the LMO for the spectral norm is given by,
$$\texttt{LMO}_{\| \cdot \|_{2 \to 2}}(G_t) = -\texttt{msign}(G_t),$$
where $\texttt{msign}(G_t)$ is the matrix sign function, $\texttt{msign}(G_t) = U V^T$ for the SVD $G_t = U \Sigma V^T$.

$\blacksquare$ Taking everything together, our dual ascent update rule becomes,
$$\begin{align}
    A^j
        &= -\eta \cdot \texttt{msign}\left(G_t + S_1 \mathbf{1}^\top + \mathbf{1} S_2^\top + S_3 \odot M \right) \\
    S_1^{j+1}
        &= S_1^j + \sigma \cdot A^j \mathbf{1} \\
    S_2^{j+1}
        &= S_2^j + \sigma \cdot (A^j)^\top \mathbf{1} \\
    S_3^{j+1}
        &= \min\left( \left(S_3^j + \sigma \cdot (A^j \odot M)\right) \odot M, 0 \right)
\end{align}$$
where $\sigma_j > 0$ is the dual ascent learning rate at dual ascent step $j$.

See [Appendix A1](#appendix-a1-jax-implementation-of-the-dual-ascent-optimizer) for implementation in JAX.

### 2.3. Metric projection onto the Birkhoff polytope

Next, we need a retraction map $\texttt{retract}_{\mathcal{B}_n}: \mathbb{R}^{n \times n} \to \mathcal{B}_n$. The Sinkhorn-Knopp operator DeepSeek used is not actually a metric projection, but rather an entropic projection (that minimizes the KL divergence). We instead use Dykstra's algorithm. See [Appendix A2](#appendix-a2-jax-implementation-of-the-metric-projection-onto-the-birkhoff-polytope-via-dykstras-algorithm) for implementation in JAX.

## 3. Results

### 3.1. Our optimizer yields larger effective weight updates vs LMO-based optimizers

![](effective_weight_update_size.png#center)

For a random $W_t \in \mathbb{B}_n$ and $G_t \in \mathbb{R}^{n \times n}$ with $n = 768$, we report the descent magnitude (measured after the retraction step),
$$\begin{equation}
    \text{descent\_magnitude} = \langle G_t, \texttt{retract}_{\mathcal{B}_n}(W_t + A_t^*) - W_t \rangle
\end{equation}$$
of our dual ascent optimizer after varying number of dual ascent steps relative to the LMO baseline (i.e., using only the LMO without considering the tangent cone constraints). We see that our optimizer yields significantly larger effective weight updates across dual ascent steps.

## Acknowledgements

Big thanks to Simo Ryu for productive discussions on the topic. Also see [X thread](https://x.com/cloneofsimo/status/2008120606311788671) for more (public) discussions.

## How to cite

```bibtex
@misc{cesista2026steepestdescentbirkhoff,
  author = {Franz Louis Cesista},
  title = {{S}teepest Descent on the Birkhoff Polytope Equipped with the Spectral Norm},
  year = {2026},
  month = {January},
  day = {4},
  url = {https://leloykun.github.io/ponder/steepest-descent-doubly-stochastic/},
}
```

## References

1. Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang (2025). mHC: Manifold-Constrained Hyper-Connections. URL https://arxiv.org/abs/2512.24880
2. Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou (2025). Hyper-Connections. URL https://arxiv.org/abs/2409.19606

## Appendix

### Appendix A1: JAX implementation of the dual ascent optimizer

See [Ponder: Rethinking Maximal Update Parametrization: Steepest Descent on Finsler-Structured (Matrix) Geometries via Dual Ascent](../steepest-descent-finsler-dual-ascent) for implementation of the `dual_ascent` function.

```python
def dual_ascent(
    G: jax.Array,  # R^(m x n)
    B: Tuple[jax.Array],  # K_dual offset
    L_primal: Callable[[jax.Array], Tuple[jax.Array]],  # R^(m x n) -> K_dual
    L_dual:  Callable[[Tuple[jax.Array]], jax.Array],  # K_dual -> R^(m x n)
    proj_K_dual: Callable[[Tuple[jax.Array]], Tuple[jax.Array]],  # K_dual -> K_dual
    norm_K_dual: Callable[[Tuple[jax.Array]], float],  # K_dual -> R
    lmo: Callable[[jax.Array], jax.Array],  # R^(m x n) -> R^(m x n)
    *,
    max_steps: int=128, sigma: float=1.0,
    rtol: float=1e-3, atol: float=1e-6,
):
    ...

def dual_ascent_doubly_stochastic(
    W: jax.Array, G: jax.Array,
    lmo: Callable[[jax.Array], jax.Array],
    *,
    tol: float=1e-8,
    max_steps: int=128, sigma: float=1.0,
    rtol: float=1e-3, atol: float=1e-6,
):
    n = W.shape[0]
    M = W <= tol

    B           = (0, 0, 0)
    L_primal    = lambda A: (jnp.sum(A, axis=1), jnp.sum(A, axis=0), A * M)
    L_dual      = lambda S: S[0][:,None] + S[1][None,:] + S[2] * M
    proj_K_dual = lambda S: (S[0], S[1], jnp.where(M, jnp.minimum(S[2], 0), jnp.zeros_like(S[2])))
    norm_K_dual = lambda S: jnp.sqrt((jnp.sum(S[0]**2) + jnp.sum(S[1]**2) + jnp.sum((S[2] * M)**2)) / (n**2 + n**2 + jnp.sum(M)))

    return dual_ascent(
        G,
        B,
        L_primal,
        L_dual,
        proj_K_dual,
        norm_K_dual,
        lmo,
        max_steps,
        sigma,
        rtol,
        atol,
    )
```

### Appendix A2: JAX implementation of the metric projection onto the Birkhoff polytope via Dykstra's algorithm

```python
def proj_nonneg(Y: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(Y, 0.0)

def proj_row_sums(Y: jnp.ndarray, target: float = 1.0) -> jnp.ndarray:
    n = Y.shape[1]
    row_sum = jnp.sum(Y, axis=1, keepdims=True)
    return Y - (row_sum - target) / n

def proj_col_sums(Y: jnp.ndarray, target: float = 1.0) -> jnp.ndarray:
    n = Y.shape[0]
    col_sum = jnp.sum(Y, axis=0, keepdims=True)
    return Y - (col_sum - target) / n

def birkhoff_project_dykstra(
    A: jnp.ndarray,
    max_iters: int = 500,
    tol: float = 1e-6,
) -> tuple[jnp.ndarray, dict]:
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Expect square (n,n) matrix."
    A = A.astype(jnp.float32)

    def residual(X):
        rs = jnp.max(jnp.abs(jnp.sum(X, axis=1) - 1.0))
        cs = jnp.max(jnp.abs(jnp.sum(X, axis=0) - 1.0))
        neg = jnp.max(jnp.maximum(-X, 0.0))
        return jnp.maximum(jnp.maximum(rs, cs), neg)

    def cond_fun(state):
        k, X, P1, P2, P3, r = state
        return jnp.logical_and(k < max_iters, r > tol)

    def body_fun(state):
        k, X, P1, P2, P3, _ = state

        Y1 = proj_nonneg(X + P1)
        P1 = (X + P1) - Y1

        Y2 = proj_row_sums(Y1 + P2, target=1.0)
        P2 = (Y1 + P2) - Y2

        Y3 = proj_col_sums(Y2 + P3, target=1.0)
        P3 = (Y2 + P3) - Y3

        Xn = Y3
        rn = residual(Xn)
        return (k + 1, Xn, P1, P2, P3, rn)

    X0 = A
    P1 = jnp.zeros_like(A)
    P2 = jnp.zeros_like(A)
    P3 = jnp.zeros_like(A)
    r0 = residual(X0)

    _, Xf, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, (0, X0, P1, P2, P3, r0))
    return Xf

birkhoff_project_dykstra_jit = jax.jit(birkhoff_project_dykstra, static_argnames=("max_iters", "tol"))
```
