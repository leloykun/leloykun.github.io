---
title: "Rethinking Maximal Update Parametrization: Steepest Descent on the Spectral Ball"
date: 2025-10-15
tags: ["Machine Learning", "Optimizers", "Architecture-Optimizer Codesign"]
author: "Franz Louis Cesista"
description: "Novel optimizers for maximally updating both the weights and activations of neural networks while keeping weight norms under control. To get there, we needed to invent an efficient, GPU/TPU-friendly method for eigenvalue clipping and solve the Steepest Descent problem on the Positive Semidefinite Cone, Convex Spectrahedron, and finally on the Spectral Ball."
summary: "Novel optimizers for maximally updating both the weights and activations of neural networks while keeping weight norms under control. To get there, we needed to invent an efficient, GPU/TPU-friendly method for eigenvalue clipping and solve the Steepest Descent problem on the Positive Semidefinite Cone, Convex Spectrahedron, and finally on the Spectral Ball."
cover:
    image: mup_spectral_ball_spectral.png
    alt: "Cover"
    relative: true
# editPost:
#     URL: "https://x.com/leloykun/status/1936191549735624977"
#     Text: "Crossposted on X (formerly Twitter)"
citation:
    title: "Efficient Eigenvalue Clipping and Steepest Descent on the Positive Semidefinite Cone"
    author:
        - "Franz Louis Cesista"
    publication_date: "2025/10/02"
---

## 1. Introduction

What is "maximal update parametrization" really about? When training neural networks, we want to (1) *maximally* update both the weights and activations (or "features") *while* (2) keeping the 'sizes' or norms of the weights, activations, and gradients under control. The former is so that we can train larger models with fewer resources and the latter is so that our multi-billion dollar training runs would not randomly blow up in our faces during training. Now, bounding only the weights and using only Lipschitz-continuous layers already suffice to control the Lipschitzness of the model; and consequently also controlling the activation and gradient norms. But bounding the weights (e.g. via weight decay) may also discard components of our updates which we could have allocated to more promising directions in the first place. How do we resolve this?

The crux is to consider the geometry of the space where our weights 'live in' and do steepest descent there. We can control the weight norms by choosing a bounded manifold and using its retraction map to keep weight norms in a comfortable range. And to prevent the retraction map from discarding parts of our updates, we can enforce that the updates be in the tangent space/cone at the current point in the manifold. Finally we can also maximally update the activations by equipping the tangent spaces/cones of the manifold with the $\texttt{RMS} \to \texttt{RMS}$ induced operator norm (or the scaled spectral norm) as discussed by [Yang et al. (2024)](https://arxiv.org/abs/2310.17813).

[Thinking Machines Lab recently published a blog post](https://thinkingmachines.ai/blog/modular-manifolds/) following the same idea where they discussed why and how to do steepest descent on the Stiefel manifold equipped with the (scaled) spectral norm. But constraining the weights to be in the Stiefel manifold is too tight of a constraint. For one, this halves the effective parameters of the model. We argue that merely enforcing an upper bound on the singular values (and letting them go to zero during training if need be) would suffice and may even be better as it does not harm model expressivity as much as constraining the singular values to all have the same value.

We call this Steepest Descent on the Spectral Ball, and we shall discuss how to get this done in [Section 5](#5-steepest-descent-on-the-spectral-ball). But on our way to solving this problem, we needed to solve subproblems such as finding an efficient and GPU/TPU-friendly way to clip the eigenvalues of a (symmetric) matrix in [Section 2](#2-eigenvalue-clipping), and numerically stable methods to compute projectors to eigenbases and singular subspaces in [Section 3.2.1](#321-numerically-stable-computation-of-the-null-space-projector), [Section 4.1.1](#411-numerically-stable-computation-of-the-eigenspace-projectors), and [Section 5.1.3](#513-numerically-stable-computation-of-the-singular-subspace-projectors). With the same tools, we also managed to solve the Steepest Descent problem on the Positive Definite Cone in [Section 3](#3-steepest-descent-on-the-psd-cone) and on the Convex Spectrahedron in [Section 4](#4-steepest-descent-on-the-convex-spectrahedron). In [Section 6.1](#61-learning-rate-transfer-xor-problem) we demonstrate that learning rate transfer (and probably also transfer of other hyperparameters) comes naturally when doing steepest descent on such manifolds. And finally, in [Section 6.2](#62-larger-updates-accelerate-generalization-addition-modulo-31-problem) we show that our approach yields in larger updates (even after applying the retraction map) and faster grokking.

## 2. Eigenvalue Clipping

In [Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping via Newton-Schulz Iteration](../spectral-clipping/), we discussed a novel method for clipping singular values of a matrix requiring only matrix multiplications. Following the same technique, we can also clip the *eigenvalues* of a (symmetric) matrix efficiently. This can be used to efficiently project matrices onto the positive semidefinite cone, and for capping the eigenvalues to a comfortable range during training.

> I have previously communicated this technique to the authors of [Factorization-free Orthogonal Projection onto the Positive Semidefinite Cone with Composite Polynomial Filtering](https://arxiv.org/abs/2507.09165) as I mistakenly thought their method for projecting onto the positive semidefinite cone was a special case of [my prior work](../spectral-clipping/). This work, however, *does* generalize their technique. I recommend reading their paper!

Let $W \in \mathbb{S}^{n}$ where $\mathbb{S}^{n} = \{W \in \mathbb{R}^{n \times n} | W = W^T\}$ is the set of all $n \times n$ real symmetric matrices. Symmetric matrices have real eigenvalues and can be diagonalized by an orthogonal matrix. We define Eigenvalue Clipping as follows:

> **Definition 1 (Eigenvalue Clipping)**. Let $W \in \mathbb{S}^{n}$ be a symmetric matrix and $W = Q \Lambda Q^T$ be its eigenvalue decomposition where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ are the eigenvalues of $W$, $\lambda_i \in \mathbb{R}$ for all $i$, and $Q^T Q = I$. Then we define Eigenvalue Clipping as the following matrix function $\texttt{eig\_clip}_{[\lambda_{min}, \lambda_{max}]}: \mathbb{S}^{n} \to \mathbb{S}^{n}$,
> $$\begin{equation}\texttt{eig\_clip}_{[\lambda_{min}, \lambda_{max}]}(W) = Q \texttt{clip}_{[\lambda_{min}, \lambda_{max}]}(\Lambda) Q^T\label{1}\end{equation}$$
> where $\lambda_{min}, \lambda_{max} \in (-\infty, \infty)$ are hyperparameters that control the minimum and maximum attainable eigenvalues of the resulting matrix and $\texttt{clip}_{[\alpha, \beta]}: \mathbb{R} \to \mathbb{R}$ is applied element-wise on the eigenvalues of $W$,
> 
> $$\begin{equation}\texttt{clip}_{[\alpha, \beta]}(x) = \begin{cases}
\alpha & \texttt{if } x < \alpha \\
x & \texttt{if } \alpha \leq x \leq \beta \\
\beta & \texttt{if } \beta < x
\end{cases}\end{equation}$$
> where $\alpha, \beta \in \mathbb{R} \cup \{-\infty, \infty\}$ and $\alpha \leq \beta$.

The naive implementation of this requires computing the eigenvalue decomposition of $W$, which is computationally expensive and requires high numerical precision (typically `float32`). Instead, we make use of the GPU/TPU-friendly method to compute the matrix sign function $\texttt{msign}$ by [Jordan et al. (2024)](https://kellerjordan.github.io/posts/muon/) and the following identity from [the previous blog post](../spectral-clipping/):

> **Proposition 2 (Computing $\texttt{clip}$ via $\texttt{sign}$).** Let $\alpha, \beta \in \mathbb{R} \cup \{-\infty, \infty\}$ and $\texttt{clip}: \mathbb{R} \to \mathbb{R}$ be the clipping function defined in Definition 1. Then,
> $$\begin{equation}\texttt{clip}_{[\alpha, \beta]}(x) = \frac{\alpha + \beta + (\alpha - x)\texttt{sign}(\alpha - x) - (\beta - x)\texttt{sign}(\beta - x)}{2}\label{4}\end{equation}$$

### 2.1 Lifting to matrix form

We can lift Equation 3 to matrix form as follows:

$$\begin{align}
    \texttt{eig\_clip}_{[\alpha, \beta]}(W)
        &= Q \texttt{clip}_{[\alpha, \beta]}(\Lambda) Q^T\nonumber\\
        &= Q \frac{(\alpha + \beta) I + (\alpha I - \Lambda)\texttt{sign}(\alpha I - \Lambda) - (\beta I - \Lambda)\texttt{sign}(\beta I - \Lambda)}{2} Q^T\nonumber\\
        &= \frac{1}{2} [(\alpha + \beta) QQ^T\nonumber\\
            &\qquad+ Q (\alpha I - \Lambda ) \texttt{sign}(\alpha I - \Lambda) Q^T\nonumber\\
            &\qquad- Q (\beta I - \Lambda ) \texttt{sign}(\beta I - \Lambda) Q^T]\nonumber\\
        &= \frac{1}{2} [(\alpha + \beta) I\nonumber\\
            &\qquad+ Q (\alpha I - \Lambda ) (Q^T Q) \texttt{sign}(\alpha I - \Lambda) Q^T\nonumber\\
            &\qquad- Q (\beta I - \Lambda ) (Q^T Q) \texttt{sign}(\beta I - \Lambda) Q^T]\nonumber\\
    \texttt{eig\_clip}_{[\alpha, \beta]}(W)
        &= \frac{1}{2} [(\alpha + \beta) I \nonumber \\
            &\qquad+ (\alpha I - W ) \texttt{msign}(\alpha I - W) \nonumber \\
            &\qquad- (\beta I - W ) \texttt{msign}(\beta I - W)]
\end{align}
$$

which we can implement in JAX as follows:

```python
def sym(W: jax.Array) -> jax.Array:
    return (W + W.T) / 2

def eig_clip(W: jax.Array, alpha: float=-1., beta: float=1.) -> jax.Array:
    assert W.shape[0] == W.shape[1], "W must be square"
    W = sym(W)
    I = jnp.eye(W.shape[0], dtype=W.dtype)
    result = (1/2) * (
        (alpha + beta) * I
        + (alpha * I - W) @ _orthogonalize(alpha * I - W)
        - (beta * I - W) @ _orthogonalize(beta * I - W)
    )
    return sym(result)
```

### 2.2 Eigenvalue ReLU and orthogonal projection onto the positive semidefinite cone

Suppose we want to bound the eigenvalues of $W$ from below by a minimum value $\alpha$. For $\alpha = 0$, this is equivalent to projecting $W$ onto the positive semidefinite cone which is useful in e.g. finance and quantum mechanics where objects are typically required to be positive semidefinite. We can do this by setting $\beta = +\infty$ in Equation 3:

$$\begin{align}
    \texttt{clip}_{[\alpha, \infty]}(x)
        &= \lim_{\beta \to \infty}\frac{\alpha + \beta + (\alpha - x)\texttt{sign}(\alpha - x) - (\beta - x)\texttt{sign}(\beta - x)}{2}\nonumber\\
        &= \frac{\alpha + \cancel{\beta} + (\alpha - x)\texttt{sign}(\alpha - x) - (\cancel{\beta} - x)}{2}\nonumber\\
    \texttt{clip}_{[\alpha, \infty]}(x)
        &= \frac{\alpha + x + (\alpha - x)\texttt{sign}(\alpha - x)}{2}
\end{align}$$

Lifting this to matrix form yields,

$$\begin{align}
    \texttt{eig\_relu}_\alpha(W)
        &= \texttt{eig\_clip}_{[\alpha, \infty]}(W)\nonumber\\
        &= Q \texttt{clip}_{[\alpha, \infty]}(\Lambda) Q^T\nonumber\\
    \texttt{eig\_relu}_\alpha(W)
        &= \frac{1}{2} [\alpha I + W + (\alpha I - W) \texttt{msign}(\alpha I - W)]
\end{align}$$

which we can implement in JAX as follows:

```python
def eig_relu(W: jax.Array, alpha: float=0.) -> jax.Array:
    W = sym(W)
    I = jnp.eye(W.shape[0], dtype=W.dtype)
    result = (1/2) * (alpha * I + W + (alpha * I - W) @ _orthogonalize(alpha * I - W))
    return sym(result)
```

For the orthogonal projection onto the positive semidefinite cone, we set $\alpha = 0$:

$$\begin{align}
    \texttt{proj\_psd}(W)
        &= \texttt{eig\_relu}_0(W) \nonumber \\
        &= \frac{1}{2} [0 + W + (0 - W) \texttt{msign}(0 - W)] \nonumber \\
    \texttt{proj\_psd}(W)
        &= \frac{1}{2} [W + W \texttt{msign}(W)].
\end{align}$$

The last equality follows from $\texttt{msign}(-W) = -\texttt{msign}(W)$. We can then implement this in JAX as follows:

```python
def proj_psd(W: jax.Array) -> jax.Array:
    W = sym(W)
    return sym((1/2) * (W + W @ _orthogonalize(W)))
```

### 2.3 Eigenvalue Hardcapping and orthogonal projection onto the negative semidefinite cone

Suppose we have symmetric matrices $W$ as weights in a neural network and we want to guarantee that the weights do not blow up *during* training. We can do this by capping the eigenvalues of $W$ to a maximum value $\beta$ after each gradient update. To do this, we can set $\alpha = -\infty$ in Equation 3:

$$\begin{align}
    \texttt{clip}_{[-\infty, \beta]}(x)
        &= \lim_{\alpha \to -\infty}\frac{\alpha + \beta + (\alpha - x)\texttt{sign}(\alpha - x) - (\beta - x)\texttt{sign}(\beta - x)}{2}\nonumber\\
        &= \frac{\cancel{\alpha} + \beta - \cancel{\alpha} + x - (\beta - x)\texttt{sign}(\beta - x)}{2}\nonumber\\
    \texttt{clip}_{[-\infty, \beta]}(x)
        &= \frac{\beta + x - (\beta - x)\texttt{sign}(\beta - x)}{2}
\end{align}$$

Lifting this to matrix form yields,

$$\begin{align}
    \texttt{eig\_hardcap}_\beta(W)
        &= \texttt{eig\_clip}_{[-\infty, \beta]}(W) \nonumber\\
        &= Q \texttt{clip}_{[-\infty, \beta]}(\Lambda) Q^T \nonumber\\
    \texttt{eig\_hardcap}_\beta(W)
        &= \frac{1}{2} [\beta I + W - (\beta I - W) \texttt{msign}(\beta I - W)]
\end{align}$$

which we can implement in JAX as follows:

```python
def eig_hardcap(W: jax.Array, beta: float=1.) -> jax.Array:
    W = sym(W)
    I = jnp.eye(W.shape[0], dtype=W.dtype)
    result = (1/2) * (beta * I + W - (beta * I - W) @ _orthogonalize(beta * I - W))
    return sym(result)
```

For the orthogonal projection onto the negative semidefinite cone, we set $\beta = 0$:
$$\begin{align}
    \texttt{proj\_nsd}(W)
        &= \texttt{eig\_hardcap}_0(W) \nonumber \\
        &= \frac{1}{2} [0 + W - (0 - W) \texttt{msign}(0 - W)] \nonumber \\
    \texttt{proj\_nsd}(W)
        &= \frac{1}{2} [W - W \texttt{msign}(W)] \\
\end{align}$$

which we can implement in JAX as follows:

```python
def proj_nsd(W: jax.Array) -> jax.Array:
    W = sym(W)
    return sym((1/2) * (W - W @ _orthogonalize(W)))
```

### 2.4. Stepfun

![](stepfun.png#center)

Stepfun applies the step function on the singular values/eigenvalues of a matrix. As we will discuss in the next sections, this would be useful for e.g. filtering or "picking out" eigenbasis vectors corresponding to eigenvalues in a certain range in a numerically stable way.


[You (2025)](https://x.com/YouJiacheng/status/1930988035195478303) first devised a implementation for the rectangular case requiring only matrix multiplications. But as can be seen in the figure above, when applied to the symmetric matrix case, it (1) also acts symmetrically to the negative eigenvalues which is not what we want, and (2) requires two (expensive) $\texttt{msign}$ calls. But a simple modification fixes both issues,
$$\begin{align}
    \texttt{eig\_stepfun}_{\alpha}(X)
        &= Q \texttt{step}_{\alpha}(\Lambda) Q^T \nonumber \\
        &= Q \frac{I + \texttt{sign}(\Lambda - \alpha I)}{2} Q^T \nonumber \\
        &= \frac{1}{2}[QQ^T + Q \texttt{sign}(\Lambda - \alpha I) Q^T] \nonumber \\
        &= \frac{1}{2}[I + \texttt{msign}(Q(\Lambda - \alpha I) Q^T)] \nonumber \\
    \texttt{eig\_stepfun}_{\alpha}(X)
        &= \frac{1}{2}[I + \texttt{msign}(X - \alpha I)]
\end{align}$$
As can be seen in the figure above, this implementation applies the step function properly and only requires one $\texttt{msign}$ call.

We can implement this in JAX as follows:
```python
def eig_stepfun(X: jax.Array, alpha=0.) -> jax.Array:
    I = jnp.eye(X.shape[0], dtype=X.dtype)
    return (I + _orthogonalize(X - alpha * I)) / 2.
```

## 3. Steepest descent on the PSD cone

### 3.1. Problem setup

Suppose we want to do steepest descent on the PSD cone under a norm $\|\cdot\|$ chosen a priori. That is, we want to do first-order optimization where we constrain our weights to be positive semidefinite and our weight updates to have bounded norm. As we previously discussed in [Heuristic Solutions for Steepest Descent on the Stiefel Manifold](../steepest-descent-stiefel/), we can do this as follows:

1. Let $W_t \in \mathcal{M}$ be the 'weight' parameter at time step $t$. Compute the "raw gradient" $G_t = \nabla f(W_t)$ via e.g. backpropagation.
2. Compute a 'optimal' descent direction $A^* \in T_{W_t} \mathcal{M}$ under the norm in the tangent space at $W_t$, $$\begin{equation} A^* = \arg\min_{A \in \mathbb{R}^{m \times n}} \langle G, A \rangle \quad \text{ s.t. } \quad \| A \|_{W_t} \leq \eta,\quad A \in T_{W_t}\mathcal{M}, \end{equation}$$ where $\eta > 0$ is the learning rate.
3. Update the weight in the direction of $A^*$, $$\widetilde{W}_{t+1} \leftarrow W_t + A^*.$$ Note that $\widetilde{W}_{t+1}$ may not be on the manifold $\mathcal{M}$. And so,
4. Retract the result back to the manifold via a retraction map $W_{t+1} \leftarrow \texttt{retract}_{\mathcal{M}}(\widetilde{W}_{t+1})$.

In our case, the manifold is the PSD cone, $\mathcal{M} := \mathbb{S}^n_{+} = \{W \in \mathbb{S}^n : W \succeq 0\}$. And so, we use the $\texttt{proj\_psd}$ function defined in [Section 2.2](#22-eigenvalue-relu-and-orthogonal-projection-onto-the-positive-semidefinite-cone) as our retraction map.
$$\texttt{retract}_{\mathbb{S}^n_{+}} := \texttt{proj\_psd} = \texttt{eig\_relu}_0.$$

To find an 'optimal' descent direction $A^*$, we can, in some cases, use known Linear Minimization Oracles (LMOs) [(Pethick et al., 2025)](https://arxiv.org/abs/2502.07529). Or, as we discussed in [Steepest Descent on Finsler-Structured (Matrix) Manifolds](../steepest-descent-finsler/), we can compute an 'optimal' descent direction $A^*$ via two orthogonal projection functions: (i) the projection onto the norm ball, $\texttt{proj}_{\| \cdot \|_{W_t} \leq \eta}$, and (ii) the projection onto the tangent space at $W_t$, $\texttt{proj}_{T_{W_t}\mathcal{M}}$.

If we choose the Frobenius norm, then the projection onto the norm ball is simply,
$$\texttt{proj}_{\| \cdot \|_F \leq \eta}(X) := \begin{cases}
    \frac{\eta}{\| X \|_F} X & \text{if } \| X \|_F > \eta \\
    X & \text{otherwise}
\end{cases}$$
Alternatively, we can also choose to do steepest descent under the $2 \to 2$ induced operator norm. As to why we might want to do this, you need to binge-read my previous blog posts. In short, controlling the $2 \to 2$ induced operator norm of our weights allows us to control the Lipschitzness of our model which has been shown to improve robustness, generalization, and training stability. In this case, we can use the eigenvalue clipping function defined in [Section 2.1](#21-lifting-to-matrix-form) to do the projection onto the spectral norm ball,
$$\texttt{proj}_{\| \cdot \|_{2 \to 2} \leq \eta} := \texttt{eig\_clip}_{[-\eta,\eta]}.$$

The tricky part is the projection onto the tangent space/cone at $W_{t} \in \mathbb{S}^n_{+}$, $\texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}}$.

### 3.2. Projection onto the tangent space/cone at a point on the PSD cone

![](eig_null.png#center)

**Special Case:** $W_{t}$ is an interior point of the PSD cone. That is, $W_{t} \in \mathbb{S}^n_{++} \subset \mathbb{S}^n_{+}$ or, equivalently, $W_{t} \succ 0$. Then the tangent space is the entire space of symmetric matrices,
$$T_{W_{t}} \mathbb{S}^n_{++} = \mathbb{S}^n.$$
And the projection onto the tangent space is simply the symmetrization operation $\texttt{sym}(X) = (X + X^T)/2$,
$$\texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{++}} = \texttt{sym}.$$

**General Case:** For any $W_{t} \in \mathbb{S}^n_{+}$, we *may* no longer have a tangent space but rather a tangent *cone*. That is, the vectors in the tangent cone still form a closed space, but if $H \in T_{W_{t}} \mathbb{S}^n_{+}$, then $-H$ may not be in $T_{W_{t}} \mathbb{S}^n_{+}$ ([Rockafellar & Wets, 2009](https://sites.math.washington.edu/~rtr/papers/rtr169-VarAnalysis-RockWets.pdf)). And thus, we need to be careful with the directions of our inputs to the projection map. The tangent cone at $W_{t} \in \mathbb{S}^n_{+}$ is given by,
$$T_{W_{t}} \mathbb{S}^n_{+} = \{ H \in \mathbb{S}^n : \underbrace{U_0^T H U_0 \succeq 0}_{\text{don't go below 0}} \}$$
where $U_0 \in \mathbb{R}^{m \times (n-r)}$ is the orthonormal basis for the null space of $W_{t}$ and $r = \texttt{rank}(W_t)$. Note that if $W_{t}$ is full rank (and therefore positive definite), then $U_0 = 0$ and we recover the special case above.

Let $\widehat{X} := \texttt{sym}(X)$, $U = \begin{bmatrix} U_{r} & U_0 \end{bmatrix}$ be the eigenbasis of $W_t$, and $P_0 = U_0 U_0^T$ be the projector onto the null space of $W_{t}$. The projection onto the tangent cone at $W_{t} \in \mathbb{S}^n_{+}$ is given by,
$$\begin{align}
    \texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}}(X)
        &= \arg\min_{H \in T_{W_{t}}\mathbb{S}^n_{+}} \| H - X \|_F^2 \nonumber \\
        &= \arg\min_{H \in T_{W_{t}}\mathbb{S}^n_{+}} \| H - (\texttt{sym}(X) + \texttt{skew}(X)) \|_F^2 \nonumber \\
        &= \arg\min_{H \in T_{W_{t}}\mathbb{S}^n_{+}} \| H - \texttt{sym}(X) \|_F^2 \nonumber \\
        &\qquad\qquad\qquad\quad- 2\underbrace{\langle \underbrace{H - \texttt{sym}(X)}_{\text{symmetric}}, \texttt{skew}(X) \rangle}_{=0} + \underbrace{\cancel{\| \texttt{skew}(X) \|_F^2}}_{\text{constant}} \nonumber \\
        &= \arg\min_{H \in T_{W_{t}}\mathbb{S}^n_{+}} \| H - \widehat{X} \|_F^2 \nonumber \\
        &= U \left[ \arg\min_{\substack{
            U^T H U \in \mathbb{S}^n \\
            U_0^T H U_0 \succeq 0
        }} \| U^T (H - \widehat{X}) U \|_F^2 \right] U^T \nonumber \\
        &= U \left[ \arg\min_{\substack{
            U^T H U \in \mathbb{S}^n \\
            U_0^T H U_0 \succeq 0
        }} \left\| \begin{bmatrix}
            U_{r}^T (H - \widehat{X}) U_{r} & U_{r}^T (H - \widehat{X}) U_0 \\
            U_0^T (H - \widehat{X}) U_{r} & U_0^T (H - \widehat{X}) U_0
        \end{bmatrix} \right\|_F^2 \right] U^T \nonumber \\
        &= U \begin{bmatrix}
            U_{r}^T \widehat{X} U_{r} & U_{r}^T \widehat{X} U_0 \\
            U_0^T \widehat{X} U_{r} & (U_0^T \widehat{X} U_0)_{+}
        \end{bmatrix} U^T \nonumber \\
        &= U \begin{bmatrix}
            U_{r}^T \widehat{X} U_{r} & U_{r}^T \widehat{X} U_0 \\
            U_0^T \widehat{X} U_{r} & U_0^T \widehat{X} U_0 - (U_0^T \widehat{X} U_0)_{-}
        \end{bmatrix} U^T \nonumber \\
        &= \widehat{X} - U_0 (U_0^T \widehat{X} U_0)_{-} U_0^T \nonumber \\
        &= \widehat{X} - (U_0 U_0^T \widehat{X} U_0 U_0^T)_{-} \nonumber \\
    \texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}}(X)
        &= \widehat{X} - \texttt{proj\_nsd}(P_0 \widehat{X} P_0)
\end{align}$$
where the fifth equality follows from $I = UU^T$ and the orthogonal invariance of the Frobenius norm, and the second-to-last equality is from the similarity-equivariance of matrix functions that acts entrywise on the eigenvalues/singular values.

In words,
> We first symmetrize the input $X$ into $\widehat{X}$ then we subtract the negative eigenvalues of the projection of $\widehat{X}$ onto the null space of $W_{t}$.

#### 3.2.1. Numerically stable computation of the null space projector

Intuitively, to construct the null space projector $P_0$, we can "select" from $Q$ the eigenvectors corresponding to the zero eigenvalues of $W_{t}$  as follows,
$$\begin{align}
    P_0
        &= Q (\mathcal{i}_{(\lambda_i = 0)}(\Lambda)) Q^T && \text{where } \mathcal{i}_{(\lambda_i = 0)}(\lambda_i) = \begin{cases}
            1 & \text{if } \lambda_i = 0 \\
            0 & \text{otherwise}
        \end{cases} \nonumber \\
        &\approx Q (\mathcal{i}_{(-\epsilon < \lambda_i < \epsilon)}(\Lambda)) Q^T && \text{for small } \epsilon > 0 \nonumber \\
        &= Q (\mathcal{i}_{(\lambda_i < \epsilon)}(\Lambda)) Q^T && \text{since } W \text{ is PSD}\nonumber \\
        &= Q (1 - \texttt{step}(\Lambda, \epsilon)) Q^T \nonumber \\
        &= I - \texttt{eig\_stepfun}(W, \epsilon)
\end{align}$$
where the second line is a relaxation to handle numerical precision issues.

Taking everything together yields,
```python
def project_to_tangent_psd(W: jax.Array, X: jax.Array, tol=1e-3) -> jax.Array:
    P0 = jnp.eye(W.shape[0], dtype=W.dtype) - eig_stepfun(W, tol)
    return jax.lax.cond(
        jnp.rint(jnp.trace(P0)) == 0,
        lambda: sym(X),  # W is an interior point, so tangent space is all symmetric matrices
        lambda: sym((X_ := sym(X)) - proj_nsd(P0 @ X_ @ P0)),
    )
```

#### 3.2.2. Sanity check

{{< collapse summary="Show contents of **3.2.2. Sanity check**" openByDefault=false >}}
For $W_{t} \in \mathbb{S}^n_{+}$ and $X \in \mathbb{R}^{n \times n}$ initialized as follows,

```python
dim = 768
nullity = 128
keys = jax.random.split(jax.random.PRNGKey(0), 2)

W = jax.random.normal(keys[0], (dim, dim))
X = jax.random.normal(keys[1], (dim, dim))

W = sym(W @ W.T)
lam, Q = jnp.linalg.eigh(W)
lam = lam.at[:nullity].set(0)
W = Q @ jnp.diag(lam) @ Q.T
```

Let $H := \texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}}(X)$ and $N = X - H$. Then we have,
| property                                                                                                |             value |
| ------------------------------------------------------------------------------------------------------- | ----------------: |
| range of eigenvalues of $P_0 X P_0$                                                                   | $[-15.67, 14.96]$ |
| range of eigenvalues of $P_0 H P_0$                                                                   |   $[0.00, 14.96]$ |
| range of eigenvalues of $P_0 N P_0$                                                                   |  $[-15.67, 0.00]$ |
| alignment of $H$ to $W$ relative to the alignment of $X$: $\langle W, H \rangle / \langle W, X \rangle$ |           $100$\% |
| alignment of $N$ to $W$: $\langle W, N \rangle / \langle W, W \rangle$                                  |             $0$\% |

This shows that $H \in T_{W_{t}}\mathbb{S}^n_{+}$ and $N \in (T_{W_{t}}\mathbb{S}^n_{+})^\perp$ as desired.

{{< /collapse >}}

### 3.3. Update rule for steepest descent on the PSD cone

We now have all the necessary components to do steepest descent under any norm on the PSD cone.

#### 3.3.1. Special case: $W_t$ is an interior point of the PSD cone

As we discussed in the previous section, if $W_t$ is full rank, then the tangent space at that point is simply the space of all symmetric matrices. An interesting coincidence is that Linear Minimization Oracles (LMOs) for common norms derived for the case without the tangency constraint preserve symmetry.

| Norm           |               LMO                | preserves symmetry? |
| :------------- | :------------------------------: | :-----------------: |
| Frobenius norm | $X \to \frac{1}{\| X \|_F} X$ |         Yes         |
| $\| \cdot \|_{2 \to 2}$  |    $X \to \texttt{msign}(X)$     |         Yes         |

Therefore, it would suffice to symmetrize the "raw gradient" $G_t$ first and then apply the LMO. This guarantees that our updates are indeed on-tangent and maximal (via theory behind LMOs). Our update rule would then be,
$$\begin{align}
    W_{t+1}
        &= \texttt{proj\_psd}\left(W_{t} + \texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(\texttt{sym}(-G_t)) \right)
\end{align}$$

#### 3.3.2. General case

In general, LMOs derived for the case without the tangency constraint often 'send' its output off-tangent. An example [we discussed in previous blog post](../steepest-descent-stiefel/) is $\texttt{msign}$ and the Stiefel manifold. In such cases, we can use either of the following two methods to compute an 'optimal' descent direction $A^*$:

1. A *heuristic* solution such as the one discussed in [Heuristic Solutions for Steepest Descent on the Stiefel Manifold](../steepest-descent-stiefel/) where we iteratively apply the projection onto the tangent space and the LMO until convergence. That is,
$$\begin{align}
    W_{t+1}
        &= \texttt{proj\_psd}\left(W_{t} + \left(\texttt{LMO}_{\|\cdot\|_{W_t} \leq \eta} \circ \texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}} \right)^K (-G_t) \right)
\end{align}$$
for some integer $K \geq 1$ denoting the number of iterations (typically, $K = 4$ to $8$ suffices; but the iteration can be terminated upon convergence).
2. An *exact* solution such as the primal-dual hybrid gradient method, $\texttt{pdhg}$, we discussed in [Steepest Descent on Finsler-Structured (Matrix) Manifolds](../steepest-descent-finsler/),
$$\begin{align}
    W_{t+1}
        &= \texttt{proj\_psd}(W_{t} + \texttt{pdhg}(W_t, G_t, \texttt{proj}_{\| \cdot \|_{W_t} \leq \eta}, \texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}}))
\end{align}$$
We can also speed up PDHG by warm-starting the initial iterate $A^0$ with the heuristic above or the solution from the previous time step $A^*_{t-1}$ (in theory, the solutions should not drift too much between time steps if we accumulate the gradients with a momentum rule).

Voilà, we now have an efficient, factorization-free, and GPU/TPU-friendly method for doing steepest descent on the PSD cone under any norm.

## 4. Steepest descent on the Convex Spectrahedron

Suppose we want to constrain our weights to have eigenvalues bounded within some range $[\alpha, \beta] \subseteq \mathbb{R}$. That is, we "place" our weights on the Convex Spectrahedron,
$$\mathcal{K}_{[\alpha, \beta]} := \{W \in \mathbb{S}^n : \alpha I \preceq W \preceq \beta I\},\qquad(\alpha < \beta)$$
and do steepest descent there under some norm chosen a priori. For the retraction map, we can use the eigenvalue clipping function defined in [Section 2.1](#21-lifting-to-matrix-form),
$$\texttt{retract}_{\mathcal{K}_{[\alpha, \beta]}} := \texttt{eig\_clip}_{[\alpha,\beta]}.$$

### 4.1. Projection onto the tangent space/cone at a point on the Convex Spectrahedron

The tangent cone at $W_t \in \mathcal{K}_{[\alpha, \beta]}$ is generally given by,
$$T_{W_t} \mathcal{K}_{[\alpha, \beta]} = \{ H \in \mathbb{S}^n : \underbrace{U_{\alpha}^T H U_{\alpha} \succeq 0}_{\text{don\'t go below } \alpha}, \underbrace{U_{\beta}^T H U_{\beta} \preceq 0}_{\text{don\'t go above } \beta} \}$$
where $U_{\alpha}$ and $U_{\beta}$ are the orthonormal bases for the $\alpha$- and $\beta$-eigenspaces of $W_t$, respectively. If $W_t$ is an interior point, that is, $\alpha I \prec W_t \prec \beta I$, then $U_\alpha = U_\beta = 0$ and the tangent space is simply the space of symmetric matrices, $T_{W_t} \mathcal{K}_{(\alpha, \beta)} = \mathbb{S}^n$.

Let $\widehat{X} := \texttt{sym}(X)$, $U := \begin{bmatrix} U_{\beta} & U_{\widetilde{r}} & U_{\alpha} \end{bmatrix}$ be the eigenbasis of $W_{t}$, and $P_\alpha := U_{\alpha}U_{\alpha}^T, P_\beta := U_{\beta}U_{\beta}^T$ be the projectors onto the $\alpha$- and $\beta$-eigenspaces of $W_t$, respectively. Then, following the strategy we discussed in [Section 3.2](#32-projection-onto-the-tangent-spacecone-at-a-point-on-the-psd-cone), the projection onto the tangent cone at $W_t \in \mathcal{K}_{[\alpha, \beta]}$ is given by,
$$\begin{align}
    \texttt{proj}_{T_{W_t}\mathcal{K}_{[\alpha, \beta]}}(X)
        &= \arg\min_{H \in T_{W_{t}}\mathcal{K}_{[\alpha, \beta]}} \| H - X \|_F^2 \nonumber \\
        &= \arg\min_{H \in T_{W_{t}}\mathcal{K}_{[\alpha, \beta]}} \| H - \widehat{X} \|_F^2 + \cancel{\text{constant}} \nonumber \\
        &= U \left[ \arg\min_{\substack{
            U^T H U \in \mathbb{S}^n \\
            U_{\alpha}^T H U_{\alpha} \succeq 0 \\
            U_{\beta}^T H U_{\beta} \preceq 0
        }} \| U^T (H - \widehat{X}) U \|_F^2 \right] U^T \nonumber \\
        &= U \begin{bmatrix}
            (U_{\beta}^T \widehat{X} U_{\beta})_{-}  & U_{\beta}^T \widehat{X} U_{\widetilde{r}}  & U_{\beta}^T \widehat{X} U_{\alpha} \\
            U_{\widetilde{r}}^T \widehat{X} U_{\beta}      & U_{\widetilde{r}}^T \widehat{X} U_{\widetilde{r}}      & U_{\widetilde{r}}^T \widehat{X} U_{\alpha} \\
            U_{\alpha}^T \widehat{X} U_{\beta} & U_{\alpha}^T \widehat{X} U_{\widetilde{r}} & (U_{\alpha}^T \widehat{X} U_{\alpha})_{+}
        \end{bmatrix} U^T \nonumber \\
        &= U \begin{bmatrix}
            U_{\beta}^T \widehat{X} U_{\beta} - (U_{\beta}^T \widehat{X} U_{\beta})_{+}  & U_{\beta}^T \widehat{X} U_{\widetilde{r}}  & U_{\beta}^T \widehat{X} U_{\alpha} \\
            U_{\widetilde{r}}^T \widehat{X} U_{\beta}      & U_{\widetilde{r}}^T \widehat{X} U_{\widetilde{r}}      & U_{\widetilde{r}}^T \widehat{X} U_{\alpha} \\
            U_{\alpha}^T \widehat{X} U_{\beta} & U_{\alpha}^T \widehat{X} U_{\widetilde{r}} & U_{\alpha}^T \widehat{X} U_{\alpha} - (U_{\alpha}^T \widehat{X} U_{\alpha})_{-}
        \end{bmatrix} U^T \nonumber \\
        &= \widehat{X} - U_{\alpha} (U_{\alpha}^T \widehat{X} U_{\alpha})_{-} U_{\alpha}^T - U_{\beta} (U_{\beta}^T \widehat{X} U_{\beta})_{+} U_{\beta}^T \nonumber \\
    \texttt{proj}_{T_{W_t}\mathcal{K}_{[\alpha, \beta]}}(X)
        &= \widehat{X} - \texttt{proj\_nsd}(P_\alpha \widehat{X} P_\alpha) - \texttt{proj\_psd}(P_\beta \widehat{X} P_\beta) \\
\end{align}$$
or in words,
> We first symmetrize the input $X$ into $\widehat{X}$ and then we subtract the negative eigenvalues of the projection of $\widehat{X}$ onto the $\alpha$-eigenspace of $W_t$ and the positive eigenvalues of the projection of $\widehat{X}$ onto the $\beta$-eigenspace of $W_t$.

#### 4.1.1. Numerically stable computation of the eigenspace projectors

As in [Section 3.2.1](#321-numerically-stable-computation-of-the-null-space-projector), we can construct the eigenspace projectors $P_\alpha$ and $P_\beta$ as follows,
$$\begin{align}
    P_\alpha
        &= Q (\mathcal{i}_{(\lambda_i = \alpha)}(\Lambda)) Q^T \nonumber \\
        &\approx Q (\mathcal{i}_{(\alpha - \epsilon < \lambda_i < \alpha + \epsilon)}(\Lambda)) Q^T && \text{for small } \epsilon > 0 \nonumber \\
        &= Q (\mathcal{i}_{(\lambda_i < \alpha + \epsilon)}(\Lambda)) Q^T && \text{since } \alpha I \preceq W \nonumber \\
        &= I - \texttt{eig\_stepfun}(W, \alpha + \epsilon)
\end{align}$$
Likewise, $P_\beta \approx \texttt{eig\_stepfun}(W, \beta - \epsilon)$ for small $\epsilon > 0$.

Taking everything together yields,
```python
def project_to_tangent_convex_spectrahedron(W: jax.Array, X: jax.Array, alpha: float, beta: float, tol=1e-3):
    P_alpha, P_beta = jnp.eye(W.shape[0], dtype=W.dtype) - eig_stepfun(W, alpha+tol), eig_stepfun(W, beta-tol)
    return jax.lax.cond(
        jnp.logical_and(jnp.rint(jnp.trace(P_alpha)) == 0, jnp.rint(jnp.trace(P_beta)) == 0),
        lambda: sym(X),  # W is in the interior, so tangent space is all symmetric matrices
        lambda: sym((X_ := sym(X)) - proj_nsd(P_alpha @ X_ @ P_alpha) - proj_psd(P_beta @ X_ @ P_beta)),
    )
```

### 4.2. Update rule for steepest descent on the Convex Spectrahedron

#### 4.2.1. Special case: $W_t$ is an interior point of the Convex Spectrahedron

If $W_t$ is an interior point of the Convex Spectrahedron $\mathcal{K}_{[\alpha, \beta]}$ (that is, $\alpha I \prec W_t \prec \beta I$), then the tangent space at that point is simply the space of all symmetric matrices. Thus, as in the [Section 3.3.1](#331-special-case--is-an-interior-point-of-the-psd-cone), we can use known LMOs that preserve symmetry. Our update rule would then be,
$$\begin{align}
    W_{t+1}
        &= \texttt{eig\_clip}_{[\alpha,\beta]}\left(W_{t} + \texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(\texttt{sym}(-G_t)) \right)
\end{align}$$

#### 4.2.2. General case

In general, we can use either the heuristic or the PDHG method discussed in [Section 3.3.2](#332-general-case),

$$\begin{align}
    W_{t+1} &= \texttt{eig\_clip}_{[\alpha,\beta]}\left(W_{t} + \left(\texttt{LMO}_{\|\cdot\|_{W_t} \leq \eta} \circ \texttt{proj}_{T_{W_t}\mathcal{K}_{[\alpha, \beta]}} \right)^K (-G_t) \right)
\end{align}$$
or,
$$\begin{align}
    W_{t+1} &= \texttt{eig\_clip}_{[\alpha,\beta]}(W_{t} + \texttt{pdhg}(W_t, G_t, \texttt{proj}_{\| \cdot \|_{W_t} \leq \eta}, \texttt{proj}_{T_{W_t}\mathcal{K}_{[\alpha, \beta]}}))
\end{align}$$

## 5. Steepest descent on the Spectral Ball

The previous examples are arguably contrived. This example is more practical.

Suppose we no longer constrain our weights to be symmetric, but we still want to bound their spectral norm. That is, we want to do steepest descent on the Spectral Ball,
$$\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R} := \{W \in \mathbb{R}^{m \times n} : \| W \|_{2 \to 2} \leq R\},$$
for some radius $R > 0$. For the retraction map, we can use the GPU/TPU-friendly Spectral Hardcap function discussed in [Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping via Newton-Schulz Iteration](../spectral-clipping/),
$$\texttt{retract}_{\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}} := \texttt{spectral\_hardcap}_{R}.$$

In [Appendix A1](#a1-steepest-descent-on-the-spectral-band), we generalize this to steepest descent on the Spectral Band where we bound the singular values within some range $[\alpha, \beta]$ to prevent weights from blowing up or vanishing.

### 5.1. Projection onto the tangent space/cone at a point on the Spectral Ball

#### 5.1.1. Shortcut via dilation (slower)

The crux is to observe that the singular values of $W_t \in \mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}$ are $\pm$ the eigenvalues of the block matrix,
$$\widetilde{W_t} := \Phi(W_t) = \begin{bmatrix}
    0 & W_t \\
    W_t^T & 0
\end{bmatrix} \in \mathcal{K}_{[-R, R]},$$
where the mapping $\Phi: \mathbb{R}^{m \times n} \to \mathbb{S}^{m+n}$ is an isometry (up to scaling by $\sqrt{2}$) and therefore we can recover the projection via $[\cdot]_{12}$. This allows us to compute the projection onto the tangent cone at $W_t \in \mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}$ via the projection onto the tangent cone at $\widetilde{W_t} \in \mathcal{K}_{[-R, R]}$,
$$\texttt{proj}_{T_{W_t}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}}(X) = \left[ \texttt{proj}_{T_{\Phi(W_t)}\mathcal{K}_{[-R, R]}}\left(\Phi(X)\right)\right]_{12}$$
which we can implement in JAX as follows:
```python
def project_to_tangent_spectral_ball(W: jax.Array, X: jax.Array, R: float, tol=1e-3) -> jax.Array:
    m, n = W.shape
    phi = lambda A: jnp.block([[jnp.zeros((m, m), dtype=A.dtype), A], [A.T, jnp.zeros((n, n), dtype=A.dtype)]])
    return jax.lax.cond(
        _power_iterate(W) < R,  # or jnp.linalg.norm(W, ord=2) < R
        lambda: X,  # W is an interior point, so tangent space is all matrices
        lambda: project_to_tangent_convex_spectrahedron(phi(W), phi(X), -R, R, tol)[:m, m:],
    )
```

#### 5.1.2. Direct approach (faster)

Similar to the previous sections, the tangent cone at $W_t \in \mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}$ is generally given by,
$$T_{W_t} \mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R} = \{ H \in \mathbb{R}^{m \times n} : \underbrace{\texttt{sym}(U_R^T H V_R) \preceq 0}_{\text{don't go above } R} \}$$
where $U_R \in \mathbb{R}^{m \times k}$ and $V_R \in \mathbb{R}^{n \times k}$ are the orthonormal bases for the left and right $R$-singular subspaces of $W_t$ (that is, the singular vectors corresponding to the singular values equal to $R$), respectively, and $k$ is the multiplicity of the singular value $R$. Note that if $W_{t}$ is an interior point, that is, $\| W_t \|_{2 \to 2} < R$, then $U_R = V_R = 0$ and the tangent space is simply the entire space of matrices, $T_{W_t} \mathcal{B}_{\|\cdot\|_{2 \to 2} < R} = \mathbb{R}^{m \times n}$.

Let $U := \begin{bmatrix} U_{< R} & U_R \end{bmatrix}$ and $V := \begin{bmatrix} V_{< R} & V_R \end{bmatrix}$ be the left and right singular bases of $W_{t}$, respectively. Following our strategy in the previous sections then yields the projection onto the tangent cone at $W_t \in \mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}$,
$$\begin{align}
    \texttt{proj}_{T_{W_t}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}}(X)
        &= \arg\min_{H \in T_{W_{t}}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}} \| H - X \|_F^2 \nonumber \\
        &= U \left[ \arg\min_{\substack{
                U^T H V \in \mathbb{R}^{m \times n} \\
                \texttt{sym}(U_{R}^T H V_{R}) \preceq 0
            }} \| U^T (H - X) V \|_F^2 \right] V^T \nonumber \\
        &= U \left[ \arg\min_{\substack{
                U^T H V \in \mathbb{R}^{m \times n} \\
                \texttt{sym}(U_{R}^T H V_{R}) \preceq 0
            }} \left\| \begin{bmatrix}
            U_{< R}^T (H - X) V_{< R} & U_{< R}^T (H - X) V_{R} \\
            U_{R}^T (H - X) V_{< R}      & U_{R}^T (H - X) V_{R}
        \end{bmatrix} \right\|_F^2 \right] V^T \nonumber \\
        &= U \begin{bmatrix}
            U_{< R}^T X V_{< R} & U_{< R}^T X V_{R} \\
            U_{R}^T X V_{< R}      & U_{R}^T X V_{R} - (\texttt{sym}(U_{R}^T X V_{R}))_{+}
        \end{bmatrix} V^T \nonumber \\
        &= X - U_R (\texttt{sym}(U_{R}^T X V_{R}))_{+} V_R^T \nonumber \\
        &= X - U_R \underbrace{(V_R^T V_R)}_{I} (\texttt{sym}(U_{R}^T X V_{R}))_{+} V_R^T \nonumber \\
        &= X - (U_R V_R^T) (\texttt{sym}(V_R U_R^T X V_{R} V_R^T))_{+} \nonumber \\
        &= X - J_{R} (\texttt{sym}(J_{R}^T X P_{V_{R}}))_{+} \nonumber \\
    \texttt{proj}_{T_{W_t}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}}(X)
        &= X - J_R \texttt{proj\_psd}(\texttt{sym}(J_{R}^T X P_{V_{R}}))
\end{align}$$
where $P_{V_{R}} := V_{R} V_{R}^T$ is the projector onto the right $R$-singular subspace of $W_t$, and $J_R := U_R V_R^T$ is the partial isometry corresponding to the $R$-singular subspace of $W_t$. The fourth equality comes from,
$$\begin{align}
    &\arg\min_{\substack{
        U_R^T H V_R \in \mathbb{R}^{m \times n} \\
        \texttt{sym}(U_{R}^T H V_{R}) \preceq 0
    }} U_{R}^T (H - X) V_{R} \nonumber \\
        &\qquad\qquad\qquad= \arg\min_{\substack{
            U_R^T H V_R \in \mathbb{R}^{m \times n} \\
            \texttt{sym}(U_{R}^T H V_{R}) \preceq 0
        }} [U_{R}^T H V_{R} - (\texttt{skew}(U_{R}^T X V_{R}) + \texttt{sym}(U_{R}^T X V_{R}))]  \nonumber \\
        &\qquad\qquad\qquad= \texttt{skew}(U_{R}^T X V_{R}) + \arg\min_{\substack{
            U_R^T H V_R \in \mathbb{R}^{m \times n} \\
            \texttt{sym}(U_{R}^T H V_{R}) \preceq 0
        }} [U_{R}^T H V_{R} - \texttt{sym}(U_{R}^T X V_{R})] \nonumber \\
        &\qquad\qquad\qquad= \texttt{skew}(U_{R}^T X V_{R}) + (\texttt{sym}(U_{R}^T X V_{R}))_{-} \nonumber \\
        &\qquad\qquad\qquad= \texttt{skew}(U_{R}^T X V_{R}) + (\texttt{sym}(U_{R}^T X V_{R}) - (\texttt{sym}(U_{R}^T X V_{R}))_{+}) \nonumber \\
        &\qquad\qquad\qquad= U_{R}^T X V_{R} - (\texttt{sym}(U_{R}^T X V_{R}))_{+} \nonumber
\end{align}$$

#### 5.1.3. Numerically stable computation of the singular subspace projectors

First, note that for $W = U \Sigma V^T$, we have $W_t^T W_t = V \Sigma^2 V^T$. Thus,
$$\begin{align}
    P_{V_{R}}
        &= V_{R} V_{R}^T \nonumber \\
        &= V (\mathcal{i}_{(\lambda_i = R^2)}(\Sigma^2)) V^T && \text{where } \lambda_i = [\Sigma^2]_i = \sigma_i^2 \nonumber \\
        &\approx V (\mathcal{i}_{(R^2 - \epsilon < \lambda_i < R^2 + \epsilon)}(\Sigma^2)) V^T && \text{for small } \epsilon > 0 \nonumber \\
        &= V (\mathcal{i}_{(\lambda_i > R^2 - \epsilon)}(\Sigma^2)) V^T && \text{since } \| W \|_{2 \to 2} \leq R \nonumber \\
        &= \texttt{eig\_stepfun}(V \Sigma^2 V^T, R^2 - \epsilon) \nonumber \\
        &= \texttt{eig\_stepfun}(W_t^T W_t, R^2 - \epsilon).
\end{align}$$
And,
$$\begin{align}
    J_R
        &= U_R V_R^T \nonumber \\
        &= U (\mathcal{i}_{(\lambda_i = R)}(\Sigma)) V^T \nonumber \\
        &= U \left( \begin{cases}
            \frac{\sigma_i}{R} 1 & \text{if } \sigma_i = R \\
            0 & \text{otherwise}
        \end{cases} \right) V^T && \text{i.e., } \mathcal{i}_{(\lambda_i = R)}(\Sigma) = \frac{\Sigma}{R}\cdot\mathcal{i}_{(\lambda_i = R)}(\Sigma)\nonumber \\
        &= U \frac{1}{R}\Sigma(V^T V) (\mathcal{i}_{(\lambda_i = R)}(\Sigma)) V^T \nonumber \\
        &= \frac{1}{R} W_t P_{V_{R}}
\end{align}$$

Taking everything together yields,
```python
def project_to_tangent_spectral_ball(W: jax.Array, X: jax.Array, R: float, tol=1e-3) -> jax.Array:
    return jax.lax.cond(
        _power_iterate(W) < R - tol,  # or jnp.linalg.norm(W, ord=2) < R - tol
        lambda: X,  # W is an interior point, so tangent space is all matrices
        lambda: X - (J_R := (1./R) * W @ (PV_R := eig_stepfun(W.T @ W / R**2, 1.-tol))) @ proj_psd(sym(J_R.T @ X @ PV_R)),
    )
```

### 5.2. Update rule for steepest descent on the Spectral Ball

#### 5.2.1. Special case: $W_t$ is an interior point of the Spectral Ball

If $W_t$ is inside the Spectral Ball, then the tangent space at that point is $\mathbb{R}^{m \times n}$ and thus the projection is simply the identity map. We can use the LMO for any norm without worry,

$$\begin{align}
    W_{t+1}
        &= \texttt{spectral\_hardcap}_{R}\left(W_{t} + \texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(-G_t) \right)
\end{align}$$

#### 5.2.2. General case

In general, we can use either the heuristic or the PDHG method discussed in [Section 3.3.2](#332-general-case),
$$\begin{align}
    W_{t+1} &= \texttt{spectral\_hardcap}_{R}\left(W_{t} + \left(\texttt{LMO}_{\|\cdot\|_{W_t} \leq \eta} \circ \texttt{proj}_{T_{W_t}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}} \right)^K (-G_t) \right)
\end{align}$$
or,
$$\begin{align}
    W_{t+1} &= \texttt{spectral\_hardcap}_{R}(W_{t} + \texttt{pdhg}(W_t, G_t, \texttt{proj}_{\| \cdot \|_{W_t} \leq \eta}, \texttt{proj}_{T_{W_t}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq R}}))
\end{align}$$

## 6. Experiments

In all of our experiments below, we constrain weight updates to have $\texttt{RMS}\to\texttt{RMS}$ induced operator norm $\leq \eta$, where $\eta > 0$ is the learning rate. We then vary the manifold to do steepest descent on, and the dualization strategy to compute the optimal update directions on the respective manifolds. In summary, we use the following maps,

| Manifold             | retraction map                                      | dualization map (interior)          | dualization map (boundary), PDHG                                                                                                                                                           |
| :------------------- | :-------------------------------------------------- | :---------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PSD Cone             | $\texttt{proj\_psd}$                               | $\texttt{msign} \circ \texttt{sym}$ | $\texttt{pdhg}\left(\cdots, \texttt{eig\_clip}_{[-\eta, \eta]}, \texttt{proj}_{T_{W_{t}}\mathbb{S}^n_{+}}\right)$                                                              |
| Convex Spectrahedron | $\texttt{eig\_clip}_{[-1,1]}$                     | $\texttt{msign} \circ \texttt{sym}$ | $\texttt{pdhg}\left(\cdots, \texttt{eig\_clip}_{[-\eta, \eta]}, \texttt{proj}_{T_{W_{t}}\mathcal{K}_{[-1,1]}}\right)$                                                          |
| Spectral Ball        | $\texttt{spectral\_hardcap}_{\sqrt{\frac{m}{n}}}$ | $\texttt{msign}$                    | $\texttt{pdhg}\left(\cdots, \texttt{spectral\_hardcap}_{\eta\sqrt{\frac{m}{n}}}, \texttt{proj}_{T_{W_{t}}\mathcal{B}_{\|\cdot\|_{2 \to 2} \leq \sqrt{\frac{m}{n}}}}\right)$ |

### 6.1. Learning rate transfer, XOR problem

As a minimal example for learning rate transfer, we train a $2 \to D \to D \to 2$ MLP on the XOR problem for 32 training steps via the [Modula](https://docs.modula.systems/) library.  We then vary the constraint set for the weights and use the PDHG algorithm to compute the optimal update directions on the respective constraint sets.

We also warm-start the initial iterate of PDHG with the 1 step of the Alternating Projections heuristic (which also solves the case when the weight $W_t$ is an interior point).

#### 6.1.1. Positive semidefinite cone

![](lr_transfer_pdhg_psd_spectral.png#center)

We only constrain the weight for the $D \to D$ linear layer to be positive semidefinite, and the other layers are constrained to the (scaled) Stiefel manifold as discussed in [Steepest Descent on Finsler-Structured (Matrix) Manifolds](../steepest-descent-finsler/). As can be seen in the Figure above, the optimal learning rates do transfer under our parametrization. However, we *cannot* impose a Lipschitz bound on the network because the positive semidefinite cone is unbounded.

#### 6.1.2. Steepest descent on the Convex Spectrahedron

![](lr_transfer_pdhg_spectrahedron_spectral.png#center)

Like above, we only constrain the weight for the $D \to D$ linear layer to be in the Convex Spectrahedron, and the other layers are constrained to the (scaled) Stiefel manifold. But here, we *can* impose a Lipschitz bound on the network because the Convex Spectrahedron is bounded. As can be seen in the Figure above, the optimal learning rates do transfer under our parametrization.

#### 6.1.3. Steepest descent on the Spectral Ball

![](lr_transfer_pdhg_spectral_ball_spectral.png#center)

Unlike the previous two experiments, we now constrain *all* the weights to be in the unit RMS-RMS ball $\mathcal{B}_{\|\cdot\|_{\texttt{RMS} \to \texttt{RMS}} \leq 1} = \mathcal{B}_{\|\cdot\|_{2 \to 2} \leq \sqrt{m/n}}$. As can be seen in the Figure above, the optimal learning rates do transfer under our parametrization.

### 6.2. Larger updates accelerate generalization, Addition-Modulo-31 problem

![](mup_spectral_ball_spectral.png#center)

> Q: Does having larger updates actually help generalization or merely introduce noise?

Preliminary results show that it is indeed the former.

---

Here we train an MLP on the Addition-Modulo-31 problem while constraining the weights to be in the $\texttt{RMS}\to\texttt{RMS}$ ball of radius $R = 4$. We use the retraction map discussed in [Section 5](#5-steepest-descent-on-the-spectral-ball) to keep the weights bounded. With the Muon optimizer, an equivalent weight decay for such a constraint would be $\lambda = 1/R = 0.25$ ([Chen et al., 2025](https://arxiv.org/abs/2506.15054); [Pethick et al., 2025](https://arxiv.org/abs/2502.07529); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)), which is too large and discards too much information from the updates at each step. See [Appendix A2](#a2-weight-decay-as-a-manifold-constraint) for more details. The problem also only has $961$ data points in total. In all, every update step in this setting matters. Noise that gets added into the updates and any information lost from the weight controls become immediately obvious in the generalization performance.

Aside from test accuracy, we also measure the weight delta between steps,
$$\| W_{t+1} - W_t \|_F = \| \texttt{retract}(W_t + A_t^*) - W_t \|_F.$$

As can be seen in the Figure above, our dualizers result in around $2\times$ larger weight deltas compared to baseline. This is because the updates our dualizers produce are in the tangent cones and so the retraction map mostly leaves them intact. In contrast, the baseline updates often have (rather large) components that get discarded by the retraction map, reducing the effective learning rate of the update.

Also notice that the 1-step Alternating Projections heuristic (projecting the "raw gradient" to the tangent cone before applying the LMO) results in similarly-sized weight deltas and also groks the problem in roughly the same number of steps as our PDHG dualizer. This suggests that few-step Alternating Projections may already suffice in practice for larger training runs.

### 6.3. NanoGPT-scale experiments [Under Construction]

## How to Cite

```bibtex
@misc{cesista2025rethinkingmupspectralball,
  author = {Franz Louis Cesista},
  title = {Rethinking Maximal Update Parametrization: Steepest Descent on the Spectral Ball},
  year = {2025},
  month = {October},
  day = {15},
  url = {https://leloykun.github.io/ponder/rethinking-mup-spectral-ball/},
}
```

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

## References

1. Greg Yang, James B. Simon, Jeremy Bernstein (2024). A Spectral Condition for Feature Learning. URL https://arxiv.org/abs/2310.17813
2. Jeremy Bernstein (2025). Modular Manifolds. URL https://thinkingmachines.ai/blog/modular-manifolds/
3. Franz Cesista (2025). Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping via Newton-Schulz Iteration. URL https://leloykun.github.io/ponder/spectral-clipping/
4. Shucheng Kang, Haoyu Han, Antoine Groudiev, Heng Yang (2025). Factorization-free Orthogonal Projection onto the Positive Semidefinite Cone with Composite Polynomial Filtering. URL https://arxiv.org/abs/2507.09165
5. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
6. You Jiacheng (2025). X post on Stepfun. URL https://x.com/YouJiacheng/status/1930988035195478303
7. Franz Cesista (2025). Heuristic Solutions for Steepest Descent on the Stiefel Manifold. URL https://leloykun.github.io/ponder/steepest-descent-stiefel/
8. Franz Cesista (2025). Steepest Descent on Finsler-Structured (Matrix) Manifolds. URL https://leloykun.github.io/ponder/steepest-descent-finsler/
9. R. Tyrrell Rockafellar, Roger J-B Wets (2009). Variational Analysis. URL https://sites.math.washington.edu/~rtr/papers/rtr169-VarAnalysis-RockWets.pdf
10. Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher (2025). Training Deep Learning Models with Norm-Constrained LMOs. URL https://arxiv.org/abs/2502.07529
11. Jeremy Bernstein (2025). The Modula Docs. URL https://docs.modula.systems/
12. Lizhang Chen, Jonathan Li, Qiang Liu (2025). Muon Optimizes Under Spectral Norm Constraints. URL https://arxiv.org/abs/2506.15054
13. Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, Zhilin Yang (2025). Muon is Scalable for LLM Training. URL https://arxiv.org/abs/2502.16982
14. Ben Keigwin, Dhruv Pai, Nathan Chen (2025). Gram-Space Manifold Muon. URL https://www.tilderesearch.com/vignettes/gram-space
15. Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Anima Anandkumar (2018). signSGD: Compressed Optimisation for Non-Convex Problems. URL https://arxiv.org/abs/1802.04434
16. Fabian Schaipp (2024). How to jointly tune learning rate and weight decay for AdamW. URL https://fabian-sp.github.io/posts/2024/02/decoupling/
17. Atli Kosson, Jeremy Welborn, Yang Liu, Martin Jaggi, Xi Chen (2025). Weight Decay may matter more than muP for Learning Rate Transfer in Practice. URL https://arxiv.org/abs/2510.19093
18. Yuandong Tian (2025). Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking. URL https://arxiv.org/abs/2509.21519

---

# Appendices

## A1. Steepest Descent on the Spectral Band

In [Section 5](#5-steepest-descent-on-the-spectral-ball), we discussed steepest descent on the manifold of matrices with bounded-above spectral norm. This prevents weight norms from blowing up, but does not prevent them from collapsing to zero and thus stalling learning anyway. This naturally leads us to steepest descent on the Spectral Band, or the manifold of matrices with bounded-above *and* bounded-below singular values,
$$\mathcal{S}_{[\alpha, \beta]} := \{W \in \mathbb{R}^{m \times n} : \alpha \leq \sigma_{\text{min}}(W) \leq \sigma_{\text{max}}(W) \leq \beta\},$$
for some inner and outer radii $0 \leq \alpha \leq \beta$.

For the retraction map, we can use the GPU/TPU-friendly Spectral Clip function discussed in [Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping via Newton-Schulz Iteration](../spectral-clipping/),
$$\texttt{retract}_{\mathcal{S}_{[\alpha, \beta]}} := \texttt{spectral\_clip}_{[\alpha, \beta]}.$$

### A1.1. Projection onto the tangent space/cone at a point on the Spectral Band

Following [Section 5.1.2](#512-direct-approach-faster), the tangent cone at a point $W_t \in \mathcal{S}_{[\alpha, \beta]}$ is generally given by,
$$T_{W_t} \mathcal{S}_{[\alpha, \beta]} = \{ H \in \mathbb{R}^{m \times n} : \underbrace{\texttt{sym}(U_{\alpha}^T H V_{\alpha}) \succeq 0}_{\text{don't go below } \alpha}, \underbrace{\texttt{sym}(U_{\beta}^T H V_{\beta}) \preceq 0}_{\text{don't go above } \beta} \}$$
where $U_{\alpha}$ and $V_{\alpha}$ are the orthonormal bases for the left and right $\alpha$-singular subspaces of $W_t$ (that is, the singular vectors corresponding to the singular values equal to $\alpha$), respectively. Likewise for $U_{\beta}$ and $V_{\beta}$ with respect to the singular value $\beta$. And if $W_t$ is an interior point, that is, $\alpha < \| W_t \|_{2 \to 2} < \beta$, then $U_{\alpha} = V_{\alpha} = U_{\beta} = V_{\beta} = 0$ and the tangent space is simply the entire space of matrices, $T_{W_t} \mathcal{B}_{\alpha < \| \cdot \|_{2 \to 2} < \beta} = \mathbb{R}^{m \times n}$.

As before, let $U := \begin{bmatrix} U_\alpha & U_{\tilde{r}} & U_\beta \end{bmatrix}$ and $V := \begin{bmatrix} V_\alpha & V_{\tilde{r}} & V_\beta \end{bmatrix}$ be the left and right singular bases of $W_{t}$, respectively. Then the projection is,
$$\begin{align}
    &\texttt{proj}_{T_{W_t}\mathcal{S}_{[\alpha, \beta]}}(X) \nonumber \\
        &\qquad= \arg\min_{H \in T_{W_{t}}\mathcal{S}_{[\alpha, \beta]}} \| H - X \|_F^2 \nonumber \\
        &\qquad= U \left[ \arg\min_{\substack{
            U^T H V \in \mathbb{R}^{m \times n} \\
            \texttt{sym}(U_{\alpha}^T H V_{\alpha}) \succeq 0 \\
            \texttt{sym}(U_{\beta}^T H V_{\beta}) \preceq 0
        }} \| U^T (H - X) V \|_F^2 \right] V^T \nonumber \\
        &\qquad= U \left[ \arg\min_{\substack{
            U^T H V \in \mathbb{R}^{m \times n} \\
            \texttt{sym}(U_{\alpha}^T H V_{\alpha}) \succeq 0 \\
            \texttt{sym}(U_{\beta}^T H V_{\beta}) \preceq 0
        }} \left\| \begin{bmatrix}
            U_{\alpha}^T (H - X) V_{\alpha}    & U_{\alpha}^T (H - X) V_{\tilde{r}} & U_{\alpha}^T (H - X) V_{\beta} \\
            U_{\tilde{r}}^T (H - X) V_{\alpha} & U_{\tilde{r}}^T (H - X) V_{\tilde{r}} & U_{\tilde{r}}^T (H - X) V_{\beta} \\
            U_{\beta}^T (H - X) V_{\alpha}     & U_{\beta}^T (H - X) V_{\tilde{r}}     & U_{\beta}^T (H - X) V_{\beta}
        \end{bmatrix} \right\|_F^2 \right] V^T \nonumber \\
        &\qquad= U \begin{bmatrix}
            U_{\alpha}^T X V_{\alpha} - (\texttt{sym}(U_{\alpha}^T X V_{\alpha}))_{-} & U_{\alpha}^T X V_{\tilde{r}} & U_{\alpha}^T X V_{\beta} \\
            U_{\tilde{r}}^T X V_{\alpha} & U_{\tilde{r}}^T X V_{\tilde{r}} & U_{\tilde{r}}^T X V_{\beta} \\
            U_{\beta}^T X V_{\alpha}     & U_{\beta}^T X V_{\tilde{r}}      & U_{\beta}^T X V_{\beta} - (\texttt{sym}(U_{\beta}^T X V_{\beta}))_{+}
        \end{bmatrix} V^T \nonumber \\
        &\qquad= X - U_{\alpha} (\texttt{sym}(U_{\alpha}^T X V_{\alpha}))_{-} V_{\alpha}^T - U_{\beta} (\texttt{sym}(U_{\beta}^T X V_{\beta}))_{+} V_{\beta}^T \nonumber \\
        &\qquad= X - U_{\alpha} \underbrace{(V_{\alpha}^T V_{\alpha})}_{=I} (\texttt{sym}(U_{\alpha}^T X V_{\alpha}))_{-} V_{\alpha}^T - U_{\beta} \underbrace{(V_{\beta}^T V_{\beta})}_{=I} (\texttt{sym}(U_{\beta}^T X V_{\beta}))_{+} V_{\beta}^T \nonumber \\
        &\qquad= X - (U_{\alpha} V_{\alpha}^T) (\texttt{sym}(V_{\alpha}U_{\alpha}^T X V_{\alpha} V_{\alpha}^T))_{-} - (U_{\beta} V_{\beta}^T) (\texttt{sym}(V_{\beta} U_{\beta}^T X V_{\beta} V_{\beta}^T))_{+} \nonumber \\
        &\qquad= X - J_{\alpha} (\texttt{sym}(J_{\alpha}^T X P_{V_{\alpha}}))_{-} - J_{\beta} (\texttt{sym}(J_{\beta}^T X P_{V_{\beta}}))_{+} \nonumber \\
    &\texttt{proj}_{T_{W_t}\mathcal{S}_{[\alpha, \beta]}}(X) \nonumber \\
        &\qquad= X - J_{\alpha} \texttt{proj\_nsd}(\texttt{sym}(J_{\alpha}^T X P_{V_{\alpha}})) - J_{\beta} \texttt{proj\_psd}(\texttt{sym}(J_{\beta}^T X P_{V_{\beta}}))
\end{align}$$
where $P_{V_{\alpha}}$ and $P_{V_{\beta}}$ are the projectors onto the right $\alpha$- and $\beta$-singular subspaces of $W_t$, respectively, and $J_{\alpha} := U_{\alpha} V_{\alpha}^T$ and $J_{\beta} := U_{\beta} V_{\beta}^T$ are the polar factors of $W_t$ restricted to the respective singular subspaces. We can compute these in a numerically stable way as in [Section 5.1.3](#513-numerically-stable-computation-of-the-singular-subspace-projectors).

Taking everything together yields,
```python
def project_to_tangent_spectral_band(W: jax.Array, X: jax.Array, alpha: float, beta: float, tol=1e-3) -> jax.Array:
    return jax.lax.cond(
        alpha == 0,
        lambda: project_to_tangent_spectral_ball(W, X, beta, tol),
        lambda: jax.lax.cond(
            jnp.logical_and(
                jnp.rint(jnp.trace(PV_alpha := jnp.eye(W.shape[1]) - eig_stepfun(W.T @ W / alpha**2, 1.+tol))) == 0,
                jnp.rint(jnp.trace(PV_beta := eig_stepfun(W.T @ W / beta**2, 1.-tol))) == 0,  # or spec_norm < beta - tol,
            ),
            lambda: X,  # tangent space is all matrices
            lambda: X
                - (J_alpha := (1./alpha) * W @ PV_alpha) @ proj_nsd(sym(J_alpha.T @ X @ PV_alpha))
                - (J_beta := (1./beta) * W @ PV_beta) @ proj_psd(sym(J_beta.T @ X @ PV_beta)),
        ),
    )
```

We can then compute the optimal updates $A^*$ as in [Section 5.2](#52-update-rule-for-steepest-descent-on-the-spectral-ball) via the projection above.

#### A1.1.1. Sanity check: Stiefel as a special case of the Spectral Band

On the Stiefel manifold $\texttt{St}(m, n) = \{ W \in \mathbb{R}^{m \times n} : W^T W = I_n \}$, the singular values of any $W \in \texttt{St}(m, n)$ are all equal to $1$. Thus, $\texttt{St}(m, n) = \mathcal{S}_{[1, 1]}$ and,
$$U_{\alpha=1} = U_{\beta=1} =: U \qquad\text{ and }\qquad V_{\alpha=1} = V_{\beta=1} =: V.$$
Without loss of generality (up to rotations), we can also choose that $U = W_t$ and $V = I_n$ such that $W_t = UIV^T$. Thus,
$$\begin{align}
    T_{W_t}\texttt{St}(m, n)
        &= T_{W_t} \mathcal{S}_{[1, 1]} \nonumber \\
        &= \{ H \in \mathbb{R}^{m \times n} : \texttt{sym}(U_{1}^T H V_{1}) \succeq 0, \texttt{sym}(U_{1}^T H V_{1}) \preceq 0 \} \nonumber \\
        &= \{ H \in \mathbb{R}^{m \times n} : \texttt{sym}(U^T H V) = 0 \} \nonumber \\
        &= \{ H \in \mathbb{R}^{m \times n} : \texttt{sym}(W_t^T H) = 0 \} \nonumber \\
        &= \{ H \in \mathbb{R}^{m \times n} : W_t^T H + H^T W_t = 0 \} \nonumber
\end{align}$$
which is simply the textbook definition of the tangent space at a point on the Stiefel manifold.

As for the tangent space, we have,
$$\begin{align}
    \texttt{proj}_{T_{W_t}\texttt{St}(m, n)}(X)
        &= \texttt{proj}_{T_{W_t}\mathcal{S}_{[1, 1]}}(X) \nonumber \\
        &= X - U_{1} (\texttt{sym}(U_{1}^T X V_{1}))_{-} V_{1}^T - U_{1} (\texttt{sym}(U_{1}^T X V_{1}))_{+} V_{1}^T \nonumber \\
        &= X - U \texttt{sym}(U^T X V) V^T \nonumber \\
        &= X - W_t \texttt{sym}(W_t^T X) \nonumber
\end{align}$$
which is, again, the textbook formula for the projection onto the tangent space at a point on the Stiefel manifold.

### A1.2. Dual ascent for steepest descent on the Spectral Band

In this work, we compute the optimal updates $A^*$ via PDHG and (orthogonal) projections onto the tangent cones/spaces and the norm balls. However, prior work by [Bernstein (2025)](https://thinkingmachines.ai/blog/modular-manifolds/) and more recently by [Keigwin, et al. (2025)](https://www.tilderesearch.com/vignettes/gram-space) use dual ascent methods instead. We will generalize this approach to the Spectral Band and also discuss *why* we believe PDHG via projections is preferable in this setting.

To recap, our optimization problem is, given a "raw gradient" $G_t \in \mathbb{R}^{m \times n}$ and a choice of norm $\| \cdot \|_{W_t}$ to do steepest descent under, we want to find the optimal update $A^*$ such that,
$$\begin{align}
    A^*
        &= \arg\min_{\| A \|_{W_t} \leq \eta} \langle G_t, A \rangle \quad \text{s.t. } \texttt{sym}(U_{\alpha}^T A V_{\alpha}) \succeq 0, \texttt{sym}(U_{\beta}^T A V_{\beta}) \preceq 0 \nonumber \\
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle G_t, A \rangle \quad \text{s.t. } \| A \|_{W_t} \leq \eta, L_{\alpha}(A) \succeq 0, L_{\beta}(A) \preceq 0
\end{align}$$
where the linear maps $L_{\alpha}, L_{\beta} : \mathbb{R}^{m \times n} \to \mathbb{S}^{r}_{\pm}$ and their adjoints are defined as,
$$\begin{align}
    L_{\alpha}(X) &= \texttt{sym}(U_{\alpha}^T X V_{\alpha}) \qquad L_{\alpha}^*(S_{\alpha}) = U_{\alpha} S_{\alpha} V_{\alpha}^T \nonumber \\
    L_{\beta}(X)  &= \texttt{sym}(U_{\beta}^T X V_{\beta}) \qquad L_{\beta}^*(S_{\beta}) = U_{\beta} S_{\beta} V_{\beta}^T \nonumber
\end{align}$$
such that $\langle L_{\alpha}(X), S_{\alpha} \rangle = \langle X, L_{\alpha}^*(S_{\alpha}) \rangle$ and likewise for $L_{\beta}$.

Restricting the dual states $S_{\alpha}$ to the negative semidefinite cone, $S_{\alpha} \in \mathbb{S}^{r_{\alpha}}_{-}$, and $S_{\beta}$ to the positive semidefinite cone, $S_{\beta} \in \mathbb{S}^{r_{\beta}}_{+}$, yields the Lagrangian,
$$\begin{align}
    \mathcal{L}(A, S_{\alpha}, S_{\beta})
        &= \langle G_t, A \rangle + \mathcal{i}_{\| \cdot \|_{W_t} \leq \eta}(A) + \langle S_{\alpha}, L_{\alpha}(A) \rangle + \langle S_{\beta}, L_{\beta}(A) \rangle \nonumber \\
        &= \mathcal{i}_{\| \cdot \|_{W_t} \leq \eta}(A) + \langle G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}), A \rangle.
\end{align}$$
One can then check that,
$$A^* = \arg\min_{\| A \|_{W_t} \leq \eta} \left[ \max_{S_{\alpha} \in \mathbb{S}^{r_{\alpha}}_{-}, S_{\beta} \in \mathbb{S}^{r_{\beta}}_{+}} \mathcal{L}(A, S_{\alpha}, S_{\beta}) \right]$$
And by Sion's minimax theorem, we can swap the order of minimization and maximization,
$$ \min_{\| A \|_{W_t} \leq \eta} \left[ \max_{S_{\alpha} \in \mathbb{S}^{r_{\alpha}}_{-}, S_{\beta} \in \mathbb{S}^{r_{\beta}}_{+}} \mathcal{L}(A, S_{\alpha}, S_{\beta}) \right] = \max_{S_{\alpha} \in \mathbb{S}^{r_{\alpha}}_{-}, S_{\beta} \in \mathbb{S}^{r_{\beta}}_{+}} \left[ \underbrace{\min_{\| A \|_{W_t} \leq \eta} \mathcal{L}(A, S_{\alpha}, S_{\beta})}_{A(S_{\alpha}, S_{\beta})} \right]$$

First, let us consider the primal minimizer,
$$\begin{align}
    A(S_{\alpha}, S_{\beta})
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \mathcal{L}(A, S_{\alpha}, S_{\beta}) \nonumber \\
        &= \arg\min_{A \in \mathbb{R}^{m \times n}} \mathcal{i}_{\| \cdot \|_{W_t} \leq \eta}(A) + \langle G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}), A \rangle \nonumber \\
        &= \arg\min_{\| A \|_{W_t} \leq \eta} \langle G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}), A \rangle \nonumber \\
        &= -\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta})) \nonumber
\end{align}$$
where $\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}$ is the linear minimization oracle for the norm $\| \cdot \|_{W_t}$ [(Pethick et al., 2025)](https://arxiv.org/abs/2502.07529). For the $\texttt{RMS} \to \texttt{RMS}$ norm, we have $\texttt{LMO}_{\| \cdot \|_{\texttt{RMS} \to \texttt{RMS}} \leq \eta}(X) = \sqrt{\frac{m}{n}}\eta \cdot \texttt{msign}(X).$

This then yields the dual problem,
$$\begin{equation}
    \max_{S_{\alpha} \in \mathbb{S}^{r_{\alpha}}_{-}, S_{\beta} \in \mathbb{S}^{r_{\beta}}_{+}} -\eta \| G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}) \|_{W_t}^*
\end{equation}$$
where $\| \cdot \|_{W_t}^*$ is the dual norm of $\| \cdot \|_{W_t}$. For the $\texttt{RMS} \to \texttt{RMS}$ norm, we have $\| \cdot \|_{\texttt{RMS} \to \texttt{RMS}}^* \propto \| \cdot \|_{\text{nuc}}$. And by chain rule, the above has supergradients,
$$\begin{align}
    \nabla_{S_{\alpha}} (-\eta\| G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}) \|_{W_t}^*)
        &= -L_{\alpha}(\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}))) \nonumber \\
    \nabla_{S_{\beta}} (-\eta\| G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}) \|_{W_t}^*)
        &= -L_{\beta}(\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(G_t + L_{\alpha}^*(S_{\alpha}) + L_{\beta}^*(S_{\beta}))) \nonumber
\end{align}$$
We can then do gradient ascent on the dual variables $S_{\alpha}$ and $S_{\beta}$ while projecting them back to their respective cones after each step. Taking everything together then yields,
$$\begin{align}
    A_t
        &= -\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(G_t + L_{\alpha}^*(S_{\alpha, t}) + L_{\beta}^*(S_{\beta, t})) \\
    S_{\alpha, t+1}
        &= \texttt{proj\_nsd}\left(S_{\alpha, t} + \sigma L_{\alpha}( A_t )\right) \\
    S_{\beta, t+1}
        &= \texttt{proj\_psd}\left(S_{\beta, t} + \sigma L_{\beta}( A_t )\right)
\end{align}$$
where $\sigma > 0$ is the dual ascent learning rate. At convergence, we have $A_t \to A^*$.

#### A1.2.1. Initialization strategy

We can initialize the dual states $S_{\alpha, 0}$ and $S_{\beta, 0}$ as zero matrices. However, notice that the update rule for $A_t$ above is already *similar* to the 1-step Alternating Projections heuristic we discussed and have shown to be effective in earlier sections.
$$\begin{align}
    \widetilde{A_0}
        &= \texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(\texttt{proj}_{T_{W_t}\mathcal{S}_{[\alpha, \beta]}}(-G_t)) \qquad\text{(1-step Alternating Projections heuristic)} \nonumber \\
        &= \texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}((-G_t) - U_{\alpha} (\texttt{sym}(U_{\alpha}^T (-G_t) V_{\alpha}))_{-} V_{\alpha}^T - U_{\beta} (\texttt{sym}(U_{\beta}^T (-G_t) V_{\beta}))_{+} V_{\beta}^T) \nonumber \\
        &= -\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(G_t + U_{\alpha} (\texttt{sym}(U_{\alpha}^T (-G_t) V_{\alpha}))_{-} V_
        {\alpha}^T + U_{\beta} (\texttt{sym}(U_{\beta}^T (-G_t) V_{\beta}))_{+} V_{\beta}^T) \nonumber \\
        &= -\texttt{LMO}_{\| \cdot \|_{W_t} \leq \eta}(G_t + L_{\alpha}^*(\widetilde{S_{\alpha, 0}}) + L_{\beta}^*(\widetilde{S_{\beta, 0}})) \nonumber \\
\end{align}$$
where,
$$\begin{align}
    \widetilde{S_{\alpha, 0}}
        &= \texttt{proj\_nsd}(L_{\alpha}(-G_t))
        \qquad\qquad
    \widetilde{S_{\beta, 0}}
        = \texttt{proj\_psd}(L_{\beta}(-G_t)) \nonumber \\
\end{align}$$

#### A1.2.2. JAX implementation

```python
def dual_ascent(
    G: jax.Array,  # R^(m x n)
    L_primal: Callable[[jax.Array], Tuple[jax.Array]],  # R^(m x n) -> K_dual
    L_dual:  Callable[[Tuple[jax.Array]], jax.Array],  # K_dual -> R^(m x n)
    proj_K_dual: Callable[[Tuple[jax.Array]], Tuple[jax.Array]],  # K_dual -> K_dual
    norm_K_dual: Callable[[Tuple[jax.Array]], float],  # K_dual -> R
    lmo: Callable[[jax.Array], jax.Array],  # R^(m x n) -> R^(m x n)
    *,
    max_steps: int=128, sigma: float=1.0,
    rtol: float=1e-3, atol: float=1e-6,
):
    def cond_fn(state):
        S, k, res = state
        return jnp.logical_and(k < max_steps, jnp.logical_and(res > atol, res > rtol * norm_K_dual(S)))

    def body_fn(state):
        S, k, _ = state
        A = -lmo(G + L_dual(S))
        gradS = L_primal(A)
        S_new = proj_K_dual(jax.tree_util.tree_map(lambda s, g: s + sigma / jnp.sqrt(k+1) * g, S, gradS))
        res = norm_K_dual(jax.tree_util.tree_map(lambda s, s_old: s - s_old, S_new, S))
        return S_new, k+1, res

    S = proj_K_dual(L_primal(-G))
    S_final, n_iters, final_res = jax.lax.while_loop(cond_fn, body_fn, (S, 0, jnp.inf))
    A_final = -lmo(G + L_dual(S_final))
    return A_final


def dual_ascent_spectral_band_spectral_norm(
    W: jax.Array, G: jax.Array,
    alpha: float, beta: float, target_norm: float,
    *,
    max_steps: int=128, sigma: float=1.0,
    eig_tol: float=1e-1, rtol: float=1e-3, atol: float=1e-6,
):
    m, n = W.shape
    PV_alpha = jnp.eye(n) - eig_stepfun(W.T @ W / alpha**2, 1.+eig_tol)
    PV_beta  = eig_stepfun(W.T @ W / beta**2, 1.-eig_tol)
    Omega = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=W.dtype)
    V_alpha, V_beta = _orthogonalize(PV_alpha @ Omega), _orthogonalize(PV_beta @ Omega)
    U_alpha, U_beta = 1./alpha * W @ V_alpha, 1./beta * W @ V_beta

    L_alpha_primal = lambda A: sym(U_alpha.T @ A @ V_alpha)
    L_beta_primal  = lambda A: sym(U_beta.T @ A @ V_beta)
    L_alpha_dual   = lambda S: U_alpha @ S @ V_alpha.T
    L_beta_dual    = lambda S: U_beta @ S @ V_beta.T

    L_primal    = lambda A: (L_alpha_primal(A), L_beta_primal(A))
    L_dual      = lambda S: L_alpha_dual(S[0]) + L_beta_dual(S[1])
    proj_K_dual = lambda S: (proj_nsd(S[0]), proj_psd(S[1]))
    norm_K_dual = lambda S: jnp.linalg.norm(S[0]) + jnp.linalg.norm(S[1])

    return dual_ascent(
        G,
        L_primal=L_primal,
        L_dual=L_dual,
        proj_K_dual=proj_K_dual,
        norm_K_dual=norm_K_dual,
        lmo=lambda X: target_norm * _orthogonalize(X),
        max_steps=max_steps, sigma=sigma,
        rtol=rtol, atol=atol,
    )
```

#### A1.2.3. Sanity check: Stiefel as a special case of the Spectral Band

Pick the $\texttt{RMS} \to \texttt{RMS}$ norm to descend under. For the Stiefel case, we have $\alpha = \beta = 1$, $U_1 = U_2 =: U$, and $V_1 = V_2 =: V$. And WLOG, up to rotations, we can choose $U = W_t$ and $V=I$. Thus,
$$\begin{align}
    A_t
        &= -\sqrt{\frac{m}{n}}\eta \cdot \texttt{msign}(G_t + L_{\alpha}^*(S_{\alpha, t}) + L_{\beta}^*(S_{\beta, t})) \nonumber \\
        &= -\sqrt{\frac{m}{n}}\eta \cdot \texttt{msign}(G_t + U S_{\alpha, t} V^T + U S_{\beta, t} V^T ) \nonumber \\
        &= -\sqrt{\frac{m}{n}}\eta \cdot \texttt{msign}(G_t + U \Lambda V^T ) \nonumber \\
        &= -\sqrt{\frac{m}{n}}\eta \cdot \texttt{msign}(G_t + W_t \Lambda )
\end{align}$$
where $\Lambda_t := S_{\alpha, t} + S_{\beta, t} \in \mathbb{S}^n$. And,
$$\begin{align}
    \Lambda_{t+1}
        &= S_{\alpha, t+1} + S_{\beta, t+1} \nonumber \\
        &= \texttt{proj\_nsd}\left(S_{\alpha, t} + \sigma L_{\alpha}( A_t )\right) + \texttt{proj\_psd}\left(S_{\beta, t} + \sigma L_{\beta}( A_t )\right) \nonumber \\
        &= S_{\alpha, t} + S_{\beta, t} + 2\sigma \texttt{sym}(U^T A_t V) \nonumber \\
        &= \Lambda_t + 2\sigma \texttt{sym}(W_t^T A_t) \nonumber \\
        &= \Lambda_t + \sigma \cdot (W_t^T A_t + A_t^T W_t).
\end{align}$$
Both match the update rules that [Bernstein (2025)](https://thinkingmachines.ai/blog/modular-manifolds/) previously derived.

#### A1.2.4. Why PDHG over dual ascent?

Mainly numerical stability.
- First, concerning the internal states. We would still need to maintain states in both cases, but PDHG only needs to maintain projected gradients and the gap between them (which we could even avoid storing explicitly). In contrast, the $S_\alpha$ and $S_\beta$ dual states can grow unboundedly large, especially with a poor choice of dual ascent learning rate $\sigma$.
- Second is that the dual ascent approach requires us to construct $U_\alpha$, $V_\alpha$, $U_\beta$, and $V_\beta$ which leads to additional numerical instability compared to just maintaining the projectors $P_{V_\alpha}$ and $P_{V_\beta}$ as in the PDHG approach.
- Third is that PDHG often converges faster in practice.

For future work, one may explore PDHG on the dual space as well, which may combine the best of both worlds.

## A2. Weight Decay as a (manifold) constraint

In [Section 6.2](#62-larger-updates-accelerate-generalization-addition-modulo-31-problem), we mentioned that the equivalent to constraining the weight norms to be inside a ball of radius $4$ is to use a weight decay of $1/4$. Here, we shall explain *why*.

### A2.1. Weight norms at equilibrium under weight decay

Prior work has shown that steepest descent under any chosen norm with weight decay $\lambda$ already (secretly) constrains the weights to be inside the corresponding norm ball of radius $1/\lambda$ ([Chen et al., 2025](https://arxiv.org/abs/2506.15054); [Pethick et al., 2025](https://arxiv.org/abs/2502.07529); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)).

> **Proposition 3 (Equilibrium of weight norms with weight decay).** Consider the update form,
> $$\begin{equation} W_{t+1} = (1 - \eta\lambda) W_t + A_t^* \end{equation}$$
> where $\lambda > 0$ is the weight decay term, $\eta > 0$ is the learning rate, and $\| A_t^* \| \leq \eta$ for some norm $\| \cdot \|$ chosen a priori. If $\eta\lambda \leq 1$, then the weight norms are upper bounded as,
> $$\lim\sup_{t \to \infty} \| W_{t} \| \to \frac{1}{\lambda}.$$

{{< collapse summary="Show **Proof of Proposition 3**" openByDefault=false >}}
> **Proof.** Unrolling the recurrence yields,
> $$W_{t} = (1 - \eta\lambda)^t W_0 + \sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} A_i^*$$
> Thus, by triangle inequality,
> $$\begin{align}
    \| W_{t} \|
        &\leq (1 - \eta\lambda)^t \| W_0 \| + \sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} \| A_i^* \| \nonumber \\
        &\leq (1 - \eta\lambda)^t \| W_0 \| + \eta \sum_{j=0}^{t-1} (1 - \eta\lambda)^{j} \nonumber \\
        &= (1 - \eta\lambda)^t \| W_0 \| + (1 - (1 - \eta\lambda)^t)\frac{1}{\lambda} \nonumber \\
\end{align}$$
> Hence, $$\lim\sup_{t \to \infty} \| W_{t} \| \to \frac{1}{\lambda} \blacksquare$$
{{< /collapse >}}

Furthermore, if the steepest descent updates stabilizes to some constant $A_t^* \to A_*^*$, then the weights also stabilizes to $W_t \to \frac{A_*^*}{\eta\lambda}$.

> **Proposition 4 (Mechanism behind gradient-weight alignment).** Consider the update form,
> $$\begin{equation} W_{t+1} = (1 - \eta\lambda) W_t + A_t^* \end{equation}$$
> where $\lambda > 0$ is the weight decay term, $\eta > 0$ is the learning rate, and $\| A_t^* \| \leq \eta$ for some norm $\| \cdot \|$ chosen a priori. If $\eta\lambda \leq 1$ and $A_t^* \to A_*^*$ as $t \to \infty$, then the weights converge as,
> $$\lim_{t \to \infty} W_{t} \to \frac{A_*^*}{\eta\lambda}.$$

{{< collapse summary="Show **Proof of Proposition 4**" openByDefault=false >}}
> **Proof.** Unrolling the recurrence yields,
> $$\begin{align}
    W_{t}
        &= (1 - \eta\lambda)^t W_0 + \sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} A_i^* \nonumber \\
        &= (1 - \eta\lambda)^t W_0 + \underbrace{\sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} A_*^* - \sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} A_*^*}_{=0} + \sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} A_i^* \nonumber \\
        &= (1 - \eta\lambda)^t W_0 + \frac{1 - (1 - \eta\lambda)^t}{\eta\lambda} A_*^* + \sum_{i=0}^{t-1} (1 - \eta\lambda)^{t-1-i} (A_i^* - A_*^*) \nonumber \\
\end{align}$$
> Thus, since $\eta\lambda \leq 1$ and $A_t^* \to A_*^*$ as $t \to \infty$, the first and last terms vanish as $t \to \infty$, yielding,
> $$\begin{align}
    \lim_{t \to \infty} W_{t}
        &= \lim_{t \to \infty} \frac{1 - (1 - \eta\lambda)^t}{\eta\lambda} A_*^* \nonumber \\
        &= \frac{A_*^*}{\eta\lambda} \blacksquare \nonumber \\
\end{align}$$
{{< /collapse >}}

Intuitively, this means that, for example, Muon "pulls" all of the *singular values* to $\frac{\eta}{\eta\lambda} = \frac{1}{\lambda}$ as weight updates align during training. With the SignSGD optimizer [(Bernstein et al., 2018)](https://arxiv.org/abs/1802.04434), we would instead expect the weight *entries* to stabilize to $\pm \frac{1}{\lambda}$. Likewise, since SignSGD is simply AdamW with the momentum parameters set to zero, we would expect AdamW to also stabilize the weight *entries* to (roughly) $\pm \frac{1}{\lambda}$ as well.

### A2.2. Why we may want to hold $\eta\lambda$ constant as we scale

Prior work has also (empirically) shown that if we want hyperparameters to transfer, we should hold the product $\eta\lambda$ constant (1) during hyperparameter search [(Schaipp, 2024)](https://fabian-sp.github.io/posts/2024/02/decoupling/) and (2) as we scale the model via muP [(Kosson et al., 2025)](https://arxiv.org/abs/2510.19093). One possible reason why is that this is (roughly) equivalent to holding the *relative* update size constant with respect to the implicit weight norm constraint radius $R = 1/\lambda$.
$$\texttt{relative\_update\_size} = \frac{\texttt{update\_size}}{\texttt{constraint\_radius}} = \frac{\eta}{1/\lambda} = \eta\lambda$$

Intuitively, this means that the amount of "effort" required to change an "absolutely yes" direction to an "absolutely no" direction remains the same no matter the model width.

A consequence of this is that, as recommended by [Kosson et al. (2025)](https://arxiv.org/abs/2510.19093), when scaling up the model width via muP, we should also scale up the weight decay by the same factor as we scale down the learning rate. But increasing the weight decay decreases the implicit constraint radius $1/\lambda$, and thus limits the search space of the weights. While this may be harmful in some settings (if the optimal point lies outside the radius), recent work has also shown that deliberately reducing the search space (but not too much) prevents the model from simply memorizing the training data early in training and thereby improves generalization later on in training [(Tian, 2025)](https://arxiv.org/abs/2509.21519).
