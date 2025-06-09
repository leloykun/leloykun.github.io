---
title: "Numerically Stable Spectral Clipping Via Newton-Schulz Iteration"
date: 2025-06-05
tags: ["Machine Learning", "Optimizers", "Architecture-Optimizer Codesign"]
author: "Franz Louis Cesista"
description: "A small step towards hardware-architecture-optimizer codesign in deep learning."
summary: "A small step towards hardware-architecture-optimizer codesign in deep learning."
cover:
    image: clip_lifting_trap_fix.png
    alt: "Cover"
    relative: true
# editPost:
#     URL: "https://x.com/leloykun/status/1847919153589735705"
#     Text: "Crossposted from X (formerly Twitter)"
---

## Introduction

Here I'll discuss a numerically stable way to perform spectral clipping, i.e., clipping the singular values of a matrix to a certain range. This is useful in deep learning because it allows us to control the 'growth' of our weights and weight updates, enabling faster and stabler feature learning. As discussed in a [previous post](../steepest-descent-non-riemannian/#22-feature-learning-perspective),
> If we want the Euclidean norm of our features and feature updates to 'grow' with the model size,
> then the *Spectral norm* of our weights and weight updates must also 'grow' with the model size.

There are multiple ways to control the spectral norm of our (matrix-structured) weights and weight updates. One is to "pull" **all** of the singular values to some target value chosen a priori. This is what the Muon optimizer already does, but only on the weight updates: it takes the raw gradient and tries to "pull" its as many of its singular values to $\sqrt{\frac{d_{out}}{d_{in}}}$. This guarantees that the update step merely changes the activation RMS-norm of that layer by at most $1$ unit. We *could* also apply this process to the weights after every update step to guarantee that the weight norms *would not* blow up, but constraining the weight space to the Stiefel manifold is too strong of a constraint. We discuss more of this in our upcoming Neurips preprint. For now, we will focus on Spectral Clipping:

> **Definition 1 (Spectral Clipping)**. Let $W \in \mathbb{R}^{m \times n}$ and $W = U \Sigma V^T$ be its singular value decomposition where $\Sigma = (\sigma\_1, \ldots, \sigma\_{min(m,n)})$ are the singular values of $W$. Then we define Spectral Clipping as the following matrix function $\texttt{spectral\\_clip}\_{[\sigma\_{min}, \sigma\_{max}]}: \mathbb{R}^{m \times n} \to \mathbb{R}^{m \times n}$,
> $$\begin{equation}\texttt{spectral\\_clip}\_{[\sigma\_{min}, \sigma\_{max}]}(W) = U \texttt{clip}\_{[\sigma\_{min}, \sigma\_{max}]}(\Sigma) V^T\label{1}\end{equation}$$
> where $\sigma\_{min}, \sigma\_{max} \in [0, \infty)$ are hyperparameters that control the minimum and maximum attainable singular values of the resulting matrix and $\texttt{clip}\_{[\alpha, \beta]}: \mathbb{R} \to \mathbb{R}$ is applied element-wise on the singular values of $W$,
> 
> $$\begin{equation}\texttt{clip}\_{[\alpha, \beta]}(x) = \begin{cases}
\alpha & \texttt{if } x < \alpha \\\\
x & \texttt{if } \alpha \leq x \leq \beta \\\\
\beta & \texttt{if } \beta < x
\end{cases}\end{equation}$$
> where $\alpha, \beta \in \mathbb{R} \cup \\{-\infty, \infty\\}$ and $\alpha \leq \beta$.

Note that since the singular values of a matrix are guaranteed to be non-negative, $\texttt{clip}$ above does not need to be bidirectional. And setting $\alpha \leq 0$ and/or $\beta = \infty$ massively simpifies our (matrix) function, resulting in efficiency gains,
- $\texttt{clip}\_{[\leq 0, \beta]}(x) = \min(x, \beta)$; and
- $\texttt{clip}\_{[\alpha, \infty]}(x) = \max(x, \alpha)$ which is simply the (shifted-)$\texttt{ReLU}$.

In practice, the former would suffice for constraining the weights of neural networks. However, we will keep both parameters $\alpha, \beta$ in this work for generality and in case one would need to constrain the weights to always be full rank to prevent the activations from collapsing in dimension.

## Towards hardware-architecture-optimizer codesign

In deep learning, we not only have to be mindful of architecture-optimizer codesign but also hardware-software codesign. That is, architectural and optimizer choices and how we implement them have to be hardware-aware so that we can squeeze as much performance as we can from our GPUs/TPUs.

For example, the naive way to compute Spectral Clipping is to directly compute the SVD, clip the singular values we get from it, then reconstruct the matrix using the clipped singular values. A JAX implementation would look like this:
```python
def naive_spectral_clip(W: jax.Array, sigma_min: float=-1., sigma_max: float=1.):
    U, S, Vt = jnp.linalg.svd(W, full_matrices=False)
    S_clipped = jnp.clip(S, min=sigma_min, max=sigma_max)
    return U @ jnp.diag(S_clipped) @ Vt
```
However, this is not recommended because computing the SVD directly (1) does not take advantage of the GPUs' tensor cores and (2) requires higher numerical precision, typically 32-bit float types. These not only slow things down but also increase precious memory usage, making it hard to scale to larger models.

Ideally, we want to *only* use operations that (1) have fast implementations on GPUs/TPUs and (2) are stable under lower numerical precision, e.g., 16-bit, 8-bit, even 4-bit float types. So, elementwise operations like matrix addition and scalar multiplication, matrix multiplication, matrix-vector products, among others are preferred, but not operations like matrix inversions or SVD decomposition, etc. With the proper coefficients, (semi-)orthogonalization via Newton-Schulz iteration for computing the matrix sign function has also been shown to be fast and numerically stable under lower precision (Jordan et al., 2024), thus we can use that here.

### Finding a suitable surrogate function for $\texttt{clip}$

This is the fun part.

So, how do we compute spectral clipping while only using simple, but fast & numerically stable operations? First, let's list the operations we can actually use and consider how they act on the matrix itself and its singular values. There are more operations we can use that aren't listed here, but these would suffice for our problem.

| **Operation**                                          |   **Matrix Form**   | **Action on Singular Values** |
| :----------------------------------------------------- | :-----------------: | :---------------------------: |
| Scalar multiplication                                  |        $cW$         |           $c\Sigma$           |
| Application of polynomial function $\texttt{p}(\cdot)$ |   $\texttt{p}(W)$   |     $\texttt{p}(\Sigma)$      |
| Application of matrix sign function                    | $\texttt{msign}(W)$ |    $\texttt{sign}(\Sigma)$    |

Let's reconstruct the $\mathbb{R} \to \mathbb{R}$ clipping on the singular values with these elementary functions first, then let's use it to construct the matrix form. Here we take advantage of the following identity,
$$\begin{equation}|x| = x \cdot \texttt{sign}(x)\end{equation}$$
With this, we can now construct $\texttt{clip}$ as follows,

![](clip_abs_trick.png#center)

> **Proposition 2 (Computing $\texttt{clip}$ via $\texttt{sign}$).** Let $\alpha, \beta \in \mathbb{R} \cup \\{-\infty, \infty\\}$ and $\texttt{clip}: \mathbb{R} \to \mathbb{R}$ be the clipping function defined in Definition 1. Then,
> $$\begin{equation}\texttt{clip}\_{[\alpha, \beta]}(x) = \frac{\alpha + \beta + (\alpha - x)\texttt{sign}(\alpha - x) - (\beta - x)\texttt{sign}(\beta - x)}{2}\label{4}\end{equation}$$

> **Proof:** It would suffice to show that,
> $$\begin{equation}\texttt{clip}\_{[\alpha, \beta]}(x) = \frac{\alpha + \beta + |\alpha - x| - |\beta - x|}{2}\end{equation}$$
> For this, we can simply check case-by-case,
> |             $x$             | $\left \| \alpha - x\right \| $ | $\left \| \beta - x\right \| $ | $\frac{\alpha + \beta +  \| \alpha - x \| - \| \beta - x \| }{2}$ |
> | :-------------------------: | :-----------------------------: | :----------------------------: | :---------------------------------------------------------------: |
> |        $x < \alpha$         |          $\alpha - x$           |          $\beta - x$           |                             $\alpha$                              |
> | $\alpha \leq x \leq \beta $ |          $x - \alpha$           |          $\beta - x$           |                                $x$                                |
> |         $\beta < x$         |          $x - \alpha$           |          $x - \beta$           |                              $\beta$                              |
> 
> Combining Equations (3) and (5) then gives us Equation $\eqref{4}$. $\blacksquare$

### Lifting to matrix form (the naive & incorrect way)

![](clip_lifting_trap.png#center)

A naive way to lift Equation $\eqref{4}$ above to matrix form is to simply replace the variables, scalar constants, and scalar (sub-)functions with their corresponding matrix form, i.e., replace $x$ with $W$, $1$ with $I$, and $\texttt{sign}(\cdot)$ with $\texttt{msign}(\cdot)$. This gives us the following matrix function,

$$\begin{align}
    \texttt{f}(W) &= (1/2) \cdot [(\alpha + \beta)I + (\alpha I - W) \texttt{msign}(\alpha I - W)^T\nonumber\\\\
    &\qquad\qquad\qquad\qquad\\;\\;- (\beta I - W) \texttt{msign}(\beta I - W)^T]
\end{align}$$

However, as communicated to me by You Jiacheng & Su Jianlin, this does not work (see figure above) because $I$ may not share the same singular vectors as $W$.

Another problem is that $\texttt{f}$ does not preserve the dimensions of the input matrix $W$. To see this, note that both $\alpha I - W$ and $\texttt{msign}(\alpha I - W)$ have shape $m \times n$ and so $(\alpha I - W) \texttt{msign}(\alpha I - W)^T$ must have shape $m \times m$. The same is true for the other term.

$$\begin{aligned}
    \texttt{f}(W) &= (1/2) \cdot [(\alpha + \beta)I_{\color{red}{m \times m}} + (\alpha I - W) \texttt{msign}(\alpha I - W)^T\\\\
    &\qquad\qquad\qquad\qquad\qquad- \underbrace{\underbrace{(\beta I - W)}_{m \times n} \underbrace{\texttt{msign}(\beta I - W)^T}\_{n \times m}}\_{\color{red}{m \times m}}]
\end{aligned}$$

### Lifting to matrix form (the proper way)

![](clip_lifting_trap_fix.png#center)

To properly lift Equation $\eqref{4}$ to matrix form, let's combine it with Equation $\eqref{1}$,
$$\begin{align}
    \texttt{spectral\\_clip}\_{[\alpha, \beta]}(W)
        &= U \texttt{clip}\_{[\alpha, \beta]}(\Sigma) V^T\nonumber\\\\
        &= U \frac{(\alpha + \beta) I + (\alpha I - \Sigma)\texttt{sign}(\alpha I - \Sigma) - (\beta I - \Sigma)\texttt{sign}(\beta I - \Sigma)}{2} V^T\nonumber\\\\
        &= (1/2) \cdot [(\alpha + \beta) UV^T\nonumber\\\\
        &\qquad\qquad+ U (\alpha I - \Sigma ) \texttt{sign}(\alpha I - \Sigma) V^T\nonumber\\\\
        &\qquad\qquad- U (\beta I - \Sigma ) \texttt{sign}(\beta I - \Sigma) V^T]\nonumber\\\\
        &= (1/2) \cdot [(\alpha + \beta) UV^T\nonumber\\\\
        &\qquad\qquad+ U (\alpha I - \Sigma ) (V^TV) \texttt{sign}(\alpha I - \Sigma) (U^TU) V^T\nonumber\\\\
        &\qquad\qquad- U (\beta I - \Sigma ) (V^TV) \texttt{sign}(\beta I - \Sigma) (U^TU) V^T]\nonumber\\\\
        &= (1/2) \cdot [(\alpha + \beta) UV^T\nonumber\\\\
        &\qquad\qquad+ (\alpha UV^T - U\Sigma V^T) (V \texttt{sign}(\alpha I - \Sigma) U^T)(UV^T)\nonumber\\\\
        &\qquad\qquad- (\beta UV^T - U\Sigma V^T)  (V \texttt{sign}(\beta I - \Sigma) U^T)(UV^T)]\nonumber\\\\
        &= (1/2) \cdot [(\alpha + \beta) UV^T\nonumber\\\\
        &\qquad\qquad+ (\alpha UV^T - U\Sigma V^T) (U \texttt{sign}(\alpha I - \Sigma) V^T)^T(UV^T)\nonumber\\\\
        &\qquad\qquad- (\beta UV^T - U\Sigma V^T)  (U \texttt{sign}(\beta I - \Sigma) V^T)^T(UV^T)]\nonumber\\\\
        &= (1/2) \cdot [(\alpha + \beta) UV^T\nonumber\\\\
        &\qquad\qquad+ (\alpha UV^T - U\Sigma V^T) \texttt{msign}(\alpha UV^T - U\Sigma V^T)^T(UV^T)\nonumber\\\\
        &\qquad\qquad- (\beta UV^T - U\Sigma V^T)  \texttt{msign}(\beta UV^T - U\Sigma V^T)^T(UV^T)]\nonumber\\\\
        &= (1/2) \cdot [(\alpha + \beta) \texttt{msign}(W)\nonumber\\\\
        &\qquad\qquad+ (\alpha \cdot\texttt{msign}(W) - W) \texttt{msign}(\alpha \cdot\texttt{msign}(W) - W)^T\texttt{msign}(W)\nonumber\\\\
        &\qquad\qquad- (\beta  \cdot\texttt{msign}(W) - W) \texttt{msign}(\beta  \cdot\texttt{msign}(W) - W)^T\texttt{msign}(W)]\nonumber\\\\
    \texttt{spectral\\_clip}\_{[\alpha, \beta]}(W)
        &= (1/2) \cdot [(\alpha + \beta)I\nonumber\\\\
        &\qquad\qquad+ (\alpha \cdot\texttt{msign}(W) - W) \texttt{msign}(\alpha \cdot\texttt{msign}(W) - W)^T\nonumber\\\\
        &\qquad\qquad- (\beta  \cdot\texttt{msign}(W) - W) \texttt{msign}(\beta  \cdot\texttt{msign}(W) - W)^T\nonumber\\\\
        &\qquad\qquad]\\;\texttt{msign}(W)\label{7}
\end{align}$$

And viola, we're done. The following code implements this in JAX,
```python
def spectral_clip(W: jax.Array, alpha: float=-1., beta: float=1.):
    if flip := W.shape[0] > W.shape[1]:
        W = W.T
    OW = _orthogonalize_via_newton_schulz(W)
    result = (1/2) * (
        (alpha + beta) * jnp.eye(W.shape[0])
        + (alpha * OW - W) @ _orthogonalize_via_newton_schulz(alpha * OW - W).T
        - (beta * OW - W) @ _orthogonalize_via_newton_schulz(beta * OW - W).T
    ) @ OW
    if flip:
        result = result.T
    return result
```
where `_orthogonalize_via_newton_schulz` above implements Jordan's (2024) Newton-Schulz iteration for computing the matrix sign function. Note that we're calling `_orthogonalize_via_newton_schulz` thrice here, which is not ideal.

## Variants and optimizations

### Sanity check: orthogonalization and scaling

As a simple test-case, let's verify that setting the lower and upper bounds to be equal results in orthogonalization and scaling of the input matrix, i.e., $\texttt{spectral\\_clip}\_{[\sigma, \sigma]}(W) = \sigma \cdot \texttt{msign}(W)$. From Equation $\eqref{7}$, we have,

$$\begin{aligned}
    \texttt{spectral\\_clip}\_{[\sigma, \sigma]}(W)
        &= (1/2) \cdot [(\sigma + \sigma)I\nonumber\\\\
        &\qquad\qquad\cancel{+ (\sigma \cdot\texttt{msign}(W) - W) \texttt{msign}(\sigma \cdot\texttt{msign}(W) - W)^T}\nonumber\\\\
        &\qquad\qquad\cancel{- (\sigma  \cdot\texttt{msign}(W) - W) \texttt{msign}(\sigma  \cdot\texttt{msign}(W) - W)^T}\nonumber\\\\
        &\qquad\qquad]\\;\texttt{msign}(W)\\\\
    \texttt{spectral\\_clip}\_{[\sigma, \sigma]}(W) &= \sigma \cdot \texttt{msign}(W)\quad\blacksquare
\end{aligned}$$

### Unbounded below: Spectral Hardcapping

![](spectral_hardcap.png#center)

> Note: Su (2025) calls this "Singular Value Clipping" or "SVC" while our upcoming paper calls this "Spectral Hardcapping".

Singular values are guaranteed to be non-negative, so if we only want to bound the singular values from above, we can simply set $\alpha = 0$ in Equation $\eqref{4}$, i.e.,
$$\begin{align}
    \texttt{clip}\_{[0, \beta]}(x) &= \frac{0 + \beta + (0 - x)\texttt{sign}(0 - x) - (\beta - x)\texttt{sign}(\beta - x)}{2}\nonumber\\\\
    \texttt{clip}\_{[0, \beta]}(x) &= \frac{\beta + x - (\beta - x)\texttt{sign}(\beta - x)}{2}
\end{align}$$
Setting $\beta = 1$ recovers Su's (2025) and You's (2025) results. And following the approach above, we get,
$$\begin{aligned}
    \texttt{spectral\\_hardcap}(W; \beta)
        &= \texttt{spectral\\_clip}\_{[0, \beta]}(W)\\\\
    \texttt{spectral\\_hardcap}(W; \beta)
        &= (1/2) \cdot [\beta \cdot \texttt{msign}(W) + W\\\\
        &\qquad\qquad- (\beta  \cdot\texttt{msign}(W) - W) \texttt{msign}(\beta  \cdot\texttt{msign}(W) - W)^T \texttt{msign}(W)]
\end{aligned}$$

The following code implements this in JAX,
```python
def spectral_hardcap(W: jax.Array, beta: float=1.):
    if flip := W.shape[0] > W.shape[1]:
        W = W.T
    OW = _orthogonalize_via_newton_schulz(W)
    aW = beta * OW - W
    result = (1/2) * (beta*OW + W - aW @ _orthogonalize_via_newton_schulz(aW).T @ OW)
    if flip:
        result = result.T
    return result
```
We are now only calling `_orthogonalize_via_newton_schulz` twice here.

### Unbounded above: Spectral (Shifted-)ReLU

![](spectral_relu.png#center)

If we only want to bound the singular values from below, we set $\beta = +\infty$ in Equation $\eqref{4}$. First note that for a fixed $x \in [0, \infty)$,
$$\lim_{\beta \to +\infty} \texttt{sign}(\beta - x) = +1$$
Thus,
$$\begin{align}
    \texttt{clip}\_{[\alpha, +\infty]}(x)
        &= \lim_{\beta \to +\infty}\frac{\alpha + \beta + (\alpha - x)\texttt{sign}(\alpha - x) - (\beta - x)\texttt{sign}(\beta - x)}{2}\nonumber\\\\
    \texttt{clip}\_{[\alpha, +\infty]}(x) &= \frac{\alpha + x + (\alpha - x)\texttt{sign}(\alpha - x)}{2}
\end{align}$$
And following the approach above, we get,
$$\begin{aligned}
    \texttt{spectral\\_relu}(W; \alpha)
        &= \texttt{spectral\\_clip}\_{[\alpha, +\infty]}(W)\\\\
    \texttt{spectral\\_relu}(W; \alpha)
        &= (1/2) \cdot [\alpha \cdot \texttt{msign}(W) + W\\\\
        &\qquad\qquad+ (\alpha  \cdot\texttt{msign}(W) - W) \texttt{msign}(\alpha  \cdot\texttt{msign}(W) - W)^T \texttt{msign}(W)]
\end{aligned}$$

The following code implements this in JAX,
```python
def spectral_relu(W: jax.Array, alpha: float=1.):
    if flip := W.shape[0] > W.shape[1]:
        W = W.T
    OW = _orthogonalize_via_newton_schulz(W)
    aW = alpha * OW - W
    result = (1/2) * (alpha*OW + W + aW @ _orthogonalize_via_newton_schulz(aW).T @ OW)
    if flip:
        result = result.T
    return result
```

## An alternative approach: Higham's Anti-Block-Diagonal Trick

![](spectral_clip_abd_vs_nested_tight.gif#center)

In the previous sections, we apply our matrix function directly on $W$ resulting in nested applications of $\texttt{msign}$. Here, we will instead use Higham's anti-block-diagonal trick (Higham, 2008). This allows us to compute `_orthogonalize_via_newton_schulz` only once, reducing the complexity of the operations albeit at the cost of more compute and memory usage. This trick may not be practical in most settings, but the reduced complexity of the operations may be worth it when designing linear attention mechanisms with the spectral clipping function as a "sub-network". A neat property is that this would allow us to naturally scale test-time compute by scaling the number of steps in `_orthogonalize_via_newton_schulz`.

> **Theorem 3 (Higham's Anti-Block-Diagonal Trick)**. Let $g: \mathbb{R} \to \mathbb{R}$ be an odd analytic scalar function, $W \in \mathbb{R}^{m \times n}$, and construct the block matrix $S \in \mathbb{R}^{(m+n) \times (m+n)}$ as,
> $$S := \begin{bmatrix}
    0 & W \\\\
    W^T & 0
\end{bmatrix}$$
> and let $g(S)$ as the primary matrix function defined from the scalar function $g$.
> Then,
> $$g(S) = \begin{bmatrix}
    0 & g(W) \\\\
    g(W^T) & 0
\end{bmatrix}$$
> and hence,
> $$g(W) = [g(S)]_{12}$$

Note that, for this trick to work, our scalar function $\texttt{clip}_{[\alpha, \beta]}$ has to be *odd*. Thus we will impose the following constraint,
$$\alpha = -\beta.$$
Also note that,
$$\texttt{clip}\_{[-\sigma\_{max}, \sigma\_{max}]}(x) = \sigma\_{max} \cdot \texttt{clip}\_{[-1, 1]}(x / \sigma\_{max})$$
and thus it would suffice to construct $\texttt{spectral\\_clip}\_{[-1, 1]}(\cdot)$ first,
$$\begin{equation}
    \texttt{spectral\\_clip}\_{[-\sigma\_{max}, \sigma\_{max}]}(W) = \sigma\_{max}\cdot\texttt{spectral\\_clip}\_{[-1, 1]}(W / \sigma\_{max}).
\end{equation}$$

Now, applying Theorem 3 with $g = \texttt{clip}\_{[-1, 1]}$ gives us,
$$\begin{equation}\texttt{spectral\\_clip}\_{[-1, 1]}(W) = \left[ \frac{(I+S) \texttt{msign}(I+S) - (I-S) \texttt{msign}(I-S)}{2} \right]\_{12}\end{equation}$$

The following code implements this in JAX,
```python
def _spectral_clip(W: jax.Array):
    m, n = W.shape
    I = jnp.eye(m + n)
    S = jnp.block([[jnp.zeros((m, m)), W], [W.T, jnp.zeros((n, n))]])
    gS = (1/2) * (
        (I + S) @ _orthogonalize_via_newton_schulz (I + S)
        - (I - S) @ _orthogonalize_via_newton_schulz (I - S)
    )
    return gS[:m, m:]  # read off the top-right block

def spectral_clip(W: jax.Array, sigma_max: float=1.):
    return sigma_max * _spectral_clip(W / sigma_max)
```

Note that we are still calling `_orthogonalize_via_newton_schulz` twice here, which is not ideal either. Luckily, there's a neat trick that allows us to compute it only once.

### Optimizing the implementation via abstract algebra

First, notice that both 
$$I + S = \begin{bmatrix}
    I_m & W \\\\
    W^T & I_n
\end{bmatrix}\qquad I - S = \begin{bmatrix}
    I_m & -W \\\\
    -W^T & I_n
\end{bmatrix}$$
are block matrices of the form
$$\begin{bmatrix}
    P & Q \\\\
    Q^T & R
\end{bmatrix}$$
where $P, R$ are symmetric matrices and $Q$ is an arbitrary matrix. It is a well-known result that such matrices form a linear sub-algebra $\mathcal{A}$, i.e., they are closed under addition, scalar multiplication, and matrix multiplication. This means that applying any polynomial function to these matrices will yield another matrix of the same form. And since we're calculating the matrix sign function with Newton-Schulz iteration, which is a composition of polynomial functions, its result must also be of the same form.

Another neat property we can take advantage of is that flipping the signs of the anti-diagonal blocks gets preserved under application of matrix polynomials.

> **Proposition 3 (Parity w.r.t. $Q \to -Q$ when applying matrix polynomial $\texttt{p}(\cdot)$)**.
> Let $A \in \mathcal{A}$ such that, $$A = \begin{bmatrix}
    P & Q \\\\
    Q^T & R
\end{bmatrix}$$
> and let,
> $$\begin{bmatrix}
    \widetilde{P} & \widetilde{Q} \\\\
    \widetilde{Q}^T & \widetilde{R}
\end{bmatrix} = \texttt{p}(A) = \texttt{p}\left(\begin{bmatrix}
    P & Q \\\\
    Q^T & R
\end{bmatrix}\right).$$
> Then,
> $$\begin{bmatrix}
    \widetilde{P} & -\widetilde{Q} \\\\
    -\widetilde{Q}^T & \widetilde{R}
\end{bmatrix} = \texttt{p}\left(\begin{bmatrix}
    P & -Q \\\\
    -Q^T & R
\end{bmatrix}\right).$$

> **Crux of the proof:** Flipping the sign of the anti-diagonal blocks gets preserved under addition, scalar multiplication, and matrix multiplication, $$\begin{bmatrix}
    1 & -1 \\\\
    -1 & 1
\end{bmatrix}\begin{bmatrix}
    1 & -1 \\\\
    -1 & 1
\end{bmatrix}
\equiv \begin{bmatrix}
    1 & -1 \\\\
    -1 & 1
\end{bmatrix}$$

Thus we have,
$$\begin{align}
    \begin{bmatrix}
        P^* & Q^* \\\\
        Q^{*T} & R^{\*}
    \end{bmatrix} &= \texttt{\\_orthogonalize\\_via\\_newton\\_schulz}(I + S) \\\\
    \begin{bmatrix}
        P^{\*} & -Q^{\*} \\\\
        -Q^{\*T} & R^{\*}
    \end{bmatrix} &= \texttt{\\_orthogonalize\\_via\\_newton\\_schulz}(I - S)
\end{align}$$
for some $Q^{\*} \in \mathbb{R}^{m \times n}$ and symmetric $P^{\*} \in \mathbb{R}^{m \times m}$, $R^{\*} \in \mathbb{R}^{n \times n}$. Together with Equation 11, we get,

$$\begin{align}
    \texttt{spectral\\_clip}\_{[-1, 1]}(W) &= \scriptsize\frac{1}{2}\left[\begin{bmatrix}
        I_m & W \\\\
        W^T & I_n
    \end{bmatrix}
    \begin{bmatrix}
        P^{\*} & Q^{\*} \\\\
        Q^{*T} & R^{\*}
    \end{bmatrix} - \begin{bmatrix}
        I_m & -W \\\\
        -W^T & I_n
    \end{bmatrix}
    \begin{bmatrix}
        P^{\*} & -Q^{\*} \\\\
        -Q^{*T} & R^{\*}
    \end{bmatrix}\right]\_{12}\\\\
    &= \scriptsize \frac{1}{2} \left[\begin{bmatrix}
        P^{\*} + WQ^{\*T} & Q^{\*} + WR^{\*} \\\\
        W^TP^{\*}+Q^{\*T} & W^TQ^{\*} + R^{\*}
    \end{bmatrix} - \begin{bmatrix}
        P^{\*} + WQ^{\*T} & -(Q^{\*} + WR^{\*}) \\\\
        -(W^TP^{\*}+Q^{\*T}) & W^TQ^{\*} + R^{\*}
    \end{bmatrix}\right]\_{12}\\\\
    &= \scriptsize\begin{bmatrix}
        0 & Q^{\*} + WR^{\*} \\\\
        W^TP^{\*}+Q^{\*T} & 0
    \end{bmatrix}\_{12} \\\\
    \texttt{spectral\\_clip}\_{[-1, 1]}(W) &= Q^{\*} + WR^{\*}
\end{align}$$

This means that we only need to call `_orthogonalize_via_newton_schulz` once, and simply read off the blocks to compute the final result, leading to massive speedups. Also note that the diagonal blocks in Equation (12) are zero, which is what we expect from Theorem 3.

In JAX, this looks like the following:
```python
def _spectral_clip(W: jax.Array):
    m, n = W.shape
    H = jnp.block([[jnp.eye(m), W], [W.T, jnp.eye(n)]])
    OH = _orthogonalize_via_newton_schulz(H)
    Q, R = OH[:m, m:], OH[m:, m:]
    return Q + W @ R

def spectral_clip(W: jax.Array, sigma_max: float=1.):
    return sigma_max * _spectral_clip(W / sigma_max, 1)
```

And a codegolf version would be,
```python
def spectral_clip_minimal(W: jax.Array, sigma_max: float=1., ortho_dtype=jnp.float32):
    OH = _orthogonalize_via_newton_schulz (jnp.block([[jnp.eye(W.shape[0]), W / sigma_max], [W.T / sigma_max, jnp.eye(W.shape[1])]]).astype(ortho_dtype)).astype(W.dtype)
    return sigma_max*OH[:W.shape[0], W.shape[0]:] + W @ OH[W.shape[0]:, W.shape[0]:]
```

### Taking advantage of symmetry

The crux is that since both $I + S$ and $I - S$ are in the sub-algebra $\mathcal{A}$, Newton-Schulz iteration must preserve their block structure. Thus, we do not actually need to materialize the entire $(m + n) \times (m + n)$ block matrices. And note that,

$$\begin{aligned}
    \begin{bmatrix}
        P_i   & Q_i\\\\
        Q_i^T & R_i
    \end{bmatrix}\begin{bmatrix}
        P_j   & Q_j\\\\
        Q_j^T & R_j
    \end{bmatrix}^T &= \begin{bmatrix}
        P_i P_j   + Q_i Q_j^T & P_i Q_j   + Q_i R_j\\\\
        Q_i^T P_j + R_i Q_j^T & Q_i^T Q_j + R_i R_j
    \end{bmatrix}
\end{aligned}$$
Thus we can implement the (blocked) matrix multiplications as,
```pyton
@jax.jit
def block_matmul(
    P1: jax.Array, Q1: jax.Array, R1: jax.Array,
    P2: jax.Array, Q2: jax.Array, R2: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    P = P1 @ P2   + Q1 @ Q2.T
    Q = P1 @ Q2   + Q1 @ R2
    R = Q1.T @ Q2 + R1 @ R2
    return P, Q, R
```
and implement one step of Newton-Schulz iteration as,
```python
def newton_schulz_iter(
    P: jax.Array, Q: jax.Array, R: jax.Array,
    a: float, b: float, c: float,
):
    I_P = a * jnp.eye(P.shape[0], dtype=P.dtype)
    I_R = a * jnp.eye(R.shape[0], dtype=R.dtype)
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)
    Ppoly = I_P + b * P2 + c * P4
    Qpoly =       b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)
```
We then initialize the blocks as $P_0 = I_{m}$, $Q_0 = \pm W$, and $R_0 = I_m$, apply Newton-Schulz iteration as described above to get $(P^\*, Q^\*, R^\*)$, and finally return $Q^\* + WR^\*$. This should give efficiency gains vs. the naive implementation.

## Experimental results [Under Construction]

This section is also still under construction.

[NanoGPT Speedrun results will be added here]

## Acknowledgements

Many thanks to Rohan Anil for initiating a [discussion thread on the topic on Twitter](https://x.com/_arohan_/status/1929945590366122037), and to Arthur Breitman, You Jiacheng, and Su Jianlin for [productive](https://x.com/ArthurB/status/1929958284754330007) [discussions](https://x.com/YouJiacheng/status/1931029612102078749) on [the topic](https://kexue.fm/archives/11006).

## How to Cite

```bibtex
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Numerically Stable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
```

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
2. Higham, Nicholas J. (2008). Functions of Matrices: Theory and Computation. SIAM.
3. Jianlin Su (2025). Calculation of spectral\\_clip (singular value clipping) via msign. Available at: https://kexue.fm/archives/11006
4. Jiacheng You (2025). On a more efficient way to compute spectral clipping via nested matrix sign functions. Available at: https://x.com/YouJiacheng/status/1931029612102078749
5. Arthur Breitman (2025). On using the matrix sign function for spectral clipping. Available at: https://x.com/ArthurB/status/1929958284754330007
