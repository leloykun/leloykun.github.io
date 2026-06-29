---
title: "Heuristic Solutions for Steepest Descent on the Stiefel Manifold"
date: 2025-07-18
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "What would Muon look like if we constrained the weights to be semi-orthogonal?"
summary: "What would Muon look like if we constrained the weights to be semi-orthogonal?"
cover:
    image: steepest-descent-stiefel-manifold.jpg
    alt: "Cover"
    relative: true
editPost:
    URL: "https://x.com/leloykun/status/1953821883482456446"
    Text: "Crossposted on X (formerly Twitter)"
---

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

## 1. Recap: Muon as RMS-to-RMS norm-constrained steepest descent

Consider a weight matrix $W \in \mathbb{R}^{m \times n}$ and a "raw gradient" or differential $G \in \mathbb{R}^{m \times n}$ we get via e.g., backpropagation. In standard gradient descent, we would update the weights as follows,
$$W \leftarrow W - \eta G,$$
where $\eta \in (0, \infty)$ is the learning rate. However, this is suboptimal because (1) the update sizes $\| G \|$ could vary wildly across steps thereby causing training instability, and (2) as we discussed in a [previous blog post on (non-)Riemannian steepest descent](../steepest-descent-non-riemannian/), it does not take into account the matrix structure of the weights. In particular, it ignores how activations or "features" evolve through the network and how the model behaves as we scale it up. Tl;dr:
> If we want the Euclidean norm $\| \cdot \|_2$ of our features and feature updates to 'grow' with the model size,
> then the *Spectral norm* $\| \cdot \|_{2 \to 2}$ of our weights and weight updates must also 'grow' with the model size.

Equivalently, following Yang et al. (2024), we can use the "natural" feature norm, the RMS norm $\|\cdot\|_{RMS} = \frac{1}{\sqrt{n}}\|\cdot\|_{2}$, and the "natural" weight norm, the RMS-to-RMS norm $\|\cdot\|_{RMS \to RMS} = \frac{\sqrt{n}}{\sqrt{m}}\| \cdot \|_{2 \to 2}$, and rephrase the above as,
> If we want the "natural" norm of our features and feature updates to be stable regardless of the model size,
> then the "natural" norm of our weights and weight updates must also be stable regardless of the model size.

We will discuss weight norm controls in the next section, but for now, instead of using the raw gradient $G$, we can instead try to find a descent direction $A \in \mathbb{R}^{m \times n}$ that is maximally aligned to $G$ while satisfying our weight update condition above,
    $$\| A \|_{RMS \to RMS} = \frac{\sqrt{n}}{\sqrt{m}}\| A \|_{2 \to 2} = \text{constant}.$$
Thus our update rule becomes,
    $$W \leftarrow W - \eta \frac{\sqrt{m}}{\sqrt{n}} A^*,$$
where
$$\begin{equation}
    A^* = \arg\max_{A\in \mathbb{R}^{m \times n}:\| A \|_{2 \to 2} = 1} \langle G, A \rangle,
\end{equation}$$
and $\langle \cdot, \cdot \rangle$ is the Frobenius inner product which measures the "alignment" between two matrices. From Bernstein & Newhouse (2024), this has a closed-form solution,
$$A^* = \operatorname{msign}(G),$$
where $\operatorname{msign}(\cdot)$ is the matrix sign function. And finally, adding a momentum term then yields the Muon optimizer (Jordan et al., 2024),
$$\begin{align*}
    M_t &= \beta M_{t-1} + (1 - \beta) G \\
    W_t &= W_{t-1} - \eta\frac{\sqrt{m}}{\sqrt{n}} \operatorname{msign}(M_t), \\
\end{align*}$$
for some momentum hyperparameter $\beta \in [0, 1)$.

> If you want to learn more about Muon and the ideas behind it, check out [Newhouse's 3-part blog series](https://www.lakernewhouse.com/writing/muon-1). I highly recommend it!

### 1.1. Recap: (non-)Riemannian steepest descent

An update step in first-order optimization on a manifold $\mathcal{M}$ goes as follows,
1. Compute an 'optimal' descent direction $A^*$ in the tangent space at the current point $W_t \in \mathcal{M}$, $A^* \in T_{W_t} \mathcal{M}$.
2. Use this to 'move' our weight $\widetilde{W}_{t+1} \leftarrow W_t - \eta A^*$, where $\eta$ is the learning rate. Note that $\widetilde{W}_{t+1}$ may not be on the manifold $\mathcal{M}$. And so,
3. Retract the result back to the manifold via a retraction map $W_{t+1} \leftarrow \operatorname{retract}(\widetilde{W}_{t+1})$.

We then repeat this process until convergence or until we find a satisfactory solution.

An important detail discussed by Large et al. (2024) and in the [previous blog post on (non-)Riemannian optimization](../steepest-descent-non-riemannian/) is that the so-called "raw gradient" $G$ we get via backpropagation is *not* actually in the tangent space, but rather in the *co*tangent space at $W$, $G \in T_W^* \mathcal{M}$, or the space of *linear functionals* acting on the tangent vectors. $G$ then is useless by itself. To make it useful, we need to map it to the tangent space first via a dualizer map, $\operatorname{dualizer}: T_W^* \mathcal{M} \mapsto T_W \mathcal{M},$
$$A^* = \operatorname{dualizer}(G) = \arg\max_{A \in T_W \mathcal{M}} \langle G, A \rangle,$$
where the $\langle \cdot, \cdot \rangle$ operation is the canonical pairing between tangent and cotangent spaces. It is technically not an inner product, but *behaves like* the Frobenius inner product.

In Euclidean space, we got lucky: $T_W \mathbb{R}^{m \times n} = T_W^* \mathbb{R}^{m \times n} = \mathbb{R}^{m \times n}$, and thus, $A^* = G$, yielding the update rule for (stochastic) gradient descent (SGD). In Riemannian manifolds, which we get by e.g. equipping the tangent spaces with a Riemannian metric (or a norm that induces such a metric), the two spaces are no longer equivalent, but they are congruent. This means that for every $G \in T_W^* \mathcal{M}$, there exists a *unique* steepest descent direction $A^* \in T_W \mathcal{M}$ we can follow to minimize the loss and vice versa. In non-Riemannian manifolds, however, the optimal $A^*$ may no longer be unique or may not even exist.

Muon then is what we get when we equip the tangent spaces of $\mathcal{M} = \mathbb{R}^{m \times n}$ with the RMS-to-RMS norm, $\| \cdot \|_{RMS \to RMS}$. So while the underlying space is still Euclidean, the change in how we measure 'distances' makes the new manifold non-Euclidean and even non-Riemannian. In the next sections, we discuss how to build Muon-like optimizers for more exotic manifolds and demonstrate how a smart choice of manifolds to 'place' our weights in can accelerate generalization.

## 2. Spectral norm-constrained steepest descent on the Stiefel manifold

As discussed in the previous section, we not only need to control the weight update norms but also the weight norms themselves. There are multiple ways to do this and we presented some novel methods in our recent work on [training transformers with enforced Lipschitz bounds](https://arxiv.org/abs/2507.13338) (Newhouse*, Hess*, Cesista* et al., 2025). However, here we will focus on constraining the weights to be semi-orthogonal, i.e.,
$$\begin{equation} W^\top W = I_n. \end{equation}$$

Semi-orthogonal matrices lie on the Stiefel manifold, $\text{St}(m, n) = \{W \in \mathbb{R}^{m \times n} \| W^\top W = I_n \}$. Differentiating Equation (2) on both sides then yields the constraint that determines membership in the tangent space at $W \in \text{St}(m, n)$,
$$T_W \text{St}(m, n) = \{A \in \mathbb{R}^{m \times n} \| W^\top A + A^\top W = 0\}.$$
But a crucial difference from prior work on optimization on the Stiefel manifold (Ablin & Peyré, 2021; Gao et al., 2022) is that we equip the tangent spaces with the spectral norm, $\| \cdot \|_{2 \to 2}$, augmenting the Stiefel manifold with a [Finsler structure](https://en.wikipedia.org/wiki/Finsler_manifold). With this, our dualizer becomes,
$$A^* = \arg\max_{A\in \mathbb{R}^{m \times n}: \| A \|_{2 \to 2} = 1} \langle G, A \rangle \quad \text{s.t. } A \in T_W\text{St}(m, n),$$
and using $\operatorname{msign}(\cdot)$ as the retraction map, our update rule becomes,
$$\begin{equation}
    W \leftarrow \operatorname{msign}\left(W - \eta A^* \right)
\end{equation}$$

Equivalently,
$$A^* = \arg\max_{A\in \mathbb{R}^{m \times n}} \langle G, A \rangle  \quad \text{s.t. } A \in \text{St}(m, n) \cap T_W \text{St}(m, n),$$
Or in words, we want to find a descent direction $A$ that is both on the Stiefel manifold and in the tangent space at the current point $W \in \text{St}(m, n)$ that maximizes the "alignment" with the raw gradient $G$. 

### 2.1. RMS-to-RMS norm-constrained steepest descent on the (scaled) Stiefel manifold

Following the natural norm conditions we discussed in the previous section, we may want to constrain our weights to be semi-orthogonal *with respect to* the RMS-to-RMS norm, i.e.,
$$W^\top W = \frac{m}{n}I_n.$$
This places our weights on the scaled Stiefel manifold, $\widetilde{\text{St}}(m, n) = \{W \in \mathbb{R}^{m \times n} \| W^\top W = s^2 I_n \}$ with scale $s = \sqrt{m}/\sqrt{n}$. We can use the same dualizer map as for the unscaled Stiefel manifold, but our update rule becomes,
$$\begin{equation}
    W \leftarrow \frac{\sqrt{m}}{\sqrt{n}} \operatorname{msign}\left(W - \eta \frac{\sqrt{m}}{\sqrt{n}} A^* \right)
\end{equation}$$

### 2.2. Retraction via rescaling

Recall that $\operatorname{msign}(X) = X (X^\top X)^{-1/2}$ and note that,

$$\begin{align*}
    (W - \eta A^*)^\top (W - \eta A^*)
        &= \underbrace{W^\top W}_{=I_n} - \eta \underbrace{((A^*)^\top W + W^\top A^*)}_{=0} + \eta^2 \underbrace{(A^*)^\top A^*}_{=I_n} \\
        &= (1 + \eta^2) I_n
\end{align*}$$

Thus we can rewrite the update rule for steepest descent on the unscaled Stiefel manifold as,
$$W \leftarrow \frac{W - \eta A^*}{\sqrt{1 + \eta^2}}$$
and for the scaled Stiefel manifold as,
$$W \leftarrow \frac{W - \eta \frac{\sqrt{m}}{\sqrt{n}} A^*}{\sqrt{1 + \eta^2}}$$

## 3. Equivalence between Bernstein's and Su's solutions

Bernstein (2025a) and Su (2025) found the following solutions to the square and full-rank case,

$$\begin{align*}
    A^*_{\text{bernstein}} &= W \operatorname{msign}(\operatorname{skew}(W^\top G))\\
    A^*_{\text{su}} &= \operatorname{msign}(G - W\operatorname{sym}(W^\top G))
\end{align*}$$
where $\operatorname{sym}(X) = \frac{1}{2}(X + X^\top)$ and $\operatorname{skew}(X) = \frac{1}{2}(X - X^\top)$.

We will show that these are equivalent, i.e., $A^*_{\text{bernstein}} = A^*_{\text{su}}$. For this, we will reuse the following proposition we discussed in a [previous post on spectral clipping](../spectral-clipping).

> **Proposition 1 (Transpose Equivariance and Unitary Multiplication Equivariance of Odd Matrix Functions)**. Let $W \in \mathbb{R}^{m \times n}$ and $W = U \Sigma V^\top$ be its reduced SVD. And let $f: \mathbb{R}^{m \times n} \to \mathbb{R}^{m \times n}$ be an odd analytic matrix function that acts on the singular values of $W$ as follows,
> $$f(W) = U f(\Sigma) V^\top.$$
> Then $f$ is equivariant under transposition and unitary multiplication, i.e.,
> $$\begin{align*}
    f(W^\top) &= f(W)^\top \\
    f(WQ^\top) &= f(W)Q^\top \quad\forall Q \in \mathbb{R}^{m \times n} \text{ such that } Q^\top Q = I_n \\
    f(Q^\top W) &= Q^\top f(W) \quad\forall Q \in \mathbb{R}^{m \times n} \text{ such that } QQ^\top = I_m
\end{align*}$$

Thus,

$$\begin{align*}
    A^*_{\text{bernstein}}
        &= W \operatorname{msign}(\operatorname{skew}(W^\top G)) \\
        &= \operatorname{msign}(W \operatorname{skew}(W^\top G)) &\text{(from Proposition 1)} \\
        &= \operatorname{msign}\left(\frac{1}{2}W W^\top G - \frac{1}{2}W G^\top W \right) \\
        &= \operatorname{msign}\left(W W^\top G - \frac{1}{2}W W^\top G - \frac{1}{2}W G^\top W \right) \\
        &= \operatorname{msign}\left(G - W\operatorname{sym}(W^\top G) \right) \\
    A^*_{\text{bernstein}} &= A^*_{\text{su}}
\end{align*}$$
where the second-to-last equality relies on $W$ being square and full-rank, which then implies that $W W^\top = I_m$.

## 4. Projection-projection perspective

One can interpret Bernstein's and Su's solutions as a two-step projection process:

1. (Orthogonal) projection to the tangent space at $W$; and
2. Projection to the closest semi-orthogonal matrix (i.e., closest point on the Stiefel manifold).

This is because the map $G \to G - W \operatorname{sym}(W^\top G)$ is actually the orthogonal projection onto the tangent space at $W$ and that $\operatorname{msign}$ projects the resulting matrix to the Stiefel manifold. We present these more rigorously as follows.

> **Theorem 2 (Orthogonal projection to the tangent space at $W$).** Let $W \in \text{St}(m, n)$ be a point on the Stiefel manifold. The projection of a vector $V \in \mathbb{R}^{m \times n}$ onto the tangent space at $W$, denoted as $T_W\text{St}(m, n)$ , is given by,
> $$\begin{equation}
    \operatorname{proj}_{T_W\text{St}(m, n)}(V) = V - W \operatorname{sym}(W^\top V)
\end{equation}$$

{{< collapse summary="Show **proof of Theorem 2**" openByDefault=false >}}
> **Proof.** First, we need to show that the normal space at $W$, $N_W\text{St}(m, n)$ is given by,
> $$N_W\text{St}(m, n) = \{ WS \| S = S^\top \}$$
> for symmetric $S$. To show this, let $A \in T_W\text{St}(m, n)$ be an arbitrary tangent vector at $W$. Then we have,
> $$\begin{align*}
    \langle A, WS \rangle &= \operatorname{tr}(A^\top WS) \\
        &= \operatorname{tr}(S W^\top A) &\text{(transpose invariance of trace)} \\
        &= -\operatorname{tr}(S A^\top W) &\text{(since $A \in T_W\text{St}(m, n)$)} \\
        &= -\operatorname{tr}(A^\top W S) &\text{(cyclic property of trace)} \\
    \langle A, WS \rangle &= -\langle A, WS \rangle
\end{align*}$$
> Thus $\langle A, WS \rangle = 0$ which implies that $A$ and $WS$ are orthogonal. Hence $WS \in N_W\text{St}(m, n)$.
>
> Now, for any $V \in \mathbb{R}^{m \times n}$, we can write it as,
> $V = \underbrace{V - WS}_{\text{candidate tangent}} + \underbrace{WS}_{\text{candidate normal}}$ for some symmetric $S$. To find $S$,
> $$\begin{align*}
    W^\top (V - WS) + (V - WS)^\top W &= 0 \\
    W^\top V - S + V^\top W - S &= 0 \\
    S &= \operatorname{sym}(W^\top V)
\end{align*}$$
> Thus, $V - W \operatorname{sym}(W^\top V) \in T_W\text{St}(m, n)$. And because of that, $W^\top (V - W \operatorname{sym}(W^\top V))$ must be skew-symmetric. Thus,
> $$\begin{equation*}
    \langle V - WS, WS \rangle
        = \langle \underbrace{W^\top (V - WS)}_{\text{skew-symmetric}}, \underbrace{S}_{\text{symmetric}} \rangle
        = 0
\end{equation*}$$
> Hence, $V - W \operatorname{sym}(W^\top V)$ is the orthogonal projection of $V$ onto the tangent space at $W$.

{{< /collapse >}}

And from Proposition 4 of Bernstein & Newhouse (2024),

> **Proposition 3 (Projection to the closest semi-orthogonal matrix).** Consider the orthogonal matrices $\mathcal{O}_{m \times n} := \{ A \in \mathbb{R}^{m \times n} : A A^\top = I_m or A^\top A = I_n \}$ and let $\| \cdot \|_F$ denote the Frobenius norm. For any matrix $G \in R^{m \times n}$ with reduced SVD $G = U \Sigma V^\top$:
> $$\arg\min_{A \in \mathcal{O}_{m \times n}} \| A - G \|_F = \operatorname{msign}(G) = UV^\top,$$
> where the minimizer $U V^\top$ is unique if and only if the matrix $G$ has full rank.

Thus we can write,
$$A^*_{\text{bernstein}} = A^*_{\text{su}} = (\operatorname{proj}_{\text{St}(m, n)} \circ \operatorname{proj}_{T_W\text{St}(m, n)})(G)$$
for the square and full-rank case at least.

### 4.1. Why Bernstein's & Su's solutions only work for the square and full-rank case

Note that the projections above aim to guarantee both of our criteria for the solution, but one step at a time. And that the $\operatorname{msign}$ after the projection may send the resulting matrix outside the tangent space at $W$. We show that this is not a problem in the square and full-rank case, but it is in the general case.

| operation                                                |              on the tangent space at $W$? |    have unit spectral norm? |
| -------------------------------------------------------- | ----------------------------------------: | --------------------------: |
| (input $G$)                                              |                            not in general |              not in general |
| 1st projection ($\operatorname{proj}_{T_W St(m, n)}(\cdot)$) |               ${\color{green}\text{yes}}$ |             not necessarily |
| 2nd projection ($\operatorname{msign}(\cdot$))                 | only for the square<br>and full-rank case | ${\color{green}\text{yes}}$ |

To demonstrate this, we will need the following proposition.

> **Proposition 4 ($\operatorname{msign}$ preserves skew-symmetry).** Let $X \in \mathbb{R}^{m \times n}$ be a skew-symmetric matrix. Then $\operatorname{msign}(X)$ is also skew-symmetric.

The proof follows directly from the transpose equivariance and oddness of $\operatorname{msign}$, i.e., $\operatorname{msign}(X) = \operatorname{msign}(-X^\top) = -\operatorname{msign}(X)^\top$.

Also note that for the general case,
$$\begin{equation}
    W W^\top + Q Q^\top = I_m \quad\text{and}\quad W^\top Q = 0
\end{equation}$$
where $Q$ is the orthonormal complement of $W$.

Thus,
$$\begin{align*}
    (\operatorname{proj}_{\text{St}(m, n)} \circ \operatorname{proj}_{T_W St(m, n)})(G)
        &= \operatorname{msign}(G - \frac{1}{2}W W^\top G - \frac{1}{2} W G^\top W) \\
        &= \operatorname{msign}((W W^\top + Q Q^\top)G - \frac{1}{2}W W^\top G - \frac{1}{2} W G^\top W) \\
        &= \operatorname{msign}(W \operatorname{skew}(W^\top G) + Q Q^\top G)
\end{align*}$$
For the square and full-rank case, we have $Q = 0$ and $W W^\top = I$. And the above simplifies to Berstein's solution,
$$(\operatorname{proj}_{\text{St}(m, n)} \circ \operatorname{proj}_{T_W St(m, n)})(G) = W \operatorname{msign}(\operatorname{skew}(W^\top G))$$
and since $\operatorname{skew}(W^\top G)$ is skew-symmetric, then $\operatorname{msign}(\operatorname{skew}(W^\top G))$ must be too. And thus,
$$\begin{align*}
    &W^\top (\operatorname{proj}_{\text{St}(m, n)} \circ \operatorname{proj}_{T_W St(m, n)})(G) + (\operatorname{proj}_{\text{St}(m, n)} \circ \operatorname{proj}_{T_W St(m, n)})(G)^\top W \\
        &\qquad= W^\top W \operatorname{msign}(\operatorname{skew}(W^\top G)) + (W\operatorname{msign}(\operatorname{skew}(W^\top G)))^\top W \\
        &\qquad= \operatorname{msign}(\operatorname{skew}(W^\top G)) + \operatorname{msign}(\operatorname{skew}(W^\top G))^\top \\
        &\qquad= 0
\end{align*}$$
Hence for the square and full-rank case, this two-step projection process guarantees that the resulting matrix has unit spectral norm *and* is in the tangent space at $W$.

For the more general case, we may no longer have $Q = 0$ or $W W^\top = I$. Thus, we cannot guarantee that $W \operatorname{skew}(W^\top G) + QQ^\top G$ is skew-symmetric. And so the $\operatorname{msign}$ after the first projection may send the resulting matrix outside the tangent space at $W$.

## 5. Heuristic solutions for the general case

### 5.1. Alignment upper bound

As established previously, for any $G \in \mathbb{R}^{m \times n}$, the solution to $\arg\max_{A \in \text{St}(m, n)} \langle G, A \rangle$ is $A^* = \operatorname{proj}_{\text{St}(m, n)}(G) = \operatorname{msign}(G)$ and thus $\max_{A \in \text{St}(m, n)} \langle G, A \rangle = \| G \|_{\text{nuc}}$ where $\| \cdot \|_{\text{nuc}}$ is the nuclear norm. With an extra linear constraint $A \in T_W \text{St}(m, n)$, we have,

$$\begin{align*}
    \max_{A \in \text{St}(m, n) \cap T_W \text{St}(m, n)} \langle G, A \rangle
        &= \max_{A \in \text{St}(m, n) \cap T_W \text{St}(m, n)} \langle \underbrace{G - \operatorname{proj}_{T_W \text{St}(m, n)}(G)}_{\in N_W\text{St}(m, n)} + \operatorname{proj}_{T_W \text{St}(m, n)}(G), A \rangle \\
        &= \max_{A \in \text{St}(m, n) \cap T_W \text{St}(m, n)} \langle \operatorname{proj}_{T_W \text{St}(m, n)}(G), A \rangle \\
        &\leq \max_{A \in \text{St}(m, n)} \langle \operatorname{proj}_{T_W \text{St}(m, n)}(G), A \rangle \\
    \max_{A \in \text{St}(m, n) \cap T_W \text{St}(m, n)} \langle G, A \rangle
        &\leq \| \operatorname{proj}_{T_W \text{St}(m, n)}(G) \|_{\text{nuc}}
\end{align*}$$

The alignment between an element in the normal space and an element in the tangent space is always zero, i.e., $\langle \operatorname{proj}_{T_W \text{St}(m, n)}(G), \operatorname{proj}_{N_W \text{St}(m, n)}(G) \rangle = 0$. Thus, we can cancel out the first term in the first equality. And we have first inequality because the max over a smaller set is less than or equal to the max over a larger set.

We achieve equality in the square and full-rank case because the maximizer for the first inequality is guaranteed to be in the tangent space at $W$, as discussed in the previous section.

### 5.2. Fixed-point iteration of alternating projections

Notice that $QQ^\top G$ is the projection of $G$ onto the column space of $Q$, $\operatorname{proj}_{\operatorname{col}(Q)}(G) = QQ^\top G$. One can think of this as the component of $G$ that is, in a sense, *not* "aligned to" $W$. In practice, this is typically small relative to the component of $G$ that *is* "aligned to" $W$. If so, then,
$$(\operatorname{proj}_{\text{St}(m, n)} \circ \operatorname{proj}_{T_W St(m, n)})(G) \approx \operatorname{msign}(W \operatorname{skew}(W^\top G) + \cancel{\operatorname{proj}_{\operatorname{col}(Q)}(G)}),$$
which means that while the resulting matrix after the two projections may not be in the tangent space at $W$, it would likely be *nearby*. And repeating this process a few times should close the gap.

Sample implementation:
```python
def project_to_stiefel_tangent_space(X, delta_X):
    return delta_X - X @ sym(X.T @ delta_X)

def orthogonalize(X):
    # copy Newton-Schulz iteration from Muon (Jordan et al., 2024)

def steepest_descent_stiefel_manifold_heuristic(W, G, num_steps=1):
    assert num_steps > 0, "Number of steps must be positive"
    A_star = G
    for _ in range(num_steps):
        A_star = project_to_stiefel_tangent_space(W, A_star)
        A_star = orthogonalize(A_star)
    return A_star
```

#### 5.2.1. Local (linear) convergence guarantee

For now, we cannot yet guarantee global convergence as it would potentially require deriving the Lyapunov function for this iteration which is extremely difficult. What we can guarantee, however, is *local* convergence due to $\text{St}(m, n)$ and $T_W\text{St}(m, n)$ being closed semi-algebraic sets. That is, assuming that the initial point $W$ is "close enough" to the intersection, we can guarantee that a subsequence of the iterates converges to a point in the intersection. Furthermore, assuming transversality, we can guarantee that the convergence is linear.

> **Definition 5 (Semi-algebraic set)** Let $\mathbb{F}$ be a real closed field. A subset $S$ of $\mathbb{F}^n$ is a semi-algebraic set if it is a finite union of sets defined by polynomial equations and inequalities.

We can construct $\text{St}(m, n) = \{ W \in \mathbb{R}^{m \times n} \| W^\top W = I_n \}$ via $n(n+1)/2$ polynomial equations, and $T_W\text{St}(m, n) = \{ A \in \mathbb{R}^{m \times n} \| W^\top A + A^\top W = 0 \}$ via $mn$ polynomial equations. Hence both are semi-algebraic sets. And the intersection of two semi-algebraic sets is also a semi-algebraic set. From Theorem 7.3 of Drusvyatskiy et al. (2016), we then have the following convergence guarantee.

> **Theorem 6 (Convergence of alternating projections on semi-algebraic sets)** Consider two nonempty closed semi-algebraic sets $X, Y \subset E$ with $X$ bounded. If the method of alternating projections starts in $Y$ and near $X$, then the distance of the iterates to the intersection $X \cap Y$ converges to zero, and hence every limit point lies in $X \cap Y$.

Setting $Y = T_W\text{St}(m, n)$, $X = \text{St}(m, n)$, and noting that the Stiefel manifold is bounded, we have that the iterates of our method of alternating projections converge to a point in the intersection $\text{St}(m, n) \cap T_W\text{St}(m, n)$.

But does this algorithm converge in sufficient time? Somewhat yes, we can guarantee local *linear* convergence assuming transversality.

> **Definition 7 (Transversality)** Let $\mathcal{M}$ and $\mathcal{N}$ be two (smooth) manifolds in a Euclidean space. We say that $\mathcal{M}$ and $\mathcal{N}$ intersect transversally at a point $x \in \mathcal{M} \cap \mathcal{N}$ when,
> $$N_x\mathcal{M} \cap N_x\mathcal{N} = \{0\}.$$

Intuitively speaking, this means that the tangent spaces at the intersection of the two manifolds have an "angle" between them, i.e., they are not parallel. The larger this angle is, the faster the convergence; and the smaller the angle, the slower the convergence. Theorem 2.1 of Drusvyatskiy et al. (2016) then gives us the following local linear convergence guarantee.

> **Theorem 8 (Linear convergence of alternating projections, assuming transversality)** If two closed sets in a Euclidean space intersect transversally at a point $\tilde{x}$, then the method of alternating projections, started nearby, converges linearly to a point in the intersection.

$\text{St}(m, n)$ and $T_W\text{St}(m, n)$ are both closed sets so the theorem above applies.

### 5.3. Ternary search over nearby feasible solutions

Here we present an alternative solution that is more efficient, but often yields slightly more suboptimal results than the fixed-point iteration method above.

#### 5.3.1. Problem decomposition

The crux is to split $\arg\max_{A\in \mathbb{R}^{m \times n}} \langle G, A \rangle$ into two optimization problems, one for the component of $G$ that is "aligned to" $W$ and one for the component of $G$ that is "not aligned to" $W$. To see this, let us first decompose $G$ and $A$ into,
$$G = W G_W + Q G_Q \qquad A = WB + QC$$
where $G_W = W^\top G$ and $G_Q = Q^\top G$. Thus,
$$\begin{align}
    \langle G, A \rangle
        &= \langle W G_W + Q G_Q, WB + QC \rangle \nonumber \\
        &= \operatorname{tr}((W G_W + Q G_Q)^\top (WB + QC)) \nonumber \\
        &= \operatorname{tr}(G_W^\top B + G_Q^\top C) \nonumber \\
    \langle G, A \rangle
        &= \langle G_W, B \rangle + \langle G_Q, C \rangle \\
\end{align}$$
The cross terms vanish in the third equality because $W^\top Q = 0$. Finding the maximizer $A^*$ for the LHS is then equivalent to finding the maximizers $B^*$ and $C^*$ for the RHS and then combining them,
$$A^* = WB^* + QC^*.$$

#### 5.3.2. Solving the two subproblems

Now, to satisfy the constraint $A \in T_W\text{St}(m, n) \cap \text{St}(m, n)$,
$$\begin{align*}
    W^\top A + A^\top W &= 0 \qquad\qquad& A^\top A &= I_n \\
    W^\top (WB + QC) + (WB + QC)^\top W &= 0 \qquad\qquad& (WB + QC)^\top (WB + QC) &= I_n \\
    B + B^\top &= 0 \qquad\qquad& B^\top B + C^\top C &= I_n \\
\end{align*}$$
That is, we require that $B$ is skew-symmetric and $C$ satisfies $C^\top C = I_n - B^\top B$.

We cannot simply treat each subproblem separately because the second constraint couples $B$ and $C$. However, for this approximation, we make the assumption that doing so would yield a "good enough" solution.

---

For the first term, we can decompose $G_W$ into its skew-symmetric and symmetric components, $G_W = \operatorname{skew}(G_W) + \operatorname{sym}(G_W)$,
$$\begin{align*}
    \arg\max_{B \text{ is skew}} \langle G_W, B \rangle
        &= \arg\max_{B \text{ is skew}} \langle \operatorname{skew}(G_W) + \operatorname{sym}(G_W), B \rangle \\
        &= \arg\max_{B \text{ is skew}} \langle \operatorname{skew}(G_W), B \rangle + \cancel{\langle \operatorname{sym}(G_W), B \rangle}
\end{align*}$$
And the maximizer is simply $\tilde{B} = \operatorname{skew}(G_W) = \operatorname{skew}(W^\top G)$.

However, because of the constraint $B^\top B + C^\top C = I$, we have to "cap" the spectral norm of $B$ to be less than or equal to 1 otherwise we would fail to construct a real-valued $C$. We can do this via a variety of methods discussed in a [previous post on spectral clipping](../spectral-clipping/) and our latest paper on [training transformers with enforced Lipschitz bounds](https://arxiv.org/abs/2507.13338) (Newhouse*, Hess*, Cesista* et al., 2025). Let $\tau \leq 1$ be the spectral norm bound, then we can choose,

$$B^*(\tau) := \operatorname{hardcap}_{\tau}(\tilde{B})
    \quad\text{or}\quad B^*(\tau) := \tau\cdot\operatorname{msign}(\tilde{B})
    \quad\text{or}\quad B^*(\tau) := \frac{\tau}{\| \tilde{B} \|_{2 \to 2}}\tilde{B}$$

These mappings preserve skew-symmetry and thus $B^*(\tau)$ satisfies the first constraint.

---

Now, parametrize $C$ as $C = UR$ where $U^\top U = I$ and $R(\tau) = (I_n - (B^*(\tau))^\top B^*(\tau))^{1/2}$. It is trivial to check that $C$ satisfies our constraints and that $R(\tau)$ is SPD. Thus, assuming we already have a fixed $B^*(\tau)$ (and consequently a fixed $R(\tau)$), solving the second subproblem,
$$\arg\max_{C} \langle G_Q, C \rangle$$
is equivalent to solving,
$$\begin{align*}
    \arg\max_{U: U^\top U = I} \langle G_Q, U R(\tau) \rangle
        &= \arg\max_{U: U^\top U = I} \operatorname{tr}(G_Q^\top U R(\tau)) \\
        &= \arg\max_{U: U^\top U = I} \operatorname{tr}(R(\tau) G_Q^\top U) \\
        &= \arg\max_{U: U^\top U = I} \langle G_Q R(\tau), U \rangle
\end{align*}$$
which has maximizer $U^* = \operatorname{msign}(G_Q R(\tau))$. Thus, $C^*(\tau) = \operatorname{msign}(G_Q R(\tau)) R(\tau)$.

---

Note that for the square and full-rank case, we have $C = 0$ and so we require $B^\top B = I$. For this, we can choose to orthogonalize $B = \operatorname{skew}(G_W)$ which then yields Bernstein's solution,
$$A^*_{\text{bernstein}} = W \operatorname{msign}(\operatorname{skew}(W^\top G))$$
This also motivates the choice of $\operatorname{msign}$ as the normalization method for $B$ more generally.

We can implement this in JAX as follows,

```python
from spectral_clipping import spectral_hardcap, spectral_normalize, orthogonalize

def matsqrt(W: jax.Array):
    # We can also compute this via Newton-Schulz iteration or Cholesky decomposition
    U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
    s_sqrt = jnp.sqrt(s)
    return U @ jnp.diag(s_sqrt) @ Vh

def construct_nearby_feasible_solution(W, Q, G, tau=0.5, normalizer_method=0):
    m, n = W.shape
    if m == n:
        # assumes full-rank
        A = W @ orthogonalize(skew(W.T @ G))  # Bernstein's solution
    else:
        B = skew(W.T @ G)
        if normalizer_method == 0:
            B_tilde = tau * orthogonalize(B)
        elif normalizer_method == 1:
            B_tilde = spectral_hardcap(B, tau)
        elif normalizer_method == 2:
            B_tilde = spectral_normalize(B, tau)
        R = jnp.linalg.cholesky(jnp.eye(n) - B_tilde.T @ B_tilde)
        # R = matsqrt(jnp.eye(n) - B_tilde.T @ B_tilde)
        C = orthogonalize(Q.T @ G @ R) @ R
        A = W @ B_tilde + Q @ C
    return A
```

#### 5.3.1. Where ternary search comes in

![](alignment-unimodal.png#center)

From Equation (7), we have,

$$\begin{align*}
    f(\tau) := \langle G, A^*(\tau) \rangle
        &= \langle G_W, B^*(\tau) \rangle + \langle G_Q, C^*(\tau) \rangle \\
        &= \langle \operatorname{skew}(G_W), B^*(\tau) \rangle + \langle G_Q, C^*(\tau) \rangle \\
        &= \langle \operatorname{skew}(G_W), \operatorname{normalized}_{\tau}(\operatorname{skew}(G_W)) \rangle + \langle G_Q, \operatorname{msign}(G_Q R(\tau)) R(\tau) \rangle \\
        &= \langle \operatorname{skew}(G_W), \operatorname{normalized}_{\tau}(\operatorname{skew}(G_W)) \rangle + \| G_QR(\tau) \|_{\text{nuc}} \\
\end{align*}$$

The first term is linear and positively-sloped as we vary $\tau$. And since the map $x \mapsto \sqrt{1 - x^2}$ is concave and non-increasing in $x \in [0, 1]$, the second term must be concave and non-increasing as we increase $\tau$. Taken together, $f$ must be unimodal. And thus we can use ternary search to find the optimal $\tau$ that maximizes $f(\tau)$.

We can implement this in JAX as follows,

```python
def ternary_search_over_taus(W, Q, G, lo=0., hi=1., normalizer_method=0, max_iter=10):
    def evaluate(tau):
        A = construct_nearby_feasible_solution(W, Q, G, tau, normalizer_method)
        return jnp.trace(G.T @ A)

    def body_fun(i, val):
        lo, hi = val
        mid1 = (2*lo + hi) / 3
        mid2 = (lo + 2*hi) / 3
        f_mid1 = evaluate(mid1)
        f_mid2 = evaluate(mid2)
        new_lo = jnp.where(f_mid1 > f_mid2, lo, mid1)
        new_hi = jnp.where(f_mid1 > f_mid2, mid2, hi)        
        return new_lo, new_hi
    final_lo, final_hi = jax.lax.fori_loop(0, max_iter, body_fun, (lo, hi))

    # Compute final midpoint and its value
    final_tau = (final_lo + final_hi) / 2
    final_value = evaluate(final_tau)
    return final_tau, final_value
```

## 6. Bonus: a Muon-like optimizer for the Embedding and Unembedding layers

Embedding layers have a hidden geometry: the (scaled-)Oblique manifold, $\widetilde{\text{Ob}}(m, n) = \{ W \in \mathbb{R}^{m \times n} \| \operatorname{diag}(W^\top W) = s^2\mathbf{1}_n \}$ with scale $s = \sqrt{m}$, or the manifold of matrices with unit-RMS-norm columns. More precisely, it is the embedding layer *and* the normalization layer right after it that results in unit-RMS-norm feature-vectors. But optimizers like Adam typically ignore this geometry and even its matrix-structure, treating the embedding layer the same as 'flat' vectors. We believe this leads to suboptimal performance and demonstrate this via grokking experiments we discuss in the next section.

What if we build an optimizer that respects this geometry?

For this, we need two things:
1. A 'dualizer' map that maps a "raw gradient" matrix $G \in \mathbb{R}^{m \times n}$ to an update direction of steepest descent on the tangent space at $W \in \widetilde{\text{Ob}}(m, n)$, i.e., $A^* \in T_W\widetilde{\text{Ob}}(m, n)$ with $\| A^* \| = 1$ for some norm $\| \cdot \|$ chosen a priori. And,
2. A 'projection' or retraction map that maps an (updated) weight matrix $W \in \mathbb{R}^{m \times n}$ back to the (scaled-)Oblique manifold.

The retraction map is simply the column-wise normalization,
$$\operatorname{col\_normalize}(W) := \operatorname{col}_j(W) \mapsto \frac{\operatorname{col}_j(W)}{\| \operatorname{col}_j(W) \|_{RMS}} = \sqrt{m}\frac{\operatorname{col}_j(W)}{\| \operatorname{col}_j(W) \|_{2}} \quad \forall 0 \leq j < n$$
where $\operatorname{col}_j(W)$ is the $j$-th column of the weight matrix $W$.

As for the dualizer, which norm should we use? We can, for example, use the RMS-to-RMS norm for consistency and still be able to use the same alternating projection method as before. However, as argued by Bernstein & Newhouse (2024) and Pethick et al. (2024), it may be more natural to use the L1-to-RMS norm, $\| \cdot \|_{1\to RMS}$ because the maximizer for the following problem,
$$\arg\max_{A: \| A \|_{1 \to RMS} = 1} \langle G, A \rangle$$
is simply $\operatorname{col\_normalize}(G) \in \widetilde{\text{Ob}}(m, n)$. That is, all of the token embedding updates would have the same size, improving training stability. Thus our update rule becomes,
$$ W \leftarrow \operatorname{col\_normalize}(W - \eta A^*)$$
where $\eta$ is the learning rate and,
$$ A^* = \arg\max_{A: \| A \|_{1 \to RMS} = 1} \langle G, A \rangle \quad \text{s.t. } A \in T_W\widetilde{\text{Ob}}(m, n),$$

Equivalently,
$$A^* = \arg\max_{A} \langle G, A \rangle  \quad \text{s.t. } A \in \widetilde{\text{Ob}}(m, n) \cap T_W \widetilde{\text{Ob}}(m, n),$$
or in words, we want to find a descent direction $A^*$ that is both on the (scaled-)Oblique manifold and in the tangent space at $W$ that maximizes the alignment with the "raw gradient" $G$.

### 6.1. Optimal solution for steepest descent on the (scaled-)Oblique manifold

The Oblique manifold is a product of hyperspheres, $\text{Ob}(m, n) = \underbrace{S^m \times \ldots \times S^m}_{n}$. So, in a sense, the columns are acting independently of each other and steepest descent on the Oblique manifold is equivalent to steepest descent on the hypersphere, applied column-wise. Generalizing Bernstein's (2025b) dualizer for steepest descent on the hypersphere to the Oblique manifold then yields,

> The optimal solution for finding the direction of steepest descent on the Oblique manifold $A^*$ given "raw Euclidean gradient" or differential $G$ is to simply project $G$ onto the tangent space at point $W \in \widetilde{\text{Ob}}(m, n)$ and then normalize column-wise.

The tangent space at $W$ is simply,
$$T_W\widetilde{\text{Ob}}(m, n) = \{A \in \mathbb{R}^{m \times n} \| \operatorname{diag}(W^\top A) = 0\}$$
or in words, the column-wise dot-product or "alignment" between $W$ and a candidate tangent vector $A$ must be zero for $A$ to be in the tangent space at $W$. The projector onto the tangent space at $W$ is then given by,
$$\operatorname{proj}_{T_W\widetilde{\text{Ob}}(m, n)}(G) = G - W \operatorname{diag}(W^\top G / m)$$
or in words, we subtract the component of $G$ that is "aligned to" $W$.

Notice then that one of the constraints is concerned with the *size* of the columns while the other is concerned with the *direction*. These can be optimized independently of each other. Thus, the solution for $A^*$ is then simply,
$$A^* = \operatorname{col\_normalize}(\operatorname{proj}_{T_W\widetilde{\text{Ob}}(m, n)}(G))$$

### 6.2. Steepest descent on the (scaled-)Row-Oblique manifold

We argue that the Unembedding layer or the 'language model head' should naturally be on the (scaled-)Row-Oblique manifold, $\widetilde{\text{RowOb}}(m, n) = \{ W \in \mathbb{R}^{m \times n} | \operatorname{diag}(WW^\top) = s^2\mathbf{1}_m \}$ with scale $s = \sqrt{n}$, or the manifold of matrices with unit-RMS-norm rows. The crux is that the logit for the $i$-th vocabulary token is given by the dot-product or 'alignment' between the $i$-th row of the weight matrix and the feature vector. So if the logits measure 'alignment', not 'size', then it is natural to constrain the rows to have unit-RMS-norm.

And since we can construct $\widetilde{\text{RowOb}}(m, n)$ by transposing $\widetilde{\text{Ob}}(m, n)$, we can use the same reasoning as above to derive the optimal solution for steepest descent on the (scaled-)Row-Oblique manifold.

Our retraction map is simply the row-wise normalization,
$$\operatorname{row\_normalize}(W) := \operatorname{row}_i(W) \mapsto \frac{\operatorname{row}_i(W)}{\| \operatorname{row}_i(W) \|_{RMS}} = \sqrt{n}\frac{\operatorname{row}_i(W)}{\| \operatorname{row}_i(W) \|_{2}} \quad \forall 0 \leq i < m$$
where $\operatorname{row}_i(W)$ is the $i$-th row of the weight matrix $W$. We then choose the $\| \cdot \|_{RMS \to \infty}$ norm because the maximizer for the following problem,
$$\arg\max_{A: \| A \|_{RMS \to \infty} = 1} \langle G, A \rangle$$
is simply $\operatorname{row\_normalize}(G) \in \widetilde{\text{RowOb}}(m, n)$. That is, the per-row updates would have even size.

Our update rule then becomes,
$$ W \leftarrow \operatorname{row\_normalize}(W - \eta A^*)$$
where $\eta$ is the learning rate and,
$$A^* = \arg\max_{A} \langle G, A \rangle  \quad \text{s.t. } A \in \widetilde{\text{RowOb}}(m, n) \cap T_W \widetilde{\text{RowOb}}(m, n),$$
which has the closed form solution,
$$\begin{align*}
    A^* &= \operatorname{row\_normalize}(\operatorname{proj}_{T_W\widetilde{\text{RowOb}}(m, n)}(G)) \\
         &= \operatorname{row\_normalize}(G - \operatorname{diag}(G W^\top / n) W) \\
\end{align*}$$

## 7. Experimental results

### 7.1. Alternating projections method beats ternary search on nearby feasible solutions on larger matrices

![](steepest-descent-stiefel.png#center)

![](steepest-descent-stiefel-edge.png#center)

Here we compare our two heuristic methods for the problem of spectral-norm constrained steepest descent on the Stiefel manifold. Observe from the figures above that the ternary search over nearby feasible solutions method results in almost optimal solutions, regardless of scale. However, the alternating projections method results in more aligned solutions, albeit at the cost of more compute and being more off-tangent.

### 7.2. Grokking on the Addition-Modulo-113 task in 44 full-batch training steps

> We will release the source code soon, but if you want early access, please email me.

![](grokking_results.png#center)

We use the same training setup for grokking experiments on the Addition-Modulo-113 problem as in a previous [post on spectral clipping](../spectral-clipping/), with new dualizers and projection maps added. Following Prieto et al. (2025), we use a 2-layer MLP (plus Embedding and Unembedding layers) with 200 hidden units per layer. All matrix multiplications are done in `bfloat16` precision.

We place our Embedding and Unembedding weights on the (scaled-)Oblique manifold and (scaled-)Row-Oblique manifold, respectively. We then vary the dualizer and retraction maps used in the linear layers and report the best median-steps-to-grokking across 64 random seeds. See figure above for the results.

Interestingly, without weight constraints, models fail to grok within 1000 full-batch training steps. This is true for both the Muon optimizer and AdamW. However, with weight constraints, we were able to achieve grokking in 44 full-batch training steps, which we believe is SOTA.

The best recipe seems to be:

| layer       |                                                               manifold |                                 $\operatorname{dualizer}$                                  |       $\operatorname{retract}$        |
| ----------- | ---------------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :-----------------------------: |
| Embedding   |                                              (Scaled-)Oblique manifold |  $\operatorname{col\_normalize} \circ \operatorname{proj}_{T_W\widetilde{\text{Ob}}(m, n)}$   |   $\operatorname{col\_normalize}$    |
| 1st Linear  | RMS-to-RMS norm ball<br>around the origin of $\mathbb{R}^{m \times n}$ |                                   $\operatorname{msign}$                                   | $\operatorname{spectral\_normalize}$ |
| 2nd Linear  | RMS-to-RMS norm ball<br>around the origin of $\mathbb{R}^{m \times n}$ |                                   $\operatorname{msign}$                                   | $\operatorname{spectral\_normalize}$ |
| Unembedding |                                          (Scaled-)Row-Oblique manifold | $\operatorname{row\_normalize} \circ \operatorname{proj}_{T_W\widetilde{\text{RowOb}}(m, n)}$ |   $\operatorname{row\_normalize}$    |

## Acknowledgements

Big thanks to Jianlin Su, Jeremy Bernstein, Vinay Rao, Antonio Silveti-Falls, Mikail Khona, Omead Pooladzandi, Simo Ryu, and Kevin Yin for productive discussions on the topic. All remaining mistakes are mine.

## How to cite

```bibtex
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {{H}euristic Solutions for {S}teepest {D}escent on the {S}tiefel manifold},
  year = {2025},
  month = {July},
  day = {18},
  url = {https://leloykun.github.io/ponder/steepest-descent-stiefel/},
}
```

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
2. Greg Yang, James B. Simon, Jeremy Bernstein (2024). A Spectral Condition for Feature Learning. URL https://arxiv.org/abs/2310.17813
3. Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein (2024). Scalable Optimization in the Modular Norm. URL https://arxiv.org/abs/2405.14813
4. Jeremy Bernstein, Laker Newhouse (2024). Old Optimizer, New Norm: An Anthology. URL https://arxiv.org/abs/2409.20325
5. Laker Newhouse (2025). Understanding Muon. URL https://www.lakernewhouse.com/writing/muon-1
6. Laker Newhouse, Preston Hess, Franz Cesista, Andrii Zahorodnii, Jeremy Bernstein, Phillip Isola (2025). Training Transformers with Enforced Lipschitz Constants. URL https://arxiv.org/abs/2507.13338
7. Jeremy Bernstein (2025a). Orthogonal manifold. URL https://docs.modula.systems/algorithms/manifold/orthogonal/
8. Jeremy Bernstein (2025b). Hypersphere. URL https://docs.modula.systems/algorithms/manifold/hypersphere/
9. Jianlin Su (2025). Steepest descent on Stiefel manifold. URL https://x.com/YouJiacheng/status/1945522729161224532
10. D. Drusvyatskiy, A.D. Ioffe, A.S. Lewis (2016). Transversality and alternating projections for nonconvex sets. URL https://arxiv.org/abs/1401.7569
11. Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher (2025). Training Deep Learning Models with Norm-Constrained LMOs. URL https://arxiv.org/abs/2502.07529
12. Bin Gao, Simon Vary, Pierre Ablin, P.-A. Absil (2022). Optimization flows landing on the Stiefel manifold. URL https://arxiv.org/abs/2202.09058
13. Pierre Ablin, Gabriel Peyré (2021). Fast and accurate optimization on the orthogonal manifold without retraction. URL https://arxiv.org/abs/2102.07432
14. Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano, Tolga Birdal (2025). Grokking at the Edge of Numerical Stability. URL https://arxiv.org/abs/2501.04697
