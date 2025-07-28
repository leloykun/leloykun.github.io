---
title: "A Simple Heuristic Solution for Steepest Descent on Stiefel Manifold"
date: 2025-07-18
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "Fast, numerically stable, and differentiable solution for steepest descent on Stiefel manifold."
summary: "Fast, numerically stable, and differentiable solution for steepest descent on Stiefel manifold."
cover:
    image: steepest-descent-stiefel-manifold.jpg
    alt: "Cover"
    relative: true
editPost:
    URL: "https://x.com/leloykun/status/1946640155655475316"
    Text: "Crossposted on X (formerly Twitter)"
---

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

> This is still a Work in Progress (WIP). I've decided to publish this earlier than planned to get feedback and iterate quickly. If you spot any mistakes, please don't hesitate to let me know! Email me at franzlouiscesista@gmail.com or tag me on X ([@leloykun](https://x.com/leloykun)).

## 1. Recap: Muon as spectral norm-constrained steepest descent

Consider a weight matrix $W \in \mathbb{R}^{m \times n}$ and a "raw gradient" or differential $G \in \mathbb{R}^{m \times n}$ we get via e.g., backpropagation. In standard gradient descent, we would update the weights as follows,
$$W \leftarrow W - \eta G,$$
where $\eta \in (0, \infty)$ is the learning rate. However, this is suboptimal because (1) the update sizes $\\| G \\|$ could vary wildly across steps thereby causing training instability, and (2) as we discussed in a [previous blog post](../steepest-descent-non-riemannian/), it does not take into account the matrix structure of the weights. In particular, it ignores how activations or "features" evolve through the network and how the model behaves as we scale it up. Tl;dr:
> If we want the Euclidean norm $\\| \cdot \\|\_2$ of our features and feature updates to 'grow' with the model size,
> then the *Spectral norm* $\\| \cdot \\|\_{2 \to 2}$ of our weights and weight updates must also 'grow' with the model size.

Alternatively, following Yang et al. (2024), we can use the "natural" feature norm, the RMS norm $\\|\cdot\\|\_{RMS} = \sqrt{n}\\|\cdot\\|\_{2}$, and the "natural" weight norm, the RMS-to-RMS norm $\\|\cdot\\|\_{RMS \to RMS} = \frac{\sqrt{m}}{\sqrt{n}}\\| \cdot \\|\_{2 \to 2}$, and we can rephrase the above as,
> If we want the "natural" norm of our features and feature updates to be stable regardless of the model size,
> then the "natural" norm of our weights and weight updates must also be stable regardless of the model size.

We will discuss weight norm controls in the next section, but for now, instead of using the raw gradient $G$, we can instead try to find a descent direction $A \in \mathbb{R}^{m \times n}$ that is maximally aligned to $G$ while satisfying our weight update condition above,
    $$\\| A \\|\_{RMS \to RMS} = \frac{\sqrt{m}}{\sqrt{n}}\\| A \\|\_{2 \to 2} = \text{constant}.$$
Thus our update rule becomes,
    $$W \leftarrow W - \eta \frac{\sqrt{m}}{\sqrt{n}} A^\*,$$
where
$$A^\* = \arg\max\_{\\| A \\|\_{2 \to 2} = 1} \langle G, A \rangle,$$
and $\langle \cdot, \cdot \rangle$ is the Frobenius inner product which measures the "alignment" between two matrices. From Bernstein & Newhouse (2024), this has a closed-form solution,
$$A^\* = \texttt{msign}(G),$$
where $\texttt{msign}(\cdot)$ is the matrix sign function. And finally, adding a momentum term then yields the Muon optimizer (Jordan et al., 2024),
$$\begin{align*}
    M\_t &= \beta M\_{t-1} + (1 - \beta) G \\\\
    W\_t &= W\_{t-1} - \eta\frac{\sqrt{m}}{\sqrt{n}} \texttt{msign}(M\_t), \\\\
\end{align*}$$
for some momentum hyperparameter $\beta \in [0, 1)$.

> If you want to learn more about Muon and the ideas behind it, check out [Newhouse's 3-part blog series](https://www.lakernewhouse.com/writing/muon-1). I highly recommend it!

## 2. Spectral norm-constrained steepest descent on Stiefel manifold

As discussed in the previous section, we not only need to control the weight update norms but also the weight norms themselves. There are multiple ways to do this and we presented some novel methods in our recent work on [training transformers with enforced Lipschitz bounds](https://arxiv.org/abs/2507.13338) (Newhouse*, Hess*, Cesista* et al., 2025). However, here we will focus on constraining the weights to be semi-orthogonal, i.e., $W^T W = I_n$.

Semi-orthogonal matrices lie on the Stiefel manifold, $\text{St}(m, n)$. And at an arbitrary point $W \in \text{St}(m, n)$, the space of all possible velocity vectors or "update directions" passing through this point is called the tangent space at $W$, denoted as $T\_W \text{St}(m, n)$. For the Stiefel manifold we have,
$$T\_W \text{St}(m, n) = \\{A \in \mathbb{R}^{m \times n} \| W^T A + A^T W = 0\\}.$$

Now, to "move around" on this manifold, we take an update step on the tangent space at $W$, $A \in T\_W \text{St}(m, n)$, use this to 'move' our weight $W \leftarrow W + \tilde{\eta} A$, then retract the result back to the manifold via $\text{msign}(\cdot)$, and repeat. Ideally, we want our update step to be maximally aligned to $G$ while also satisfying unit-normed-ness. Thus our update rule becomes,
$$W \leftarrow \text{msign}\left(W - \eta\frac{\sqrt{m}}{\sqrt{n}}A^\* \right)$$
where
$$A^\* = \arg\max\_{\\| A \\|\_{2 \to 2} = 1} \langle G, A \rangle \quad \text{s.t. } A \in T\_W\text{St}(m, n),$$

Equivalently,
$$A^\* = \arg\max \langle G, A \rangle  \quad \text{s.t. } A \in \text{St}(m, n) \cap T\_W \text{St}(m, n),$$
Or in words, we want to find a descent direction $A$ that is both on the Stiefel manifold and in the tangent space at $W$ that maximizes the "alignment" with the raw gradient $G$. 

## 3. Equivalence between Bernstein's and Su's solutions

Bernstein (2025) and Su (2025) found the following solutions to the square and full-rank case,

$$\begin{align*}
    A^\*\_{\text{bernstein}} &= W \texttt{msign}(\texttt{skew}(W^TG))\\\\
    A^\*\_{\text{su}} &= \texttt{msign}(G - W\texttt{sym}(W^T G))
\end{align*}$$

We will show that these are equivalent, i.e., $A^\*\_{\text{bernstein}} = A^\*\_{\text{su}}$. For this, we will reuse the following proposition we discussed in a [previous post on spectral clipping](../spectral-clipping).

> **Proposition 1 (Transpose Equivariance and Unitary Multiplication Equivariance of Odd Matrix Functions)**. Let $W \in \mathbb{R}^{m \times n}$ and $W = U \Sigma V^T$ be its reduced SVD. And let $f: \mathbb{R}^{m \times n} \to \mathbb{R}^{m \times n}$ be an odd analytic matrix function that acts on the singular values of $W$ as follows,
> $$f(W) = U f(\Sigma) V^T.$$
> Then $f$ is equivariant under transposition and unitary multiplication, i.e.,
> $$\begin{align*}
    f(W^T) &= f(W)^T \\\\
    f(WQ^T) &= f(W)Q^T \quad\forall Q \in \mathbb{R}^{m \times n} \text{ such that } Q^TQ = I_n \\\\
    f(Q^TW) &= Q^Tf(W) \quad\forall Q \in \mathbb{R}^{m \times n} \text{ such that } QQ^T = I_m
\end{align*}$$

Thus,

$$\begin{align*}
    A^\*\_{\text{bernstein}}
        &= W \texttt{msign}(\texttt{skew}(W^TG)) \\\\
        &= \texttt{msign}(W \texttt{skew}(W^TG)) &\text{(from Proposition 1)} \\\\
        &= \texttt{msign}\left(\frac{1}{2}W W^T G - \frac{1}{2}W G^T W \right) \\\\
        &= \texttt{msign}\left(W W^T G - \frac{1}{2}W W^T G - \frac{1}{2}W G^T W \right) \\\\
        &= \texttt{msign}\left(G - W\texttt{sym}(W^T G) \right) \\\\
    A^\*\_{\text{bernstein}} &= A^\*\_{\text{su}}
\end{align*}$$
where the second-to-last equality relies on $W$ being square and full-rank, which then implies that $W W^T = I\_m$.

## 4. Projection-projection perspective

One can interpret Bernstein's and Su's solutions as a two-step projection process:

1. (Orthogonal) projection to the tangent space at $W$; and
2. Projection to the closest semi-orthogonal matrix (i.e., closest point on the Stiefel manifold).

This is because the map $G \to G - W \texttt{sym}(W^T G)$ is actually the orthogonal projection onto the tangent space at $W$ and that $\texttt{msign}$ projects the resulting matrix to the stiefel manifold. We present these more rigorously as follows.

> **Theorem 2 (Orthogonal projection to the tangent space at $W$).** Let $W \in \text{St}(m, n)$ be a point on the Stiefel manifold. The projection of a vector $V \in \mathbb{R}^{m \times n}$ onto the tangent space at $W$, denoted as $T\_W\text{St}(m, n)$ , is given by,
> $$\begin{equation}
    \texttt{proj}\_{T\_W\text{St}(m, n)}(V) = V - W \text{sym}(W^T V)
\end{equation}$$

{{< collapse summary="Show **proof of Theorem 2**" openByDefault=false >}}
> **Proof.** First, we need to show that the normal space at $W$, $N\_W\text{St}(m, n)$ is given by,
> $$N\_W\text{St}(m, n) = \\{WS \| S = S^T\\}$$
> for symmetric $S$. To show this, let $A \in T\_W\text{St}(m, n)$ be an arbitrary tangent vector at $W$. Then we have,
> $$\begin{align*}
    \langle A, WS \rangle &= \text{tr}(A^T WS) \\\\
        &= \text{tr}(S W^T A) &\text{(transpose invariance of trace)} \\\\
        &= -\text{tr}(S A^T W) &\text{(since $A \in T\_W\text{St}(m, n)$)} \\\\
        &= -\text{tr}(A^T W S) &\text{(cyclic property of trace)} \\\\
    \langle A, WS \rangle &= -\langle A, WS \rangle
\end{align*}$$
> Thus $\langle A, WS \rangle = 0$ which implies that $A$ and $WS$ are orthogonal. Hence $WS \in N\_W\text{St}(m, n)$.
>
> Now, for any $V \in \mathbb{R}^{m \times n}$, we can write it as,
> $V = \underbrace{V - WS}\_{\text{candidate tangent}} + \underbrace{WS}\_{\text{candidate normal}}$ for some symmetric $S$. To find $S$,
> $$\begin{align*}
    W^T (V - WS) + (V - WS)^T W &= 0 \\\\
    W^T V - S + V^TW - S &= 0 \\\\
    S &= \text{sym}(W^T V)
\end{align*}$$
> Thus, $V - W \text{sym}(W^T V) \in T\_W\text{St}(m, n)$. And because of that, $W^T (V - W \text{sym}(W^T V))$ must be skew-symmetric. Thus,
> $$\begin{equation*}
    \langle V - WS, WS \rangle
        = \langle \underbrace{W^T (V - WS)}\_{\text{skew-symmetric}}, \underbrace{S}\_{\text{symmetric}} \rangle
        = 0
\end{equation*}$$
> Hence, $V - W \text{sym}(W^T V)$ is the orthogonal projection of $V$ onto the tangent space at $W$.

{{< /collapse >}}

And from Proposition 4 of Bernstein & Newhouse (2024),

> **Proposition 3 (Projection to the closest semi-orthogonal matrix).** Consider the orthogonal matrices $\mathcal{O}\_{m \times n} := \\{ A \in \mathbb{R}^{m \times n} : A A^T = I\_m or A^T A = I\_n \\}$ and let $\\| \cdot \\|\_F$ denote the Frobenius norm. For any matrix $G \in R^{m \times n}$ with reduced SVD $G = U \Sigma V^T$:
> $$\arg\min\_{A \in \mathcal{O}\_{m \times n}} \\| A - G \\|\_F = \texttt{msign}(G) = UV^T,$$
> where the minizer $UV^T$ is unique if and only if the matrix $G$ has full rank.

Thus we can write,
$$A^\*\_{\text{bernstein}} = A^\*\_{\text{su}} = (\texttt{proj}\_{\text{St}(m, n)} \circ \texttt{proj}\_{T\_W\text{St}(m, n)})(G)$$
for the square and full-rank case at least.

### 4.1. Why Bernstein's & Su's solutions only work for the square and full-rank case

Note that the projections above aim to guarantee both of our criteria for the solution, but one step at a time. And that the $\texttt{msign}$ after the projection may send the resulting matrix outside the tangent space at $W$. We show that this is not a problem in the square and full-rank case, but it is in the general case.

| operation                                                |              on the tangent space at $W$? |    have unit spectral norm? |
| -------------------------------------------------------- | ----------------------------------------: | --------------------------: |
| (input $G$)                                              |                            not in general |              not in general |
| 1st projection ($\texttt{proj}\_{T\_W St(m, n)}(\cdot)$) |               ${\color{green}\text{yes}}$ |             not necessarily |
| 2nd projection ($\texttt{msign}(\cdot$))                 | only for the square<br>and full-rank case | ${\color{green}\text{yes}}$ |

To demonstrate this, we will need the following proposition.

> **Proposition 4 ($\texttt{msign}$ preserves skew-symmetry).** Let $X \in \R^{m \times n}$ be a skew-symmetric matrix. Then $\texttt{msign}(X)$ is also skew-symmetric.

The proof follows directly from the transpose equivariance and oddness of $\texttt{msign}$, i.e., $\texttt{msign}(X) = \texttt{msign}(-X^T) = -\texttt{msign}(X)^T$.

Also note that for the general case,
$$\begin{equation}
    WW^T + QQ^T = I\_m \quad\text{and}\quad W^T Q = 0
\end{equation}$$
where $Q$ is the orthonormal complement of $W$.

Thus,
$$\begin{align*}
    (\texttt{proj}\_{\text{St}(m, n)} \circ \texttt{proj}\_{T\_W St(m, n)})(G)
        &= \texttt{msign}(G - \frac{1}{2}W W^T G - \frac{1}{2} W G^T W) \\\\
        &= \texttt{msign}((WW^T + QQ^T)G - \frac{1}{2}W W^T G - \frac{1}{2} W G^T W) \\\\
        &= \texttt{msign}(W \texttt{skew}(W^T G) + QQ^T G)
\end{align*}$$
For the square and full-rank case, we have $Q = 0$ and $W W^T = I$. And the above simplifies to Berstein's solution,
$$(\texttt{proj}\_{\text{St}(m, n)} \circ \texttt{proj}\_{T\_W St(m, n)})(G) = W \texttt{msign}(\texttt{skew}(W^T G))$$
and since $\texttt{skew}(W^T G)$ is skew-symmetric, then $\texttt{msign}(\texttt{skew}(W^T G))$ must be too. And thus,
$$\begin{align*}
    &W^T (\texttt{proj}\_{\text{St}(m, n)} \circ \texttt{proj}\_{T\_W St(m, n)})(G) + (\texttt{proj}\_{\text{St}(m, n)} \circ \texttt{proj}\_{T\_W St(m, n)})(G)^T W \\\\
        &\qquad= W^T W \texttt{msign}(\texttt{skew}(W^T G)) + (W\texttt{msign}(\texttt{skew}(W^T G)))^T W \\\\
        &\qquad= \texttt{msign}(\texttt{skew}(W^T G)) + \texttt{msign}(\texttt{skew}(W^T G))^T \\\\
        &\qquad= 0
\end{align*}$$
Hence for the square and full-rank case, this two-step projection process guarantees that the resulting matrix has unit spectral norm *and* is in the tangent space at $W$.

For the more general case, we have $Q = 0$ and $W W^T = I$ may no longer hold true. Thus, we cannot guarantee that $W \texttt{skew}(W^T G) + QQ^T G$ is skew-symmetric. And so the $\texttt{msign}$ after the first projection may send the resulting matrix outside the tangent space at $W$.

## 5. Heuristic solutions for the general case

### 5.1. Alignment upper bound

As established previously, for any $G \in \mathbb{R}^{m \times n}$, the solution to $\arg\max\_{A \in \text{St}(m, n)} \langle G, A \rangle$ is $A^\* = \texttt{proj}\_{\text{St}(m, n)}(G) = \texttt{msign}(G)$ and thus $\max\_{A \in \text{St}(m, n)} \langle G, A \rangle = \\| G \\|\_{\text{nuc}}$ where $\\| \cdot \\|\_{\text{nuc}}$ is the nuclear norm. With an extra linear constraint $A \in T\_W \text{St}(m, n)$, we have,

$$\begin{align*}
    \max\_{A \in \text{St}(m, n) \cap T\_W \text{St}(m, n)} \langle G, A \rangle
        &= \max\_{A \in \text{St}(m, n) \cap T\_W \text{St}(m, n)} \langle \underbrace{G - \texttt{proj}\_{T\_W \text{St}(m, n)}(G)}\_{\in N\_W\text{St}(m, n)} + \texttt{proj}\_{T\_W \text{St}(m, n)}(G), A \rangle \\\\
        &= \max\_{A \in \text{St}(m, n) \cap T\_W \text{St}(m, n)} \langle \texttt{proj}\_{T\_W \text{St}(m, n)}(G), A \rangle \\\\
        &\leq \max\_{A \in \text{St}(m, n)} \langle \texttt{proj}\_{T\_W \text{St}(m, n)}(G), A \rangle \\\\
    \max\_{A \in \text{St}(m, n) \cap T\_W \text{St}(m, n)} \langle G, A \rangle
        &\leq \\| \texttt{proj}\_{T\_W \text{St}(m, n)}(G) \\|\_{\text{nuc}}
\end{align*}$$

The alignment between an element in the normal space and an element in the tangent space is always zero, i.e., $\langle \texttt{proj}\_{T\_W \text{St}(m, n)}(G), \texttt{proj}\_{N\_W \text{St}(m, n)}(G) \rangle = 0$. Thus, we can cancel out the first term in the first equality. And we have first inequality because the max over a smaller set is less than or equal to the max over a larger set.

We achieve equality in the square and full-rank case because the maximizer for the first inequality is guaranteed to be in the tangent space at $W$, as discussed in the previous section.

### 5.2. Fixed-point iteration of alternating projections

Notice that $QQ^T G$ is the projection of $G$ onto the column space of $Q$, $\texttt{proj}\_{\text{col}(Q)}(G) = QQ^T G$. One can think of this as the component of $G$ that is, in a sense, *not* "aligned to" $W$. In practice, this is typically small relative to the component of $G$ that *is* "aligned to" $W$. If so, then,
$$(\texttt{proj}\_{\text{St}(m, n)} \circ \texttt{proj}\_{T\_W St(m, n)})(G) \approx \texttt{msign}(W \texttt{skew}(W^T G) + \cancel{\texttt{proj}\_{\text{col}(Q)}(G)}),$$
which means that while the resulting matrix after the two projections may not be in the tangent space at $W$, it would likely be *nearby*. And repeating this process a few times should close the gap.

Here's a sample implementation,
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

For now, we cannot yet guarantee global convergence as it would potentially require deriving the Lyapunov function for this iteration which is extremely difficult. What we can guarantee, however, is *local* convergence due to $\text{St}(m, n)$ and $T\_W\text{St}(m, n)$ being closed semi-algebraic sets. That is, assuming that the initial point $W$ is "close enough" to the intersection, we can guarantee that a subsequence of the iterates converges to a point in the intersection. Furthermore, assuming transversality, we can guarantee that the convergence is linear.

> **Definition 5 (Semi-algebraic set)** Let $\mathbb{F}$ be a real closed field. A subset $S$ of $\mathbb{F}^n$ is a semi-algebraic set if it is a finite union of sets defined by polynomial equations and inequalities.

We can construct $\text{St}(m, n) = \\{ W \in \mathbb{R}^{m \times n} \| W^T W = I\_n \\}$ via $n(n+1)/2$ polynomial equations, and $T\_W\text{St}(m, n) = \\{ A \in \mathbb{R}^{m \times n} \| W^T A + A^T W = 0 \\}$ via $mn$ polynomial equations. Hence both are semi-algebraic sets. And the intersection of two semi-algebraic sets is also a semi-algebraic set. From Theorem 7.3 of Drusvyatskiy et al. (2016), we then have the following convergence guarantee.

> **Theorem 6 (Convergence of alternating projections on semi-algebraic sets)** Consider two nonempty closed semi-algebraic sets $X, Y \subset E$ with $X$ bounded. If the method of alternating projections starts in $Y$ and near $X$, then the distance of the iterates to the intersection $X \cap Y$ converges to zero, and hence every limit point lies in $X \cap Y$.

Setting $Y = T\_W\text{St}(m, n)$, $X = \text{St}(m, n)$, and noting that the Stiefel manifold is bounded, we have that the iterates of our method of alternating projections converge to a point in the intersection $\text{St}(m, n) \cap T\_W\text{St}(m, n)$.

But does this algorithm converge in sufficient time? Somewhat yes, we can guarantee local *linear* convergence assuming transversality.

> **Definition 7 (Transversality)** Let $\mathcal{M}$ and $\mathcal{N}$ be two (smooth) manifolds in a Euclidean space. We say that $\mathcal{M}$ and $\mathcal{N}$ intersect transversally at a point $x \in \mathcal{M} \cap \mathcal{N}$ when,
> $$N\_x\mathcal{M} \cap N\_x\mathcal{N} = \\{0\\}.$$

Intuitively speaking, this means that the tangent spaces at the intersection of the two manifolds have an "angle" between them, i.e., they are not parallel. The larger this angle is, the faster the convergence; and the smaller the angle, the slower the convergence. Theorem 2.1 of Drusvyatskiy et al. (2016) then gives us the following local linear convergence guarantee.

> **Theorem 8 (Linear convergence of alternating projections, assuming transversality)** If two closed sets in a Euclidean space intersect transversally at a point $\tilde{x}$, then the method of alternating projections, started nearby, converges linearly to a point in the intersection.

$\text{St}(m, n)$ and $T\_W\text{St}(m, n)$ are both closed sets so the theorem above applies.

### 5.3. Ternary search over nearby feasible solutions

Here we present an alternative solution that is more efficient, but often yields slightly more suboptimal results than the fixed-point iteration method above.

#### 5.3.1. Problem decomposition

The crux is to split $\arg\max\_{A} \langle G, A \rangle$ into two optimization problems, one for the component of $G$ that is "aligned to" $W$ and one for the component of $G$ that is "not aligned to" $W$. To see this, let's first decompose $G$ and $A$ into,
$$G = W G\_W + Q G\_Q \qquad A = WB + QC$$
where $G\_W = W^T G$ and $G\_Q = Q^T G$. Thus,
$$\begin{align}
    \langle G, A \rangle
        &= \langle W G\_W + Q G\_Q, WB + QC \rangle \nonumber \\\\
        &= \text{tr}((W G\_W + Q G\_Q)^T (WB + QC)) \nonumber \\\\
        &= \text{tr}(G\_W^T B + G\_Q^T C) \nonumber \\\\
    \langle G, A \rangle
        &= \langle G\_W, B \rangle + \langle G\_Q, C \rangle \\\\
\end{align}$$
The cross terms vanish in the fourth equality because $W^T Q = 0$. Finding the maximizer $A^\*$ for the LHS is then equivalent to finding the maximizers $B^\*$ and $C^\*$ for the RHS and then combining them,
$$A^\* = WB^\* + QC^\*.$$

#### 5.3.2. Solving the two subproblems

Now, to satisfy the constraint $A \in T\_W\text{St}(m, n) \cap \text{St}(m, n)$,
$$\begin{align*}
    W^T A + A^T W &= 0 \qquad\qquad& A^T A &= I\_n \\\\
    W^T (WB + QC) + (WB + QC)^T W &= 0 \qquad\qquad& (WB + QC)^T(WB + QC) &= I\_n \\\\
    B + B^T &= 0 \qquad\qquad& B^T B + C^T C &= I\_n \\\\
\end{align*}$$
That is, we require that $B$ is skew-symmetric and $C$ satisfies $C^T C = I\_n - B^T B$.

We cannot simply treat each subproblem separately because the second constraint couples $B$ and $C$. However, for this approximation, we make the assumption that doing so would yield a "good enough" solution.

---

For the first term, we can decompose $G\_W$ into its skew-symmetric and symmetric components, $G\_W = \texttt{skew}(G\_W) + \texttt{sym}(G\_W)$,
$$\begin{align*}
    \arg\max\_{B \text{ is skew}} \langle G\_W, B \rangle
        &= \arg\max\_{B \text{ is skew}} \langle \texttt{skew}(G\_W) + \texttt{sym}(G\_W), B \rangle \\\\
        &= \arg\max\_{B \text{ is skew}} \langle \texttt{skew}(G\_W), B \rangle + \cancel{\langle \texttt{sym}(G\_W), B \rangle}
\end{align*}$$
And the maximizer is simply $\tilde{B} = \texttt{skew}(G\_W) = \texttt{skew}(W^T G)$.

However, because of the constraint $B^T B + C^T C = I$, we have to "cap" the spectral norm of $B$ to be less than or equal to 1 otherwise we would fail to construct a real-valued $C$. We can do this via a variety of methods discussed in a [previous post on spectral clipping](../spectral-clipping/) and our latest paper on [training transformers with enforced Lipschitz bounds](https://arxiv.org/abs/2507.13338) (Newhouse*, Hess*, Cesista* et al., 2025). Let $\tau \leq 1$ be the spectral norm bound, then we can choose,

$$B^\*(\tau) := \texttt{hard\\_cap}\_{\tau}(\tilde{B})
    \quad\text{or}\quad B^\*(\tau) := \tau\cdot\texttt{msign}(\tilde{B})
    \quad\text{or}\quad B^\*(\tau) := \frac{\tau}{\\| \tilde{B} \\|\_{2 \to 2}}\tilde{B}$$

These mappings preserve skew-symmetry and thus $B^\*(\tau)$ satisfies the first constraint.

---

Now, let's parametrize $C$ as $C = UR$ where $U^T U = I$ and $R(\tau) = (I_n - (B^\*(\tau))^T B^\*(\tau))^{1/2}$. It is trivial to check that $C$ satisfies our constraints and that $R(\tau)$ is SPD. Thus, assuming we already have a fixed $B^\*(\tau)$ (and consequently a fixed $R(\tau)$), solving the second subproblem,
$$\arg\max\_{C} \langle G\_Q, C \rangle$$
is equivalent to solving,
$$\begin{align*}
    \arg\max\_{U} \langle G\_Q, U R(\tau) \rangle
        &= \arg\max\_{U} \text{tr}(G\_Q^T U R(\tau)) \\\\
        &= \arg\max\_{U} \text{tr}(R(\tau) G\_Q^T U) \\\\
        &= \arg\max\_{U} \langle G\_Q R(\tau), U \rangle
\end{align*}$$
which has maximizer $U^\* = \texttt{msign}(G\_Q R(\tau))$. Thus, $C^\*(\tau) = \texttt{msign}(G\_Q R(\tau)) R(\tau)$.

---

Note that for the square and full-rank case, we have $C = 0$ and so we require $B^T B = I$. For this, we can choose to orthogonalize $B = \texttt{skew}(G\_W)$ which then yields Bernstein's solution,
$$A^\*\_{\text{bernstein}} = W \texttt{msign}(\texttt{skew}(W^T G))$$
This also motivates the choice of $\texttt{msign}$ as the normalization method for $B$ more generally.

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

From Equation (3), we have,

$$\begin{align*}
    f(\tau) := \langle G, A^\*(\tau) \rangle
        &= \langle G\_W, B^\*(\tau) \rangle + \langle G\_Q, C^\*(\tau) \rangle \\\\
        &= \langle \texttt{skew}(G\_W), B^\*(\tau) \rangle + \langle G\_Q, C^\*(\tau) \rangle \\\\
        &= \langle \texttt{skew}(G\_W), \texttt{normalized}\_{\tau}(\texttt{skew}(G\_W)) \rangle + \langle G\_Q, \texttt{msign}(G\_Q R(\tau)) R(\tau) \rangle \\\\
        &= \langle \texttt{skew}(G\_W), \texttt{normalized}\_{\tau}(\texttt{skew}(G\_W)) \rangle + \\| G\_QR(\tau) \\|\_{\text{nuc}} \\\\
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

## 6. Experimental results [Under Construction]

![](steepest-descent-stiefel.png#center)

![](steepest-descent-stiefel-edge.png#center)

## How to cite

```bibtex
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"A Simple Heuristic Solution for Steepest Descent on Stiefel Manifold"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/steepest-descent-stiefel/},
}
```

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
2. Greg Yang, James B. Simon, Jeremy Bernstein (2024). A Spectral Condition for Feature Learning. URL https://arxiv.org/abs/2310.17813
3. Jeremy Bernstein, Laker Newhouse (2024). Old Optimizer, New Norm: An Anthology. URL https://arxiv.org/abs/2409.20325
4. Laker Newhouse (2025). Understanding Muon. URL https://www.lakernewhouse.com/writing/muon-1
5. Laker Newhouse, R. Preston Hess, Franz Cesista, Andrii Zahorodnii, Jeremy Bernstein, Phillip Isola (2025). Training Transformers with Enforced Lipschitz Constants. URL https://arxiv.org/abs/2507.13338
6. Jeremy Bernstein (2025). Orthogonal manifold. URL https://docs.modula.systems/algorithms/manifold/orthogonal/
7. Jianlin Su (2025). Steepest descent on Stiefel manifold. URL https://x.com/YouJiacheng/status/1945522729161224532
8. D. Drusvyatskiy, A.D. Ioffe, A.S. Lewis (2016). Transversality and alternating projections for nonconvex sets. URL https://arxiv.org/abs/1401.7569
