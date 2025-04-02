---
title: "Napkin Math on Non-Euclidean Trust Region Optimization"
date: 2025-03-24
tags: ["Machine Learning", "Muon"]
author: "Franz Louis Cesista"
description: "A possible reason why Muon converges faster & does better at higher learning rates than Adam."
summary: "A possible reason why Muon converges faster & does better at higher learning rates than Adam."
# cover:
#     image: cover.jpg
#     alt: "Cover"
# editPost:
#     URL: "https://x.com/leloykun/status/1901267939267162351"
#     Text: "Crossposted from X (formerly Twitter)"
---

In a [previous post](../steepest-descent-schatten-p/), we talked about how to derive some common optimizers from a choice of norm. In this post, we'll go over [Kovalev's recent paper on Non-Euclidean Trust Region Optimization](https://arxiv.org/abs/2503.12645) and how we can use its main result to explain why Muon converges faster and does better at higher learning rates than Adam.

## Non-Euclidean Trust Region Optimization

We consider the following optimization problem:
$$\min_{x \in \mathcal{X}} f(x)$$
where $f(\cdot): \mathcal{X} \rightarrow \mathbb{R}$ is a bounded from below and differentiable objective function and $\mathcal{X}$ is finite-dimensional vector space equipped with an inner product $\langle \cdot, \cdot \rangle: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ and a norm $\|\| \cdot \|\|: \mathcal{X} \rightarrow \mathbb{R}$ which may or may not coincide with the inner product.

> Note: Kovalev's objective actually has a regularization term, $R(\cdot): \mathcal{X} \rightarrow \mathbb{R} \cup \\{ +\infty \\}$ that is proper (i.e. not $\infty$ everywhere), closed, and convex. So the original objective is $\min_{x \in \mathcal{X}} [F(x) = f(x) + R(x)]$. We omit this for simplicity.

### Assumptions

The paper's results rely on the following assumptions, which I find to be quite reasonable:

1. We have access to an unbiased and bounded variance stochastic estimator $g(\cdot; \xi) : \mathcal{X} \rightarrow \mathcal{X}$ of the gradient $\nabla f(\cdot)$, $\xi \sim D$ is a random variable sampled from a distribution $D$. I.e., $$\mathbb{E}_{\xi \sim D}[g(x; \xi)] = \nabla f(x)\quad\text{and}\quad \mathbb{E}\_{\xi \sim D}[\|\|g(x; \xi) - \nabla f(x)\|\|_2^2] \leq \sigma^2\quad \forall x \in \mathcal{X}\tag{A1}$$ where $\sigma > 0$ is a positive variance parameter and $\|\|\cdot\|\|_2 = \sqrt{\langle \cdot, \cdot \rangle}$.

2. The gradient $\nabla f(\cdot)$ is Lipschitz continuous with constant $L > 0$. I.e., $$\|\|\nabla f(x) - \nabla f(x')\|\|^{\dagger} \leq L\|\|x - x'\|\|\quad \forall x, x' \in \mathcal{X}\tag{A2}$$ where $\|\|\cdot\|\|^{\dagger}$ is the dual norm of $\|\|\cdot\|\|$.

3. To connect the norms in **(A1)** and **(A2)**, we use the following inequality: $$\|\|x\|\|^{\dagger} \leq \rho\cdot\|\|x\|\|_2\quad \forall x \in \mathcal{X}\tag{A3}$$ where $\rho > 0$ is a positive constant. Note that this $\rho$ always exists due to the norm equivalence theorem for finite-dimensional vector spaces.

These are sufficient conditions for the convergence of the following trust region optimization algorithm.

### Algorithm 1: Stochastic Non-Euclidean Trust-Region Gradient Method with Momentum

![](tro-algo-1.png#center)

![](tro-algo-2.png#center)

### Important Results

The first is Lemma 3 which gives an upper bound on the distance between the momentum $m_k$ and the true gradient $\nabla f(x_k)$, following the update rule in Algorithm 1. And the second is Theorem 2 which provides convergence guarantees for Algorithm 1.

> **(Kovalev's) Lemma 3.** Let Assumptions **(A1)** to **(A3)** hold and let $m_0 = g(x_0; \xi_0)$. Then the iterations of Algorithm 1 satisfy the following inequality for $k \geq 1$:
$$\mathbb{E}[\|\| m_k - \nabla f(x_k) \|\|^{\dagger}] \leq (1-a)^k \rho \sigma + \sqrt{\alpha}\rho\sigma+\frac{L\eta}{\alpha}.$$

This is why, unlike OSGDM, we must accumulate the momentum term *before* we apply the dualizer. Otherwise, we can't guarantee that our momentum term would actually be useful, nor even converge. And this lemma also works for any well-defined norm $\|\|\cdot\|\|$ on $\mathcal{X}$. Previous work by Cutkosky & Mehta (2020) has only shown this for the Frobenius norm.

> **(Kovalev's) Theorem 2.** Let Assumptions **(A1)** to **(A3)** hold and let $m_0 = g(x_0; \xi_0)$. Then the iterations of Algorithm 1 satisfy the following inequality:
$$\mathbb{E}\left[ \min_{k=1,\ldots,K} \|\| \nabla f(x_k) \|\|^{\dagger} \right] \leq \frac{\Delta_0}{\eta K} + \frac{2\rho\sigma}{\alpha K} + 2\sqrt{\alpha}\rho\sigma + \frac{3L\eta}{2} + \frac{2L\eta}{\alpha},$$ where $\Delta_0 = f(x_0) - \inf_x f(x)$. Hence, to reach precision $\mathbb{E}\left[ \min\_{k=1,\ldots,K} \|\| \nabla f(x\_k) \|\|^{\dagger} \right] \leq \epsilon$, it is sufficient to choose the stepsize $\eta$ and the momentum parameter $\alpha$ as follows:
$$\eta = \min\left\\{\frac{\epsilon}{16L}, \frac{\epsilon^3}{256\rho^2\sigma^2L} \right\\},\quad\quad \alpha=\min\left\\{ 1, \frac{\epsilon^2}{16\rho^2\sigma^2} \right\\},$$ and the number of iterations $K$ as follows:
$$K = \left\lceil \max\left\\{ \frac{2048L\Delta_0\rho^2\sigma^2}{\epsilon^4}, \frac{256\rho^3\sigma^3}{\epsilon^3},\frac{128L\Delta_0}{\epsilon^2}, \frac{16\rho\sigma}{\epsilon} \right\\}\right\rceil.$$

I think this is the real meat of the paper. And this works for any well-defined norm $\|\|\cdot\|\|$ on $\mathcal{X}$, not just the Frobenius norm.

However, the norm embedding constant $\rho$ and the Lipschitz constant $L$ both depend on the choice of norm. And as they increase, the lower $\eta$ and $\alpha$ become and the higher $K$ becomes.

## A Possible Reason Why Muon Outperforms Adam

We're entering napkin math territory here, so take everything with a grain of salt.

Here we'll show that Adam's $\rho$ and $L$ are higher than Muon's and consequently its $\eta$ and $\alpha$ are lower and $K$ is higher. And this may explain why Muon converges faster and does better at higher learning rates than Adam.

### Preliminaries

Let's focus on linear layers. I.e., $\mathcal{X} = \mathbb{R}^{m \times n}$. And following Bernstein & Newhouse (2024), we can interpret Adam and Muon as steepest descent under the operator norms $\|\|A\|\|\_{\text{max}}$ and $\|\|A\|\|\_{2\to 2}$, respectively.

| Optimizer |                                Norm ($\|\|\cdot\|\|$)                                 |                Dual Norm ($\|\|\cdot\|\|^{\dagger}$)                |
| :-------: | :-----------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
|   Adam    | $\quad\|\|A\|\|_{\text{max}} = \|\|vec(A)\|\|\_\infty = \max\_{i,j}\|A\_{i,j}\|\quad$ |             $\|\|A\|\|\_{1} = \sum\_{i,j}\|A\_{i,j}\|$              |
|   Muon    |    $\quad\|\|A\|\|_{2\to 2} = \|\|\sigma(A)\|\|\_\infty = \max_i \sigma_i(A)\quad$    | $\|\|A\|\|_{\text{nuc}} = \|\|\sigma(A)\|\|_1 = \sum_i \sigma_i(A)$ |

where $A \in \mathbb{R}^{m \times n}$, $\sigma(A) = (\sigma(A)_1, \sigma(A)_2, \ldots, \sigma(A)_r)$ are the singular values of $A$, and $r = \text{rank}(A) \leq \min(m, n)$.

We will also heavily rely on the following inequality in our proofs below:

> **The monotonicity inequality for $p$-norms.** For any vector $x \in \mathbb{R}^N$ and for $0 < p' \leq q' \leq \infty$, the $p$-norms satisfy $$\|\|x\|\|\_{q'} \leq \|\|x\|\|\_{p'} \leq N^{1/{p'} - 1/{q'}}\|\|x\|\|\_{q'},\tag{1}$$

which is a direct consequence of the Hölder's inequality.

### Adam has a higher norm embedding constant $\rho$ than Muon

**For Adam:**
> $$
\begin{align*}
    \|\|A\|\|\_{\text{max}}^{\dagger}
        &= \|\|A\|\|_1 = \|\|vec(A)\|\|_1\\\\
        &\leq (mn)^{1/1 - 1/2}\|\|vec(A)\|\|_2 && \text{(from $\bm{(1)}$, with $p'=1$ and $q'=2$)}\\\\
        &\leq \sqrt{mn}\|\|vec(A)\|\|_2\\\\
    \|\|A\|\|\_{\text{max}}^{\dagger} &\leq \sqrt{mn}\|\|A\|\|_F \tag{2}
\end{align*}
$$
Thus, Adam's $\rho$ is $\sqrt{mn}$.

Likewise, **for Muon:**
> $$
\begin{align*}
    \|\|A\|\|_{2\to 2}^{\dagger}
        &= \|\|A\|\|\_{nuc} = \|\|\sigma(A)\|\|_1\\\\
        &\leq (\min\\{m,n\\})^{1/1 - 1/2}\|\|\sigma(A)\|\|\_{2} && \text{(from $\bm{(1)}$, with $p'=1$ and $q'=2$)}\\\\
        &\leq \sqrt{\min\\{m,n\\}}\|\|\sigma(A)\|\|\_{2}\\\\
    \|\|A\|\|\_{2\to 2}^{\dagger} &\leq \sqrt{\min\\{m,n\\}}\|\|A\|\|\_F \tag{3}
\end{align*}
$$
Thus, Muon's $\rho$ is $\sqrt{\min\\{m,n\\}}$ which is always less than Adam's $\sqrt{mn}$.

### Adam has a higher Lipschitz constant $L$ than Muon

We make another assumption:

> Suppose that we have a constant $\hat{L} > 0$ such that for all $X, X' \in \mathbb{R}^{m\times n}$ we have $$\|\|\nabla f(X) - \nabla f(X')\|\|_F \leq \hat{L} \|\| X - X' \|\|_F.\tag{A4}$$

**For Adam:**
> $$
\begin{align*}
    \|\|vec(A)\|\|\_2 &\leq (mn)^{1/2 - 0}\|\|vec(A)\|\|\_\infty && \text{(from $\bm{(1)}$, with $p'=2$ and $q'=\infty$)}\\\\
    \|\|A\|\|_F       &\leq \sqrt{mn}\|\|A\|\|\_{\text{max}}\tag{4}
\end{align*}
$$
Thus for all $X, X' \in \mathbb{R}^{m\times n}$,
$$
\begin{align*}
    \|\|\nabla f(X) - \nabla f(X')\|\|\_\text{max}^{\dagger}
        &\leq \sqrt{mn}\|\|\nabla f(X) - \nabla f(X')\|\|_F && \text{(from $\bm{(2)}$)}\\\\
        &\leq \hat{L}\sqrt{mn}\|\|X - X'\|\|_F && \text{(from $\bm{(A4)}$)}\\\\
    \|\|\nabla f(X) - \nabla f(X')\|\|\_\text{max}^{\dagger}
        &\leq \hat{L}mn\|\|X - X'\|\|\_\text{max} && \text{(from $\bm{(4)}$)}\tag{5}\\\\
\end{align*}
$$
Thus, Adam's $L$ is $\hat{L}mn$.

Likewise, **for Muon:**
> $$
\begin{align*}
    \|\|\sigma(X)\|\|\_2 &\leq (\min\\{m,n\\})^{1/2 - 0}\|\|\sigma(X)\|\|\_\infty && \text{(from $\bm{(1)}$, with $p'=2$ and $q'=\infty$)}\\\\
    \|\|X\|\|_F       &\leq \sqrt{\min\\{m,n\\}}\|\|X\|\|\_{2\to 2}\tag{6}
\end{align*}
$$
Thus for all $X, X' \in \mathbb{R}^{m\times n}$,
$$
$$
\begin{align*}
    \|\|\nabla f(X) - \nabla f(X')\|\|\_{2\to 2}^{\dagger}
        &\leq \sqrt{\min\\{m,n\\}}\|\|\nabla f(X) - \nabla f(X')\|\|_F && \text{(from $\bm{(3)}$)}\\\\
        &\leq \hat{L}\sqrt{\min\\{m,n\\}}\|\|X - X'\|\|_F && \text{(from $\bm{(A4)}$)}\\\\
    \|\|\nabla f(X) - \nabla f(X')\|\|\_{2\to 2}^{\dagger}
        &\leq \hat{L}\min\\{m,n\\}\|\|X - X'\|\|\_{2\to 2} && \text{(from $\bm{(6)}$)}\tag{7}\\\\
\end{align*}
$$
$$
Thus, Muon's $L$ is $\hat{L}\min\\{m,n\\}$ which is always less than Adam's $\hat{L}mn$.

### Conclusion

Let's go back to Kovalev's main result:
$$\eta = \min\left\\{\frac{\epsilon}{16L}, \frac{\epsilon^3}{256\rho^2\sigma^2L} \right\\},\quad\quad \alpha=\min\left\\{ 1, \frac{\epsilon^2}{16\rho^2\sigma^2} \right\\},$$ $$K = \left\lceil \max\left\\{ \frac{2048L\Delta_0\rho^2\sigma^2}{\epsilon^4}, \frac{256\rho^3\sigma^3}{\epsilon^3},\frac{128L\Delta_0}{\epsilon^2}, \frac{16\rho\sigma}{\epsilon} \right\\}\right\rceil$$

Muon has a lower norm embedding constant $\rho$ and Lipschitz constant $L$ than Adam. Thus, Muon's learning rate $\eta$ and momentum $\alpha$ should be higher. And in practice, we do observe that Muon loves higher learning rates. Also notice that a lower $\rho$ and $L$ leads to a lower $K$. This means that, theoretically, Muon should converge faster than Adam--which is exactly what we observe in practice.

### Caveats

The bounds here are *very* loose. And for SGD, which is steepest descent under the Frobenius norm, we have $\rho = 1$ and $L = \hat{L}$ which implies that it's better than both Adam and Muon, which is not true in practice. Thus, again, take everything  here with a grain of salt.

## How to Cite

```bibtex
@misc{cesista2025tro,
  author = {Franz Louis Cesista},
  title = {Napkin Math on Non-Euclidean Trust Region Optimization},
  year = {2025},
  url = {http://leloykun.github.io/ponder/napkin-math-trust-region-opt/},
}
```

## References

1. Dmitry Kovalev (2025). Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization. Available at: https://arxiv.org/abs/2503.12645
2. Jeremy Bernstein and Laker Newhouse. “Old optimizer, new norm: An anthology.” arXiv preprint arXiv:2409.20325 (2024).
3. Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza Vlado, Jiacheng You, Franz Cesista, Braden Koszarsky, and @Grad62304977. modded-nanogpt: Speedrunning the NanoGPT baseline. 2024. Available at: https://github.com/KellerJordan/modded-nanogpt.
4. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/.
5. Moonshot AI Team (2025). Muon is Scalable for LLM Training. URL https://arxiv.org/abs/2502.16982
6. Cutkosky, A., & Mehta, H. (2020). Momentum improves normalized SGD. In Proceedings of the 37th International Conference on Machine Learning (PMLR Vol. 119, pp. 2260-2268). http://proceedings.mlr.press/v119/cutkosky20b.html
