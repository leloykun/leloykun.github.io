---
title: "Napkin Math on Non-Euclidean Trust Region Optimization"
date: 2025-03-24
tags: ["Machine Learning", "Muon"]
author: "Franz Louis Cesista"
description: "A possible reason why Muon converges faster & does better at higher learning rates than Adam."
summary: "A possible reason why Muon converges faster & does better at higher learning rates than Adam."
# cover:
#     image: "cover.jpg"
#     alt: "Cover"
# editPost:
#     URL: "https://x.com/leloykun/status/1901267939267162351"
#     Text: "Crossposted from X (formerly Twitter)"
---

[UNDER CONSTRUCTION]

In a [previous post](../steepest-descent-schatten-p/), we talked about how to derive some common optimizers from a choice of norm. In this post, we'll go over Kovalev's recent paper on Non-Euclidean Trust Region Optimization and how we can use its main result to explain why Muon converges faster and does better at higher learning rates than Adam.

## Non-Euclidean Trust Region Optimization

We consider the following optimization problem:
$$\min_{x \in \mathcal{X}} f(x)$$
where $f(\cdot): \mathcal{X} \rightarrow \mathbb{R}$ is a bounded from below and differentiable objective function and $\mathcal{X}$ is finite-dimensional vector space equipped with an inner product $\langle \cdot, \cdot \rangle: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ and a norm $\|\| \cdot \|\|: \mathcal{X} \rightarrow \mathbb{R}$ which may or may not coincide with the inner product.

---

We make three assumptions:

> **(A1).** We have access to an unbiased and bounded variance stochastic estimator $g(\cdot; \xi) : \mathcal{X} \rightarrow \mathcal{X}$ of the gradient $\nabla f(\cdot)$, $\xi \sim D$ is a random variable sampled from a distribution $D$. I.e., $$\mathbb{E}_{\xi \sim D}[g(x; \xi)] = \nabla f(x)\quad\text{and}\quad \mathbb{E}\_{\xi \sim D}[\|\|g(x; \xi) - \nabla f(x)\|\|_2^2] \leq \sigma^2\quad \forall x \in \mathcal{X}$$ where $\sigma > 0$ is a positive variance parameter and $\|\|\cdot\|\|_2 = \sqrt{\langle \cdot, \cdot \rangle}$.

> **(A2).** The gradient $\nabla f(\cdot)$ is Lipschitz continuous with constant $L > 0$. I.e., $$\|\|\nabla f(x) - \nabla f(x')\|\|_* \leq L\|\|x - x'\|\|\quad \forall x, x' \in \mathcal{X}$$ where $\|\|\cdot\|\|\_*$ is the dual norm of $\|\|\cdot\|\|$.

> **(A3).** To connect the norms in **(A1)** and **(A2)**, we use the following inequality: $$\|\|x\|\|_* \leq \rho\cdot\|\|x\|\|_2\quad \forall x \in \mathcal{X}$$ where $\rho > 0$ is a positive constant. Note that this $\rho$ always exists due to the norm equivalence theorem for finite-dimensional vector spaces.

---

> **(Kovalev's) Theorem 2.** Let Assumptions **(A1)** to **(A3)** hold, and let $m_0 = g(x_0; \xi_0)$. Then the iterations of Algorithm 1 satisfy the following iequality:
$$\mathbb{E}\left[ \min_{k=1,\ldots,K} \|\| \nabla f(x_k) \|\|_* \right] \leq \frac{\Delta_0}{\eta K} + \frac{2\rho\sigma}{\alpha K} + 2\sqrt{\alpha}\rho\sigma + \frac{3L\eta}{2} + \frac{2L\eta}{\alpha},$$ where $\Delta_0 = \nabla f(x_0) - \inf_x f(x)$. Hence, to reach precision $\mathbb{E}\left[ \min\_{k=1,\ldots,K} \|\| \nabla f(x\_k) \|\|\_* \right] \leq \epsilon$, it is sufficient to choose the stepsize $\eta$ and the momentum parameter $\alpha$ as follows:
$$\eta = \min\left\\{\frac{\epsilon}{16L}, \frac{\epsilon^3}{256\rho^2\sigma^2L} \right\\},\quad\quad \alpha=\min\left\\{ 1, \frac{\epsilon^2}{16\rho^2\sigma^2} \right\\},$$ and the number of iterations $K$ as follows:
$$K = \left\lceil \max\left\\{ \frac{2048L\Delta_0\rho^2\sigma^2}{\epsilon^4}, \frac{256\rho^3\sigma^3}{\epsilon^3},\frac{128L\Delta_0}{\epsilon^2}, \frac{16\rho\sigma}{\epsilon} \right\\}\right\rceil.$$

First, notice that the theorem works for any well-defined norm $\|\|\cdot\|\|$ on $\mathcal{X}$. However, $\rho$ and the Lipschitz constant $L$ depend on the choice of norm. And as they increase, the lower $\eta$ and $\alpha$ become and the higher $K$ becomes.

---

## Adam vs. Muon

Here we'll show that Adam's $\rho$ and $L$ are higher than Muon's and consequently its $\eta$ and $\alpha$ are lower and $K$ is higher. And this largely explains why Muon converges faster and does better at higher learning rates than Adam.

### Preliminaries

Following Bernstein & Newhouse (2024), we can interpret Adam and Muon as steepest descent under operator norms:

| Optimizer |                                         Norm                                          |                   Dual Norm                    |
| :-------: | :-----------------------------------------------------------------------------------: | :--------------------------------------------: |
|   Adam    | $\quad\|\|A\|\|_{\text{max}} = \|\|vec(A)\|\|\_\infty = \max\_{i,j}\|A\_{i,j}\|\quad$ |   $\|\|x\|\|\_{1} = \sum\_{i,j}\|A\_{i,j}\|$   |
|   Muon    |    $\quad\|\|A\|\|_{2\to 2} = \|\|\sigma(A)\|\|\_\infty = \max_i \sigma_i(A)\quad$    | $\|\|x\|\|\_{\text{nuc}} = \sum_i \sigma_i(A)$ |

where $\sigma(A) = (\sigma(A)_1, \sigma(A)_2, \ldots, \sigma(A)_r)$ are the singular values of $A$ and $r \leq \min(m, n)$ is the rank of $A$.

And we will also heavily rely on the following inequality in our proofs below:

> **The monotonicity for $p$-norms.** For any vector $x \in \mathbb{R}^N$ and for $0 < p \leq q \leq \infty$, the $p$-norms satisfy $$\|\|A\|\|_q \leq \|\|A\|\|_p \leq N^{1/p - 1/q}\|\|A\|\|_q,$$ which is a direct consequence of the Hölder's inequality.

### Adam has a higher $\rho$ than Muon

Let $x \in \mathbb{R}^{m \times n}$. We can let $p = 1$ and $q = 2$ above to get,

> For Adam:
$$
\begin{align*}
    \|\|A\|\|_1 &= \|\|\sigma(A)\|\|_1\\\\
                &\leq (mn)^{1/1 - 1/2}\|\|\sigma(A)\|\|_2\\\\
                &\leq \sqrt{mn}\|\|\sigma(A)\|\|_2\\\\
    \|\|A\|\|_1 &\leq \sqrt{mn}\|\|A\|\|_F
\end{align*}
$$
Thus, Adam's $\rho$ is $\sqrt{mn}$.

> Likewise, for Muon:
$$
\begin{align*}
    \|\|A\|\|\_{nuc} &= \|\|\sigma(A)\|\|_1\\\\
                     &\leq \min\\{m,n\\}^{1/1 - 1/2}\|\|\sigma(A)\|\|\_{2}\\\\
                     &\leq \sqrt{\min\\{m,n\\}}\|\|\sigma(A)\|\|\_{2}\\\\
    \|\|A\|\|\_{nuc} &\leq \sqrt{\min\\{m,n\\}}\|\|A\|\|\_F
\end{align*}
$$
Thus, Muon's $\rho$ is $\sqrt{\min\\{m,n\\}}$ which is always less than $\sqrt{mn}$.

### Adam has a higher $L$ than Muon

We make another assumption:

> **(A4).** Suppose that for all $x, x'$ we have $$\|\|\nabla f(x) - \nabla f(x')\|\|_F \leq \|\| x - x' \|\|_F.$$

Now, to get an upper bound on the RHS, let $p = 2$ and $q = \infty$ above to get,

> For Adam:
$$
\begin{align*}
    \|\|\sigma(A)\|\|\_2 &\leq (mn)^{1/2 - 0}\|\|\sigma(A)\|\|\_\infty\\\\
    \|\|A\|\|_F       &\leq \sqrt{mn}\|\|A\|\|\_{\text{max}}
\end{align*}
$$
Thus for all $x, x'$,
$$
\begin{align*}
    \|\|x - x'\|\|_F &\leq \sqrt{mn}\|\|x - x'\|\|\_{\text{max}}\\\\
    \|\|\nabla f(x) - \nabla f(x')\|\|_F &\leq \sqrt{mn}\|\|x - x'\|\|\_{\text{max}}\quad\quad\text{from (A4)}\\\\
    \frac{1}{\sqrt{mn}}\|\|\nabla f(x) - \nabla f(x')\|\|_1 &\leq \sqrt{mn}\|\|x - x'\|\|\_{\text{max}}\\\\
    \|\|\nabla f(x) - \nabla f(x')\|\|_1 &\leq mn\|\|x - x'\|\|\_{\text{max}}
\end{align*}
$$
Thus, Adam's $L$ is $mn$.

> Likewise, for Muon:
$$
\begin{align*}
    \|\|\sigma(A)\|\|\_2 &\leq (\min\\{m,n\\})^{1/2 - 0}\|\|\sigma(A)\|\|\_\infty\\\\
    \|\|A\|\|_F       &\leq \sqrt{\min\\{m,n\\}}\|\|A\|\|\_{2\to 2}
\end{align*}
$$
Thus for all $x, x'$,
$$
\begin{align*}
    \|\|x - x'\|\|_F                                        &\leq \sqrt{\min\\{m,n\\}}\|\|x - x'\|\|\_{2\to 2}\\\\
    \|\|\nabla f(x) - \nabla f(x')\|\|_F                    &\leq \sqrt{\min\\{m,n\\}}\|\|x - x'\|\|\_{2\to 2}\quad\quad\text{from (A4)}\\\\
    \frac{1}{\sqrt{\min\\{m,n\\}}}\|\|\nabla f(x) - \nabla f(x')\|\|\_{\text{nuc}} &\leq \sqrt{\min\\{m,n\\}}\|\|x - x'\|\|\_{2\to 2}\\\\
    \|\|\nabla f(x) - \nabla f(x')\|\|\_{\text{nuc}}                    &\leq \min\\{m,n\\}\|\|x - x'\|\|\_{2\to 2}
\end{align*}
$$
Thus, Muon's $L$ is $\min\\{m,n\\}$ which is always less than $mn$.

### Conclusion

Let's go back to:
$$\eta = \min\left\\{\frac{\epsilon}{16L}, \frac{\epsilon^3}{256\rho^2\sigma^2L} \right\\},\quad\quad \alpha=\min\left\\{ 1, \frac{\epsilon^2}{16\rho^2\sigma^2} \right\\},$$ $$K = \left\lceil \max\left\\{ \frac{2048L\Delta_0\rho^2\sigma^2}{\epsilon^4}, \frac{256\rho^3\sigma^3}{\epsilon^3},\frac{128L\Delta_0}{\epsilon^2}, \frac{16\rho\sigma}{\epsilon} \right\\}\right\rceil$$

Muon has a lower $\rho$ and $L$ than Adam. Thus, Muon's $\eta$ and $\alpha$ should be higher. And, at least for the learning rate, this is actually the case. Also notice that a lower $\rho$ and $L$ means a lower $K$, which means that, theoretically, Muon should converge faster than Adam which is what we observe in practice.

## How to Cite

```bibtex
@misc{cesista2025tro,
  author = {Franz Louis Cesista},
  title = {Napkin Math on Non-Euclidean Trust Region Optimization},
  year = {2025},
  url = {http://leloykun.github.io/ponder/napkin-math-trust-region-opt/},
}
```
