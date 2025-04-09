---
title: "Muon and a Selective Survey on Steepest Descent in Riemannian and Non-Riemannian Manifolds"
date: 2025-04-03
tags: ["Machine Learning", "Muon"]
author: "Franz Louis Cesista"
description: "Muon from first principles, what makes it different from other optimizers, and why it works so well."
summary: "Muon from first principles, what makes it different from other optimizers, and why it works so well."
cover:
    image: cover.png
    alt: "Cover"
    relative: true
editPost:
    URL: "https://x.com/leloykun/status/1907211629982556320"
    Text: "Crossposted from X (formerly Twitter)"
---

> This is still a Work in Progress (WIP). I've decided to publish this earlier than planned to get feedback and iterate quickly. If you spot any mistakes, please don't hesitate to let me know! Email me at franzlouiscesista@gmail.com or tag me on X ([@leloykun](https://x.com/leloykun)).

A new optimizer called Muon (Jordan et al., 2024a) has recently been shown to outperform Adam (Kingma et al., 2014) in both small-scale language model training (Jordan et al., 2024b), and larger-scale language model training (Moonshot AI Team, 2025) by a factor of up to 2x in terms of flops efficiency. For non-matrix-valued parameters in a neural network, Muon falls back to Adam. But for matrix-valued parameters, Muon first semi-orthogonalizes the gradient before subtracting it from the parameter. It can also be viewed as steepest descent under the Spectral norm (Bernstein et al., 2024).

## 0. Introduction

In deep learning, the goal is find a *function* that maps input data to output data such that a certain optimization objective $\mathcal{L}$ (called "loss function") is minimized. We parametrize this function with a set of weights $\\{W_l\\}$ that are typically matrix-valued. However, previous work on deep learning optimization algorithms typically ignore the matrix-structure and functional nature of these weights. A common strategy involves flattening the weights into a vector and treating them as such while others flatten then unflatten the weights in some intermediate step (Kingma et al., 2014; Li, 2015; Gupta et al., 2018; Surya et al., 2024; Pooladzandi et al., 2024). Worse, this is also prevalent in related fields such as evolutionary algorithms research (Salimans et al., 2017; Braun et al., 2024), among others.

And as demonstrated by Jordan et al. (2024a) on small-scale language model training, and Moonshot AI Team (2025) on larger-scale language model training, a simple change of perspective from a weights-as-vectors to a weights-as-matrices perspective can lead to significant improvements in training efficiency. Thus, following the work of Berstein et al. (2024), we argue--and demonstrate--that the underlying geometry of the weights are crucial to training performance and that we can reason, from first principles, about the properties of the geometry in which our weights (should) "live" in.

This work is a selective survey of latest advancements in deep learning optimization research, with a focus on new developments in late 2024 and early 2025. However, we also make several novel contributions:
1. In Sections 1 and 3, we formalize Steepest Descent in Riemannian and non-Riemannian manifolds, and how different choices of norms lead to different classes of deep learning optimization algorithms.
2. We also formalize the connection between preconditioners in optimizers and the metric tensor in Riemannian steepest descent, and how we can use this to develop more robust intuitions on optimizer design such as when to update the preconditioner.
3. We also discuss the connection between preconditioners and dualizers in optimizers, and when to use one over the other.
4. We also show that the optimizer CASPR (Surya et al., 2024) reduces to Muon when accumulation on the (left and right) preconditioners is disabled.
5. In Sections 2 and 4, we motivate the Muon optimizer from first principles, and show how it can be viewed as a steepest descent under the spectral norm. We also discuss many possible reasons why it works so well in practice, despite not fitting in with more established intuitions in the field.
6. In Section 6, we dicuss how to further improve Muon by optimizing the coefficients of the Newton-Schulz iteration. We also discuss how to use Muon to improve itself. And finally,
7. In Section 7, we discuss convergence guarantees for Muon.

## 1. Preliminaries

We consider the following optimization problem,
$$\begin{equation} \arg\min_{W \in \bm{\mathcal{W}}} \mathcal{L}(W), \end{equation}$$
where $\mathcal{L}(\cdot): \bm{\mathcal{W}} \rightarrow \mathbb{R}$ is a bounded-below and differentiable objective function, and $\bm{\mathcal{W}}$ is a real-valued, finite-dimensional, matrix manifold equipped with a norm $||\cdot||$ chosen a priori. If the norm is induced by an inner product (i.e., the parallelogram law holds), then $\bm{\mathcal{W}}$ is a Riemannian manifold. Otherwise, it is a non-Riemannian manifold. Thus, not only does the choice of norm naturally lead to different optimization algorithms, but also to two *flavors* of optimizers, preconditioners and dualizers, which we will discuss in the following sections.

In practice, $\mathcal{L}$ often does not have a simple, closed-form solution, so we resort to iterative methods of the form,
$$W_{t+1} = W_{t} - \lambda \widehat{\Delta W}\_t,$$
where $\lambda > 0$ is a positive learning rate parameter, $W_t \in \mathcal{W}$ is the "current" point at step $t$, $W_{t+1} \in \mathcal{W}$ is the "updated" point at step $t+1$, and $-\widehat{\Delta W}\_t \in T_{W_t}\mathcal{W}$ is the direction of steepest descent at step $t$,
$$
\begin{align}
    -\widehat{\Delta W}\_t &= \arg\min_{\substack{\Delta W \in T_{W\_t}\mathcal{W}\\\\ ||\Delta W|| = 1}} d\mathcal{L}\_{W\_t}(\Delta W)\\\\
    \widehat{\Delta W}\_t &= \arg\max_{\substack{\Delta W \in T_{W\_t}\mathcal{W}\\\\ ||\Delta W|| = 1}} d\mathcal{L}\_{W\_t}(\Delta W)
\end{align}
$$
where $T_{W\_t}\mathcal{W}$ is the tangent space at $W\_t$, $d\mathcal{L}\_{W\_t}(\cdot): T_{W\_t}\mathcal{W} \rightarrow \mathbb{R}$ is the differential of $\mathcal{L}$ at $W\_t$, and $d\mathcal{L}\_{W\_t}(\Delta W)$ is the directional derivative of $\mathcal{L}$ at $W\_t$ in the direction of $\Delta W$.

We also often do not have access to the exact differential. However, either through, e.g., backpropagation if downstream operations are differentiable (Rumelhart et al., 1986), we often do have access to a stochastic estimator of the differential in coordinate form,

> **Assumption 1:** Let $\partial\mathcal{L}(W)\_{\text{coord}} \in T^*\_W\mathcal{W}$ be the coordinate representation of the differential $d\mathcal{L}\_W(\cdot): T\_W\mathcal{W} \rightarrow \mathbb{R}$ at $W \in \bm{\mathcal{W}}$ such that,
> $$d\mathcal{L}\_W(\cdot) = \langle \partial\mathcal{L}(W)\_{\text{coord}}, \cdot \rangle\_F,$$
> where $\langle \cdot, \cdot \rangle\_F$ is the Frobenius inner product. We assume that we have access to a stochastic estimator $\partial\mathcal{L}(W; \xi)\_{\text{coord}} \in T^\*\_W\mathcal{W}$ of the differential in coordinate form that is unbiased and has bounded variance. That is,
> 
> $$
\begin{align*}
    &\mathbb{E}\_{\xi \sim D}[\partial\mathcal{L}(W; \xi)\_{\text{coord}}] = \partial\mathcal{L}(W)\_{\text{coord}} && \forall W \in \bm{\mathcal{W}}\\\\
    &\mathbb{E}\_{\xi \sim D}[||\partial\mathcal{L}(W; \xi)\_{\text{coord}} - \partial\mathcal{L}(W)\_{\text{coord}} ||_F^2] \leq \sigma^2 && \forall W \in \bm{\mathcal{W}}
\end{align*}
$$
> where $\xi$ is a random variable sampled from a  distribution $D$, $\sigma > 0$ is a positive variance parameter, and $||\cdot||_F = \sqrt{\langle \cdot, \cdot \rangle_F}$ is the corresponding Frobenius norm.

We also make the following standard continuity assumption on the differential $\partial\mathcal{L}(\cdot)$ (Mokhtari et al., 2018; Kovalev, 2025),
> **Assumption 2:** The differential $\partial\mathcal{L}(\cdot)\_{\text{coord}}$ is Lipschitz continuous with respect to the norm $||\cdot||$ with Lipschitz constant $L > 0$. That is, for all $W \in \bm{\mathcal{W}}$,
$$
\begin{equation}
    ||\partial\mathcal{L}(W + \Delta W)\_{\text{coord}} - \partial\mathcal{L}(W)\_{\text{coord}}||^\dagger \leq L||\Delta W|| \quad \forall \Delta W \in T_W\bm{\mathcal{W}}
\end{equation}
$$
where $||\cdot||^\dagger$ is the dual norm of $||\cdot||$.

And in the following sections, we will also discuss optimizers that precondition the differentials,

> **Definition 1 (Preconditioning).** In an optimizer, a preconditioner $\mathcal{P}(\cdot; W): T^\*\_W\mathcal{W} \rightarrow T_W\mathcal{W}$ is a (possibly point-dependent) linear transform that maps the coordinate representation of the differential we have access to $\partial\mathcal{L}(W; \xi)\_{\text{coord}} \in T^\*\_W\mathcal{W}$ to a descent direction in the tangent space $\widehat{\Delta W} \in T\_W\mathcal{W}$. That is, at any $W \in \mathcal{W}$, we have a matrix $P_W$ such that,
> $$\Delta W = \mathcal{P}(\partial\mathcal{L}(W; \xi)\_{\text{coord}}; W) = P_{W} \partial\mathcal{L}(W; \xi)\_{\text{coord}}$$
> $$W_{t+1} = W_t - \lambda P_{W_t} \partial\mathcal{L}\_\xi(W_t)\_{\text{coord}}.$$
>
> It is also common to assume that we can decompose $P_W$ into a Kronecker product $P_W = L_W \otimes R_W$ (Li, 2015; Gupta et al., 2018, Surya et al., 2024), such that our update rule becomes,
> $$
\begin{align*}
    \Delta W &= P_{W} \partial\mathcal{L}(W; \xi)\_{\text{coord}}\\\\
    \Delta W &= \left( L_{W} \otimes R_{W} \right) \partial\mathcal{L}(W; \xi)\_{\text{coord}}\\\\
    \Delta W &= L_{W} \partial\mathcal{L}(W; \xi)\_{\text{coord}} R_{W}
\end{align*}
$$
> $$W_{t+1} = W_t - \lambda L_{W_t} \partial\mathcal{L}(W_t)\_{\text{coord}} R_{W_t}.$$
> We call $L_W$ and $R_W$ as the left and right preconditioners, respectively.

## 2. Why do Steepest Descent Under the Spectral Norm?

The geometry of $\mathcal{W}$ and the optimizer we will need both depend on the choice of norm $||\cdot||$. Our core argument is that it is most natural to do steepest descent under the spectral norm $||\cdot||_{2 \to 2}$ in the context of training the linear weights $W$ of a neural network. The spectral norm induces $\mathcal{W}$ to be non-Riemannian, and therefore, intuitions on optimization we have developed in Riemannian manifolds may not apply.

### 2.1. Majorization-Minimization Perspective [Under Review]

We can upper bound our objective function $\mathcal{L}$ by the following approximation at an arbitrary point $W \in \bm{\mathcal{W}}$,
$$
\begin{equation}
    \mathcal{U}(\Delta W; W) = \mathcal{L}(W) + \langle \partial\mathcal{L}(W)\_{\text{coord}}, \Delta W \rangle_F + \frac{\lambda}{2}||\Delta W||^2
\end{equation}
$$
for some norm $||\cdot||$. Using standard arguments, we can show that,
$$\mathcal{L}(W + \Delta W) \leq \mathcal{U}(\Delta W; W)$$
for all $\Delta W \in T_W\bm{\mathcal{W}}$ as long as $\lambda \leq L$ (Hunter et al., 2004).

A natural strategy to (iteratively) minimize $\mathcal{L}$ from point $W \in \mathcal{W}$ then is to (iteratively) minimize the majorant $\mathcal{U}(\cdot; W)$. And as discussed by Carlson et al. (2015), the spectral norm gives us a very tight upper bound and is thus a good choice. In fact, the spectral norm gives the tightest bound among all the Schatten-$p$ norms (the Frobenius norm included). And just as importantly, Equation (5) above has a simple, closed-form solution for the spectral norm as we will discuss in Section 4.

### 2.2 Feature Learning Perspective

This section can be summarized as,
> If we want the Euclidean norm of our features and feature updates to 'grow' with the model size,
> then the *Spectral norm* of our weights and weight updates must also 'grow' with the model size.

Suppose that we have a linear transform $x_{l+1} = W_{l} x_{l}$ at the $l$-th layer of a neural network where $x_l \in \mathcal{R}^{d_l}$ and $x_{l+1} \in \mathcal{R}^{d_{l+1}}$ are the input and output hidden representations (or "features"), respectively, and $W_l \in \mathcal{R}^{d_{l+1} \times d_l}$ is the weight matrix. Additionally, let $\Delta x_l \in \mathcal{R}^{d_l}$, $\Delta x_{l+1} \in \mathcal{R}^{d_{l+1}}$, and $\Delta W_l \in \mathcal{R}^{d_{l+1} \times d_l}$ be their updates after a backward pass.

Ideally, we want the sizes of both the hidden representations $x_l$ and their updates $\Delta x_l$ to scale with the model width $d_l$. Otherwise, if the hidden representations are 'too small', we are wasting capacity, in a sense (Elhage et al., 2022); and if they are 'too large', we are pushing the model towards the edge of numerical stability and prevent grokking (Prieto et al., 2025). Likewise, if the updates are 'too small', they vanish at larger scales, slowing down convergence; and if they are 'too large', they cause training instability. Yang et al. (2024) summarizes this as follows,

> **Desideratum 1 (Feature Learning).** We desire that our features $x_l$ and feature updates $\Delta x_l$ be of size,
$$
\begin{equation}
    ||x_l||_2 = \Theta(\sqrt{d_l})\quad\text{and}\quad ||\Delta x_l||_2 = \Theta(\sqrt{d_l})\quad\text{for all layers } l = 1, 2, \ldots, L-1
\end{equation}
$$
> where $f(d) = \Theta(g(d))$ means that $f(d)$ "scales like" or "grows in the order of" $g(d)$. More formally, there exists positive real constants $c, C > 0$ and positive integer $D$ such that, for all $d > D$, we have $c \cdot g(d) \leq f(d) \leq C \cdot g(d)$.

We ensure this by imposing constraints on the size of the weights $W_l$ and their updates $\Delta W_l$:

1. From the definition of the spectral norm, we have,
$$
\begin{align*}
    x\_{l+1} &= W\_l x\_l\\\\
    ||x\_{l+1}||\_2 &\leq ||W\_l||\_{2\to 2} \cdot ||x\_l||\_2
\end{align*}
$$
Combining this with Desideratum 1, we have,
$$
\begin{align*}
    \underbrace{||x\_{l+1}||\_2}\_{\Theta(\sqrt{d_{l+1}})}
        &\leq ||W\_l||\_{2\to 2} \cdot \underbrace{||x\_l||\_2}\_{\Theta(\sqrt{d_l})}
\end{align*}
$$
Thus the size of the weights $W_l$ must be,
$$
\begin{equation}
    ||W_l||\_{2 \to 2} = \Theta\left(\sqrt{\frac{d\_{l+1}}{d\_l}}\right)
\end{equation}
$$

1. Now let's consider the feature updates $\Delta x_l$,
$$
\begin{align*}
    x\_{l+1} + \Delta x\_{l+1} &= (W\_l + \Delta W\_l)(x\_l + \Delta x\_l)\\\\
    \Delta x\_{l+1} &= W_l \Delta x_l + \Delta W_l x_l + \Delta W_l \Delta x_l\\\\
    ||\Delta x\_{l+1}||\_2 &\leq ||W\_l||\_{2\to 2} \cdot ||\Delta x\_l||\_2 + ||\Delta W\_l||\_{2\to 2} \cdot ||x\_l||\_2 + ||\Delta W\_l||\_{2\to 2} \cdot ||\Delta x\_l||\_2
\end{align*}
$$
Combining this with Desideratum 1 and our result above, we have,
$$
\begin{align*}
    \underbrace{||\Delta x\_{l+1}||\_2}\_{\Theta(\sqrt{d_{l+1}})}
        &\leq
            \underbrace{||W\_l||\_{2\to 2}}\_{\Theta\left(\sqrt{\frac{d\_{l+1}}{d\_l}}\right)} \cdot \underbrace{||\Delta x\_l||\_2}\_{\Theta\left(\sqrt{d\_l}\right)}
            + ||\Delta W\_l||\_{2\to 2} \cdot \underbrace{||x\_l||\_2}\_{\Theta\left(\sqrt{d\_l}\right)}
            + ||\Delta W\_l||\_{2\to 2} \cdot \underbrace{||\Delta x\_l||\_2}\_{\Theta\left(\sqrt{d\_l}\right)}
\end{align*}
$$
Thus the size of the weight updates $\Delta W_l$ must be,
$$||\Delta W_l||\_{2 \to 2} = \Theta\left(\sqrt{\frac{d\_{l+1}}{d\_l}}\right)$$

These become our *Spectral Scaling Conditions* (Yang et al., 2024),

> **Condition 1 (Spectral Scaling).** The spectral norms of our weights $W_l$ and weight updates $\Delta W_l$ must be,
$$
\begin{equation}
    ||W_l||\_{2\to 2} = \Theta\left(\sqrt{\frac{d_{l+1}}{d_l}}\right)\quad\text{and}\quad||\Delta W_l||\_{2\to 2} = \Theta\left(\sqrt{\frac{d_{l+1}}{d_l}}\right)\quad\text{at layers } l = 1, \ldots, L-1
\end{equation}
$$

### 2.3 Input-Tensor Alignment Phenomenon [Under Construction]

## 3. Steepest Descent in Riemannian and Non-Riemannian Manifolds

Let us consider the different cases of the geometry of $\bm{\mathcal{W}}$ induced by the choice of norm $||\cdot||$.

### 3.1. $\bm{\mathcal{W}}$ is Euclidean

That is, we pick the Frobenius norm $||\cdot||_F$ as our norm. In this case, our points, differentials, and gradients are all already in standard Euclidean coordinates. Thus,

$$
\begin{align*}
    \widehat{\Delta W}
        &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} d\mathcal{L}\_W(\Delta W)\\\\
        &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} \langle \partial\mathcal{L}(W)\_{\text{coord}}, \Delta W \rangle_F\\\\
        &= \frac{\partial\mathcal{L}(W)\_{\text{coord}}}{||\partial\mathcal{L}(W)\_{\text{coord}}||\_F}\\\\
    \widehat{\Delta W} &\approx \frac{\partial\mathcal{L}(W; \xi)\_{\text{coord}}}{||\partial\mathcal{L}(W; \xi)\_{\text{coord}}||\_F}\\\\
\end{align*}
$$

Thus, our update rule becomes,
$$W_{t+1} = W_t - \hat{\lambda} \partial\mathcal{L}(W; \xi)\_{\text{coord}}$$
where $\hat{\lambda} = \frac{\lambda}{||\partial\mathcal{L}(W; \xi)\_{\text{coord}}||_F}$. This is simply Stochastic Gradient Descent (SGD) with an adaptive learning rate.

### 3.2. $\bm{\mathcal{W}}$ is a Riemannian Manifold

That is, our choice of norm $||\cdot||$ admits a smoothly-varying metric $g_W(\cdot, \cdot): T_W \bm{\mathcal{W}} \times T_W \bm{\mathcal{W}} \rightarrow \mathbb{R}$ for each $W \in \bm{\mathcal{W}}$ such that,
$$
\begin{align*}
    ||U|| &= \sqrt{g_W(U, U)}&&\forall U \in T\_W\mathcal{W}\\\\
    \text{and}\quad g_W(U, V) &= \langle U, V \rangle_{G_W} = \langle G_W U, V \rangle_F = \text{tr}(U^T G_W V) &&\forall U,V \in T\_W\mathcal{W}
\end{align*}
$$
for some (symmetric) positive-definite matrix $G_W$ that may depend on the point $W$.

> **Special Cases.**
> 1. *Euclidean Manifold:* $G_W = I$ for all $W \in \bm{\mathcal{W}}$. In this case we have,
$$||U|| = \sqrt{g_W(U, U)} = \sqrt{\langle I U, U \rangle_F} = \sqrt{\langle U, U \rangle_F} = ||U||_F,\quad\forall U \in T_W\mathcal{W}$$
which is simply the Euclidean case above.
> 2. *Euclidean Manifold in Disguise:* $G_W$ is a constant matrix, i.e. it does not depend on $W$, but may not be the identity matrix. Since the metric matrix $G_W$ is guaranteed to be (symmetric) positive-definite, we can always factor it as $G_W = C^T C$ for some invertible matrix $C$. Thus,
$$||U|| = \sqrt{g_W(U, U)} = \sqrt{\text{tr}(U^T C^T C U)} = \sqrt{\langle \overline{U}, \overline{U} \rangle_F} = ||\overline{U}||_F\quad\forall U \in T\_W\mathcal{W}$$
where $\overline{U} = CU\in CT_W\mathcal{W}$. This means that, up to a simple, linear change of coordinates, this case is equivalent to the Euclidean case above.
> 
> Our proofs below still hold in these special cases. But note that, the metric matrix $G_W$ may depend on the point $W \in \mathcal{W}$ and thus potentially induce a non-zero curvature somewhere on the manifold, making it non-Euclidean.

An interesting property of Riemannian manifolds is that we have a canonical bijection between differentials $d\mathcal{L}_W(\cdot) \in T_W^* \bm{\mathcal{W}}$ and gradients $\nabla \mathcal{L}(W) \in T_W \bm{\mathcal{W}}$ such that,
$$d\mathcal{L}_W(\cdot) = \langle \nabla \mathcal{L}(W), \cdot \rangle.$$

Now notice that,
$$
\begin{align*}
    d\mathcal{L}_W(\cdot) &= \langle \nabla \mathcal{L}(W), \cdot \rangle\\\\
    d\mathcal{L}_W(\cdot) &= \langle \underbrace{G_W\nabla \mathcal{L}(W)}\_{\partial\mathcal{L}(W)\_{\text{coord}}}, \cdot \rangle_F\\\\
    G_W\nabla \mathcal{L}(W) &= \partial\mathcal{L}(W)\_{\text{coord}}\\\\
    \nabla \mathcal{L}(W) &= G_W^{-1} \partial\mathcal{L}(W)\_{\text{coord}}\\\\
\end{align*}
$$

Thus,

$$
\begin{align}
    \widehat{\Delta W} &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} d\mathcal{L}\_W(\Delta W)\nonumber\\\\
        &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} \langle \nabla \mathcal{L}(W), \Delta W \rangle\nonumber\\\\
        &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} \langle G_W^{-1} \partial\mathcal{L}(W)\_{\text{coord}}, \Delta W \rangle\\\\
        &= \frac{G_W^{-1} \partial\mathcal{L}(W)\_{\text{coord}}}{||G_W^{-1}\partial\mathcal{L}(W)\_{\text{coord}}||}\nonumber\\\\
    \widehat{\Delta W} &\approx \frac{G_W^{-1}\partial\mathcal{L}(W; \xi)\_{\text{coord}}}{||G_W^{-1}\partial\mathcal{L}(W; \xi)\_{\text{coord}}||}\nonumber\\\\
\end{align}
$$
where the maximum above can be achieved by aligning $\Delta W$ with $G_W^{-1}\partial\mathcal{L}(W)\_{\text{coord}}$ and scaling such that $||\Delta W|| = 1$ (Absil, 2008). Thus our update rule becomes,
$$W_{t+1} = W\_t - \hat{\lambda} G_{W\_t}^{-1}\partial\mathcal{L}(W_t)\_{\text{coord}}$$
where $\hat{\lambda} = \frac{\lambda}{||G_{W\_t}^{-1}\partial\mathcal{L}(W_t)\_{\text{coord}}||}$. This is Riemannian Stochastic Gradient Descent (RSGD) with an adaptive learning rate. And if we let $P_W = G_W^{-1}$ be the preconditioner at point $W$, we can relate this to Preconditioned Stochastic Gradient Descent (PSGD) algorithms (Li, 2015; Pooladzandi et al., 2024).

> **Important Takeaway:** In a Riemannian manifold, the preconditioner $P_W$ is *unique* and *well-defined* at each point $W \in \mathcal{W}$, but *may not be constant* across the manifold. Thus, as we move across the manifold, we may need to recompute or update our running estimate of the preconditioner. However, if our updates are "small-enough" by some definition, then we may not need to update it at every step; near convergence, or even earlier, it would suffice to update only every $K > 1$ steps for some positive integer $K$ chosen a priori.

### 3.3. $\bm{\mathcal{W}}$ is a Non-Riemannian Manifold

In this case, our choice of norm $||\cdot||$ does not admit a well-behaved metric $g_W(\cdot, \cdot)$ and consequently also does not admit a well-behaved inner product $\langle \cdot, \cdot \rangle$ such that $||\cdot|| = \sqrt{\langle \cdot, \cdot \rangle}$ for all $W \in \mathcal{W}$. Our differentials $d\mathcal{L}_W(\cdot)$ are still well-defined, but we no longer have the bijective relationship between differentials and gradients. And so, we do not always have a unique $V \in T_W\mathcal{W}$ such that $d\mathcal{L}_W(\cdot) = \langle V, \cdot \rangle$ if this inner product even exists.

While we still have access to the stochastic estimator of the differential in standard Euclidean coordinates $\partial\mathcal{L}(W)\_{\text{coord}}$ from Assumption 1, it no longer has geometric meaning by itself. More precisely, we no longer have information on how to transform $\partial\mathcal{L}(W)\_{\text{coord}}$ to the direction of steepest descent by a (possibly point-dependent) change of coordinates on the tangent space $T_W\mathcal{W}$.

We can, however, still use Assumption 1 to define a dualizer for our norm, $\text{dualizer}\_{||\cdot||}(\cdots; W) : T^\*\_W\mathcal{W} \rightarrow T\_W\mathcal{W}$ for $W \in \mathcal{W}$, that maps the differential we get empirically to a direction of steepest descent,
$$
\begin{align*}
    \widehat{\Delta W}
        &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} d\mathcal{L}\_W(\Delta W)\\\\
        &= \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} \langle \partial\mathcal{L}(W)\_{\text{coord}}, \Delta W \rangle_F\\\\
        &\approx \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} \langle \partial\mathcal{L}(W; \xi)\_{\text{coord}}, \Delta W \rangle_F\\\\
    \widehat{\Delta W} &= \text{dualizer}\_{||\cdot||}(\partial\mathcal{L}(W; \xi)\_{\text{coord}}; W)
\end{align*}
$$
where,
$$
\begin{equation}
    \text{dualizer}\_{||\cdot||}(\cdots; W) = \arg\max_{\substack{\Delta W \in T_W\mathcal{W}\\\\ ||\Delta W|| = 1}} \langle \cdots, \Delta W \rangle_F
\end{equation}
$$
and to simplify our notation, we use $\text{dualizer}\_{||\cdot||}(\cdots)$ if this map is independent of $W$.

> **Important Takeaways:**
> 1. In the Riemannian case, the dualization is equivalent to preconditioning. But in the non-Riemannian case, the dualizer may not be linear and may even have multiple solutions! However, as we will discuss in the next section, it may suffice to approximate the dualizer with a (linear) preconditioner in practice.
> 2. The intuitions we have developed for the Riemannian case may not apply in the non-Riemannian case. Dualizers in the non-Riemannian case may behave counterintuitively, and may not even have approximate (linear) preconditioners that would work well in practice. And so, we must be vigilant about the geometry of the manifold we are working with.
> 3. Instantaneous (i.e. preconditioner-free) dualization like what Muon does may be more appropriate in cases where the dualizer has a closed form solution *and* is cheap-enough to compute, according to one's compute budget.

## 4. Muon as Steepest Descent in a Non-Riemannian Manifold

### 4.1. The Muon Optimizer

> **Algorithm 1 (Muon)** by Jordan et al. (2024a). The weights are treated independently.
> 
> **Inputs:** Initial weight $W_0 \in \mathcal{W}$, and momentum term $M_0 \in \mathcal{W}$.
> 
> **Parameters:** Learning rate $\lambda > 0$, momentum decay $\beta \in [0, 1)$, and number of iterations $T \in \\{1, 2, \ldots\\}$
> 
> $\textbf{for } t = 0, 1, \ldots, T-1 \textbf{ do}\\\\
\text{... Compute }G_t = \partial\mathcal{L}(W; \xi)\_{\text{coord}}\\\\
\text{... Compute }W\_{t+1}\text{ and }M\_{t+1}\text{ as follows:}\\\\
\text{....... }M_{t+1} = \beta M_t + (1 - \beta) G_t\\\\
\text{....... }O_{t+1} = \text{approx-orth}(M\_{t+1})\\\\
\text{....... }W_{t+1} = W_t - \lambda O\_{t+1}
$
>
> **Output:** $W_T \in \mathcal{W}$.

> **Algorithm 2 (Approximate Orthogonalization through Newton-Schulz Iteration)**
>
> ```py
> def zeropower_via_newtonschulz(G: Tensor, steps: int=5):
>     assert G.ndim == 2
>     a, b, c = (3.4445, -4.7750, 2.0315)
>     X = G.bfloat16()
>     X /= (X.norm() + 1e-7)
>     if G.size(-2) > G.size(-1):
>         X = X.mT
>     for _ in range(steps):
>         A = X @ X.mT
>         B = b * A + c * A @ A
>         X = a * X + B @ X
>     if G.size(-2) > G.size(-1):
>         X = X.mT
>     return X
> ```

Muon (Algorithm 1) is an optimizer for matrix-valued parameters in neural networks (Jordan et al., 2024a). For each weight $W \in \mathcal{W}$, it first accumulates the momentum term, then approximately semi-orthogonalizes the result using the Newton-Schulz iteration (Algorithm 2), before applying it as an update to the weights.

We can fold the momentum term into $\partial\mathcal{L}(W; \xi)\_{\text{coord}}$ as it can be seen as a way to smooth out outlier empirical gradients. In fact, Mokhtari et al. (2018) and more recently Kovalev (2025) have shown that, under Muon's update rule, the momentum term does become a tighter approximation of the true gradient $\partial\mathcal{L}(W)\_{\text{coord}}$ as the number of iterations $T$ increases.

And while Muon only approximately (semi-)orthogonalizes the gradient, we have found that it still empirically performs just as well as exact orthogonalization. We will discuss this in more detail in the next sections. Muon is also not the first optimizer that does approximate orthogonalization. For example, Carlson et al.'s randomized algorithm Sketching (2015) does this explicitly, and so does Shampoo (Gupta et al., 2018), CASPR (Surya et al., 2024), and PSGD (Li, 2015) implicitly through their preconditioners. However, Muon is the first, non-randomized, preconditioner-free optimizer that explicitly aims to orthogonalize the gradient.

An interesting fact from prior work (Carlson et al., 2015; Flynn, 2017; Mokhtari et al., 2018, Bernstein et al., 2024) is that the dualizer for steepest descent under the spectral norm $||\cdot||_{2 \to 2}$ is exactly this orthogonalization process,
$$
\begin{equation}
    \text{dualizer}\_{||\cdot||\_{2\to 2}}(\partial\mathcal{L}(W; \xi)\_{\text{coord}}) = UV^T
\end{equation}
$$
where $U\Sigma V^T$ is the singular value decomposition (SVD) of $\partial\mathcal{L}(W; \xi)\_{\text{coord}}$. The spectral norm does not admit a well-behaved inner product. And so, Muon, and related optimizers, can be thought of as steepest descent in a non-Riemannian manifold.

In the next section, we discuss why Muon can be viewed as an instantaneous version of already existing optimizers such as Shampoo, CASPR, PSGD, and etc. We will also discuss an alternate perspective on how such preconditioning optimizers can be viewed as approximators to the dualizer of the spectral norm.

### 4.2 Approximating Dualization with Preconditioners

As we have shown in Section 3.2, the dualization process in Riemannian steepest descent always has an equivalent preconditioning process by letting $P_W = G_W^{-1}$ at every point $W \in \bm{\mathcal{W}}$. And likewise, if we have a preconditioning process where every $P_W$ is invertible, then it can be thought of as Riemannian steepest descent under the metric $G_W = P_W^{-1}$.

However, we may not always have an equivalent preconditioning process in non-Riemannian manifolds. And if there is, it may not be unique. Here, let's examine multiple preconditioning processes that approximate the dualizer under the spectral norm in turn.

**4.2.1. The matrix sign function.**

> **Definition (Matrix Sign Function)** Let $X = \mathbb{R}^{m \times n}$. The matrix sign function is defined as,
$$
\begin{align*}
    \text{msign}(X) &= (XX^T)^{-1/2} X = X (X^T X)^{-1/2}\\\\
    \text{msign}(X) &= U V^T
\end{align*}
$$
where $U\Sigma V^T$ is the singular value decomposition (SVD) of $X$.

Thus, one can interpret Muon's update rule as simply the matrix sign function applied to $\partial\mathcal{L}(W_t)\_{\text{coord}}$. And if we let $G_t = \partial\mathcal{L}(W_t)\_{\text{coord}}$ and $P\_{W_t} = (G_t G_t^T)^{-1/2}$, then we arrive at a preconditioner form of Muon's update rule,
$$
\begin{align*}
    W_{t+1} &= W_t - \lambda P_{W_t} \partial\mathcal{L}(W_t)\_{\text{coord}}\\\\
        &= W_t - \lambda (\partial\mathcal{L}(W_t)\_{\text{coord}} \partial\mathcal{L}(W_t)\_{\text{coord}}^T)^{-1/2} \partial\mathcal{L}(W_t)\_{\text{coord}}\\\\
    W_{t+1} &= W_t - \lambda UV^T
\end{align*}
$$

**4.2.2. Shampoo.** Let $G_t = \partial\mathcal{L}(W_t)\_{\text{coord}}$. Then Shampoo (Gupta et al., 2018; Anil et al., 2020) has the following update rule,

$$L_t := L_{t-1} + G_t G_t^T, \quad\quad R_t := R_{t-1} + G_t^T G_t$$
$$\Delta W_t = L^{-1/4}_t G_t R^{-1/4}_t$$

As previously noted by Bernstein et al. (2024) and Anil (2024), Shampoo reduces to Muon if we disable the updates on the left and right preconditioners. That is, let $G_t = U\Sigma V^T$ be the singular value decomposition (SVD) of $G_t$ and let
$$
L_t := G_t G_t^T\quad\quad R_t := G_t^T G_t
$$
Then,
$$\begin{aligned}
    \Delta W_t &= L^{-1/4}_t G_t R^{-1/4}_t\\\\
        &= (G_t G_t^T)^{-1/4} G_t (G_t^T G_t)^{-1/4}\\\\
        &= (U\Sigma V^T V \Sigma U^T)^{-1/4} U\Sigma V^T (V \Sigma U^T U \Sigma V^T)^{-1/4}\\\\
        &= U\left(\frac{\Sigma}{\sqrt{\Sigma^2}} \right)V^T\\\\
    \Delta W_t &= UV^T
\end{aligned}$$
which is Muon's update rule. $\blacksquare$

**4.2.3. CASPR.** Let $G_t = \partial\mathcal{L}(W_t)\_{\text{coord}}$. Then CASPR (Surya et al., 2024) has the following update rule,

$$L_t := L_{t-1} + G_t G_t^T, \quad\quad R_t := R_{t-1} + G_t^T G_t$$
$$\tilde{L}_t := L_t + \epsilon I_m, \quad\quad \tilde{R}_t := R_t + \epsilon I_n$$
$$\Delta W_t = (\tilde{L}^{-1/2}_t G_t + 2 \tilde{L}^{-1/4}_t G_t \tilde{R}^{-1/4}_t + G_t \tilde{R}^{-1/2}_t)/4.$$

As previously noted by Cesista (2025), CASPR reduces to Muon if we disable the updates on the left and right preconditioners. That is, let $G_t = U\Sigma V^T$ be the singular value decomposition (SVD) of $G_t$ and let
$$
L_t := G_t G_t^T\quad\quad R_t := G_t^T G_t
$$
Then,
$$\begin{aligned}
    \Delta W_t
        &= (\tilde{L}^{-1/2}_t G_t + 2 \tilde{L}^{-1/4}_t G_t \tilde{R}^{-1/4}_t + G_t \tilde{R}^{-1/2}_t)/4\\\\
        &= (1/4) \cdot [(G_t G_t^T + \epsilon I_m)^{-1/2} G_t\\\\
            &\quad\quad\quad+ 2 (G_t G_t^T + \epsilon I_m)^{-1/4} G_t (G_t^T G_t + \epsilon I_n)^{-1/4}\\\\
            &\quad\quad\quad+ G_t (G_t^T G_t + \epsilon I_n)^{-1/2}]\\\\
        &= (1/4) \cdot [[(U \Sigma V^T) (U \Sigma V^T)^T + \epsilon U I U^T]^{-1/2}(U \Sigma V^T)\\\\
            &\quad\quad\quad + 2[(U \Sigma V^T) (U \Sigma V^T)^T + \epsilon U I U^T]^{-1/4}(U \Sigma V^T) [(U \Sigma V^T)^T (U \Sigma V^T) + \epsilon V I V^T]^{-1/4}\\\\
            &\quad\quad\quad + (U \Sigma V^T) [(U \Sigma V^T)^T (U \Sigma V^T) + \epsilon V I V^T]^{-1/2}]\\\\
        &= (1/4) \cdot [U(\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T + U(2\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T + U(\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T]\\\\
        &= (1/4) \cdot [U(1+2+1)(\Sigma(\Sigma^2 + \epsilon I)^{-1/2})V^T]\\\\
        &= U\left(\frac{\Sigma}{\sqrt{\Sigma^2 + \epsilon I}} \right)V^T \\\\
    \Delta W_t &\approx UV^T
\end{aligned}$$
which is Muon's update rule. $\blacksquare$

**4.2.4. PSGD Family. [Under Review]** This family of optimizers (Li, 2015 & 2018; Pooladzandi, 2024) explicitly tries to learn the preconditioner $\mathcal{P}(\cdot; W)$ according to some criterion to ensure training stability and, potentially, faster convergence.  This criterion is involved with the noise suppression gain which is defined as,
$$
\text{noise\\_suppresion\\_gain}\_{||\cdot||_F}(P)
    = \frac{\mathbb{E}[||H_0^{-1}\epsilon'||_F^2]}{\mathbb{E}[||P\epsilon'||_F^2]}
    = \frac{\mathbb{E}[(\epsilon')^T H_0^{-2} \epsilon']}{\mathbb{E}[(\epsilon')^T P^2 \epsilon']},
$$
where $\epsilon'$ is some (matrix-valued) noise term on the Hessian $H$, which aims to reduce the noise of the preconditioned gradients.

We get different update rules depending on which Lie group we restrict the preconditioner $P$ to. However, the criterion above typically leads to update rules that *whitens*, i.e. decorrelates, the entries of the gradient $\partial\mathcal{L}(W; \xi)\_{\text{coord}}$. And so while Muon can be seen as an instantaneous version of PSGD, an important difference is that Muon merely projects the gradient to its nearest (semi-)orthogonal matrix, but not necessarily decorrelates the entries.

For future work, it would also be interesting to see what kind of update rules we get if we measure the noise suppression gain with respect to the spectral norm instead of the Frobenius norm. That is,
$$
\text{noise\\_suppresion\\_gain}\_{||\cdot||\_{2\to 2}}(P) = \frac{\mathbb{E}[||H_0^{-1}\epsilon'||\_{2\to 2}]}{\mathbb{E}[||P\epsilon'||\_{2\to 2}]}
$$

## 5. Steepest Descent under Elementwise $p$-Norms and Schatten-$p$ Norms

> **Definition 2 (Vector $p$-Norms).** Given $p \in [1, \infty]$, the vector $p$-norm of a finite-dimensional, real-valued vector $\bm{x} \in \mathbb{R}^n$ is defined as,
> $$
||\bm{x}||\_p = \begin{cases}
    \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p} & \text{if } 1 \leq p < \infty\\\\
    \max_{i} |x_i| & \text{if } p = \infty
\end{cases}
$$
>
> **Examples:**
> 1. $p = 1$: The Manhattan/Taxicab norm, $||\bm{x}||\_1 = \sum_{i=1}^{n} |x_i|$
> 2. $p = 2$: The Euclidean norm, $||\bm{x}||\_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2} = ||\bm{x}||\_F$
> 3. $p = \infty$: The Max norm, $||\bm{x}||\_\infty = \max_{i} |x_i|$

> **Definition 3 (Matrix Elementwise $p$-Norms).** Given $p = [1, \infty]$, the elementwise $p$-norm of a finite-dimensional, real-valued matrix $X \in \mathbb{R}^{m \times n}$ is defined as,
> $$
||X||\_{e,p} = \begin{cases}
    \left(\sum_{i=1}^{m} \sum_{j=1}^{n} |X_{ij}|^p\right)^{1/p} & \text{if } 1 \leq p < \infty\\\\
    \max_{i,j} |X_{ij}| & \text{if } p = \infty
\end{cases}
$$
or equivalently,
> $$||X||\_{e,p} = ||\text{vec}(X)||\_p,$$
> where $\text{vec}(\cdot)$ is the vectorization operator that stacks the columns of $X$ into a single vector.
>
> **Examples:**
> 1. $p = 1$: The Sum-of-Sums norm, $||X||\_{e,1} = \sum_{i=1}^{m} \sum_{j=1}^{n} |X_{ij}|$
> 2. $p = 2$: The Frobenius norm, $||X||\_{e,2} = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |X_{ij}|^2} = ||X||\_F$
> 3. $p = \infty$: The Max-of-Max norm, $||X||\_{e,\infty} = \max_{i,j} |X_{ij}|$

> **Definition 4 (Schatten-$p$ Norms).** Given $p = [1, \infty]$, the Schatten-$p$ norm of a finite-dimensional, real-valued matrix $X \in \mathbb{R}^{m \times n}$ is defined as,
$$
||X||\_{S_p} = \begin{cases}
    \left(\sum_{i=1}^{\min(m, n)} |\sigma_i(X)|^p\right)^{1/p} & \text{if } 1 \leq p < \infty\\\\
    \max_{i} \sigma_i(X) & \text{if } p = \infty
\end{cases}
$$
where $\sigma(X) = (\sigma_1(X), \ldots, \sigma_{\min(m,n)}(X))$ are the singular values of $X$. Or equivalently,
> $$||X||\_{S_p} = ||\sigma(X)||_p$$
> 
> **Examples:**
> 1. $p = 1$: The Nuclear norm, $||A||\_{S_1} = \sum_{i=1}^{\min(m,n)} |\sigma_i(A)| = ||A||_{\text{nuc}}$
> 2. $p = 2$: The Frobenius norm, $||A||\_{S_2} = \left(\sum_{i=1}^{\min(m,n)} |\sigma_i(A)|^2\right)^{\frac{1}{2}} = ||A||\_F$
> 3. $p = \infty$: The Spectral norm, $||A||\_{S_{\infty}} = \max_{i} \sigma_i(A) = ||A||\_{2 \to 2}$

A special case is when $p = 2$, and so we have the Frobenius norm. Equipping $\mathcal{W}$ with this norm gives us the standard Euclidean manifold, which is Riemannian. However, Propositions (6) and (7) below still applies.

And to find the dualizers for the Schatten-$p$ norms, we will use the following inequality,

> **Theorem 5 (von Neumann's Trace Inequality).** Let $A, B \in \mathbb{R}^{m \times n}$. Then the following inequality holds,
> $$\text{tr}(A^TB) \leq \sum_{i=1}^{\min(m,n)} \sigma_i(A) \sigma_i(B),$$
> where $\sigma(A) = (\sigma_1(A), \ldots, \sigma_{\min(m,n)}(A))$ and $\sigma(B) = (\sigma_1(B), \ldots, \sigma_{\min(m,n)}(B))$ are the singular values of $A$ and $B$, respectively. And equality holds if and only if $A$ and $B$ share singular vectors. If so, then,
> $$
\begin{align*}
    \text{tr}(A^TB) &= \sum_{i=1}^{\min(m,n)} \sigma_i(A) \sigma_i(B)\\\\
    \langle A, B\rangle_F &= \langle \sigma(A), \sigma(B) \rangle_F
\end{align*}
$$

### 5.1. Dualizers for Elementwise $p$-Norms and Schatten-$p$ Norms

> **Proposition 7.** Given $p = [1, \infty]$, the dualizer for the Schatten-$p$ norm is:
$$
\text{dualizer}\_{||\cdot||\_{S_p}}(X) = \begin{cases}
    U \frac{\text{diag}\left(\sigma\_1(X)^{q-1}, \ldots, \sigma\_{\min(m,n)}(X)^{q-1}\right)}{||X||\_{S_q}^{q-1}} V^T & \text{if } 1 \leq p < \infty\\\\
    UV^T & \text{if } p = \infty
\end{cases}
$$
where $\frac{1}{p} + \frac{1}{q} = 1$, and $X = U\Sigma V^T$ is the singular value decomposition of $X \in \mathbb{R}^{m \times n}$.
> 
> **Proof:** For a given $X \in T_W\mathcal{W}$ at $W \in \mathcal{W}$, let $T^\* \in T_W\mathcal{W}$ be,
$$
\begin{align*}
    T^* &= \text{dualizer}\_{||\cdot||\_{S_p}}(X; W)\\\\
    T^* &= \arg\max\_{\substack{T \in T_W\mathcal{W}\\\\ ||T||\_{S_p} = 1}} \langle X, T \rangle_F\\\\
    T^* &= \arg\max\_{\substack{T \in T_W\mathcal{W}\\\\ ||T||\_{S_p} = 1}} \text{tr}(X^T T)
\end{align*}
$$
Then from von Neumann's Trace Inequality, we know that $T^\*$ must share singular vectors with $X$ and that,
$$
\begin{align*}
    T^* &= \arg\max\_{\substack{T \in T_W\mathcal{W}\\\\ ||T||\_{S_p} = 1}} \sum\_{i=1}^{\min(m,n)} \sigma_i(X) \sigma_i(T)\\\\
    T^* &= \arg\max\_{\substack{T \in T_W\mathcal{W}\\\\ ||\sigma(T)||\_{p} = 1}} \langle \sigma(X), \sigma(T) \rangle_F
\end{align*}
$$
Thus, our optimization problem reduces to,
$$\arg\max\_{\sigma(T)} \sum\_{i=1}^{\min(m,n)} \sigma\_i(X) \sigma_i(T) \quad\text{s.t.}\quad \sum\_{i=1}^{\min(m,n)} \sigma\_{i}(T)^p = 1$$
And solving via Lagrange multipliers, we have,
$$\sigma_i(T) = \frac{\sigma_i(X)^{q-1}}{||X||\_{S_q}^{q-1}}$$
where $\frac{1}{p} + \frac{1}{q} = 1$. Note that this is indepdent of $W$. Hence,
$$T^* = \text{dualizer}\_{||\cdot||\_{S_p}}(X) = U \frac{\text{diag}\left(\sigma\_1(X)^{q-1}, \ldots, \sigma\_{\min(m,n)}(X)^{q-1}\right)}{||X||\_{S_q}^{q-1}} V^T\quad\blacksquare$$

### 5.2. Stochastic Gradient Descent and Muon as Special Cases of Steepest Descent under Schatten-$p$ Norms

**5.2.1. Recovering SGD.** Let $p = 2$, and so we have $q = 2$ and $||\cdot||\_{S_2} = ||\cdot||\_F$. Thus for $\partial\mathcal{L}(W; \xi)\_{\text{coord}} \in T^\*\_W\mathcal{W}$ at $W \in \mathcal{W}$, we have,
$$
\begin{align*}
    \Delta W
        &= \text{dualizer}\_{||\cdot||\_{S_\infty}}(\partial\mathcal{L}(W; \xi)\_{\text{coord}}; W)\\\\
        &= U \frac{\text{diag}\left(\sigma\_1(\partial\mathcal{L}(W; \xi)\_{\text{coord}})^{2-1}, \ldots, \sigma\_{\min(m,n)}(\partial\mathcal{L}(W; \xi)\_{\text{coord}})^{2-1}\right)}{||\partial\mathcal{L}(W; \xi)\_{\text{coord}}||\_{S_2}^{2-1}} V^T\\\\
    \Delta W &= \frac{\partial\mathcal{L}(W; \xi)\_{\text{coord}}}{||\partial\mathcal{L}(W; \xi)\_{\text{coord}}||\_F}
\end{align*}
$$
which matches the update rule we expect from SGD.

**5.2.2. Recovering Muon.** Let $p = \infty$, and so we have $q = 1$ and $||\cdot||\_{S\_\infty} = ||\cdot||\_{2\to 2}$. Thus for $\partial\mathcal{L}(W; \xi)\_{\text{coord}} \in T^\*\_W\mathcal{W}$ at $W \in \mathcal{W}$, we have,
$$
\begin{align*}
    \Delta W
        &= \text{dualizer}\_{||\cdot||\_{2 \to 2}}(\partial\mathcal{L}(W; \xi)\_{\text{coord}}; W)\\\\
        &= U \frac{\text{diag}\left(\sigma\_1(\partial\mathcal{L}(W; \xi)\_{\text{coord}})^{1-1}, \ldots, \sigma\_{\min(m,n)}(\partial\mathcal{L}(W; \xi)\_{\text{coord}})^{1-1}\right)}{||\partial\mathcal{L}(W; \xi)\_{\text{coord}}||\_{S_1}^{1-1}} V^T\\\\
    \Delta W &= UV^T
\end{align*}
$$
which matches the update rule we expect from Muon.

### 5.3. Why Muon Still Works Well in Practice Despite the Approximate Orthogonalization

We observe that, qualitatively, steepest descent under the Schatten-$p$ norm very quickly converges to steepest descent under the spectral norm as $p$ approaches $\infty$. This is probably why optimizers like Sketching and Muon work well in practice despite not perfectly orthogonalizing the gradients.

To support this, we show that the (1) variance of singular values post-dualization, and the (2) relative size, and (3) stable rank of the dualized gradients under the Schatten-$p$ norm quadratically converges to those of the Spectral norm as $p$ approaches $\infty$. And in fact, at $p = 32$, the results are already very close to those of the Spectral norm.

**5.3.1. On the variance of singular values post-dualization**

> **Proposition 8.** The variance of the singular values post-dualization under the Schatten-$p$ Norm converges quadratically to $0$ as $p$ approaches $\infty$.

> **Proof:** Let $X \in \mathbb{R}^{m \times n}$ and let $t_i$ be the $i$-th singular value post-dualization. From Proposition 7 earlier, we have
$$
\begin{align*}
    t_i &= \left(\frac{\sigma_i(X)}{||X||\_{S_q}}\right)^{q-1}\\\\
    t_i &= \exp\left((q-1)\ln\frac{\sigma_i(X)}{||X||\_{S_q}}\right)\\\\
    t_i &\approx 1 + (q-1)\ln\frac{\sigma_i(X)}{||X||\_{S_q}}
\end{align*}
$$
where the last line follows from first-order Taylor approximation of $t_i$. Thus, the mean and variance are:
$$
\begin{align*}
    \mathbb{E}[t_i] &\approx 1 + (q-1)\mathbb{E}\left[\ln\frac{\sigma_i(X)}{||X||\_{S_q}}\right]\\\\
    \mathbb{E}[t_i] &\approx 1 + (q-1)\ln\frac{\mathbb{E}[\sigma_i(X)]}{||X||\_{S_q}}\\\\
    t_i - \mathbb{E}[t_i] &\approx (q-1)\ln\left[\sigma_i(X) - \mathbb{E}[\sigma_i(X)]\right]\\\\
    Var[t_i] &\approx (q-1)^2\mathbb{E}\left[\ln^2\left[\sigma_i(X) - \mathbb{E}[\sigma_i(X)]\right]\right]\\\\
    Var[t_i] &\approx \frac{1}{(p-1)^2}\mathbb{E}\left[\ln^2\left[\sigma_i(X) - \mathbb{E}[\sigma_i(X)]\right]\right]
\end{align*}
$$
Hence, the variance of the singular values post-dualization converges quadratically to $0$ as $p$ approaches $\infty$.

Empirically, we can see this in the following plot. And at $p = 32$, the variance of the resulting singular values are already close to $0$.
![](../steepest-descent-schatten-p/var_sv_dualizer.png#center)

**5.3.2. On the relative size and stable rank of gradients**

> **Definition 6: Relative Size of a Gradient.** Given a norm $||\cdot||$ chosen a priori, the relative size of a gradient-update $\Delta W$ relative to the parameter matrix $W$ is defined as:
$$\text{relsize}(\Delta W) = \frac{||\Delta W||}{||W||}$$

> **Definition 7: Stable Rank.** The stable rank of a matrix $A$ is defined as $$srank(A) = \frac{||A||\_F^2}{||A||\_{2 \to 2}^2}$$

As we can see in the following plot, the raw gradients have very low-stable rank. But the stable rank of the gradients post-dualization under the Schatten-$p$ norm converges very quickly to that of the Spectral norm as $p$ approaches $\infty$.

![](../steepest-descent-schatten-p/srank_sv_dualizer.png#center)

One can interpret this as, for some large enough $p$, the dualized gradient is already very close to being "maximal" in a sense. And increasing $p$ further would only offer rapidly diminishing returns.

A side-effect of this is that it allows the model parameters to "escape" the small region around the initialization and explore the parameter space more effectively, contrary to prior work on SGD and Adam optimizers (Lee et al., 2020; Jesus et al., 2021). This phenomenon is known as *weight erasure* (Bernstein, 2024).

![](../steepest-descent-schatten-p/weight-erasure.png)

## 6. Optimizing Muon's Newton-Schulz Coefficients [Under Construction]

## 7. Convergence Guarantees [Under Construction]

## Acknowledgements

Many thanks to Omead Pooladzandi, Simo Ryu, and Antonio Silveti-Falls for their feedback on this work. I have been (and still is) trying to incorporate their feedback into this work. And also to Jeremy Bernstein and Keller Jordan for conversations on optimization and machine learning, in general.

## How to Cite

```bibtex
@misc{cesista2025sdnr,
  author = {Franz Louis Cesista},
  title = {Muon and a Selective Survey on Steepest Descent in Riemannian and Non-Riemannian Manifolds},
  year = {2025},
  url = {http://leloykun.github.io/ponder/steepest-descent-non-riemannian/},
}
```

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
2. Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and @fernbear.bsky.social and Boza Vlado and You Jiacheng and Franz Cesista and Braden Koszarsky and @Grad62304977 (2024). modded-nanogpt: Speedrunning the NanoGPT baseline. URL https://github.com/KellerJordan/modded-nanogpt/
3. Diederik P. Kingma, Jimmy Ba (2014). Adam: A Method for Stochastic Optimization. URL https://arxiv.org/abs/1412.6980
4. Moonshot AI Team (2025). Muon is Scalable for LLM Training. URL https://arxiv.org/abs/2502.16982
5. Jeremy Bernstein and Laker Newhouse. “Old optimizer, new norm: An anthology.” arXiv preprint arXiv:2409.20325 (2024).
6. Jeremy Bernstein, Laker Newhouse (2024). Modular Duality in Deep Learning. URL https://arxiv.org/abs/2410.21265
7. Jeremy Bernstein (2024). "Weight erasure." Available at: https://docs.modula.systems/examples/weight-erasure/
8. Greg Yang, James B. Simon, Jeremy Bernstein (2024). A Spectral Condition for Feature Learning. URL https://arxiv.org/abs/2310.17813
9. Xi-Lin Li (2015). Preconditioned Stochastic Gradient Descent. URL https://arxiv.org/abs/1512.04202
10. Xi-Lin Li (2018). Preconditioner on Matrix Lie Group for SGD. URL https://arxiv.org/abs/1809.10232
11. Omead Pooladzandi, Xi-Lin Li (2024). Curvature-Informed SGD via General Purpose Lie-Group Preconditioners. URL https://arxiv.org/abs/2402.04553
12. P.-A. Absil, R. Mahony, and Rodolphe Sepulchre (2008). Optimization Algorithms on Matrix Manifolds. Princeton University Press.
13. David E Carlson, Edo Collins, Ya-Ping Hsieh, Lawrence Carin, Volkan Cevher. Preconditioned Spectral Descent for Deep Learning. In Advances in Neural Information Processing Systems 28 (NIPS 2015), 2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html
14. Thomas Flynn. The duality structure gradient descent algorithm: Analysis and applications to neural networks. arXiv:1708.00523, 2017. URL https://arxiv.org/abs/1708.00523
15. Hunter, D. R. and Lange, K. (2004). A tutorial on MM algorithms. The American Statistician, 58(1):30–37.
16. Elhage, et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022.
17. Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano, Tolga Birdal (2025). Grokking at the Edge of Numerical Stability. URL https://arxiv.org/abs/2501.04697
18. Vineet Gupta, Tomer Koren, Yoram Singer (2018). Shampoo: Preconditioned Stochastic Tensor Optimization. URL https://arxiv.org/abs/1802.09568
19. Rohan Anil et al. “Scalable second order optimization for deep learning.” arXiv preprint arXiv:2002.09018 (2020).
20. Surya, S., Duvvuri, Devvrit, F., Anil, R., Hsieh, C., & Dhillon, I.S. (2024). Combining Axes Preconditioners through Kronecker Approximation for Deep Learning. International Conference on Learning Representations.
21. Franz Louis Cesista (2025). {CASPR} Without Accumulation is {M}uon. URL https://leloykun.github.io/ponder/caspr-wo-accum-is-muon/
22. Rohan Anil. “Just some fun linear algebra.” X post, 6 Oct. 2024, Available at: https://x.com/_arohan_/status/1843050297985466565.
23. Dmitry Kovalev (2025). Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization. Available at: https://arxiv.org/abs/2503.12645
24. Lee, Jaehoon, et al. “Wide Neural Networks of Any Depth Evolve as Linear Models under Gradient Descent.” Journal of Statistical Mechanics: Theory and Experiment, vol. 2020, no. 12, Dec. 2020, p. 124002. Crossref, https://doi.org/10.1088/1742-5468/abc62b.
25. Jesus, Ricardo J., et al. “Effect of Initial Configuration of Weights on Training and Function of Artificial Neural Networks.” Mathematics, vol. 9, no. 18, Sept. 2021, p. 2246. Crossref, https://doi.org/10.3390/math9182246.
26. Aryan Mokhtari, Hamed Hassani, Amin Karbasi (2018). Stochastic Conditional Gradient Methods: From Convex Minimization to Submodular Maximization. URL https://arxiv.org/abs/1804.09554
27. David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams (1986). Learning representations by back-propagating errors. URL https://www.nature.com/articles/323533a0
28. Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. https://arxiv.org/abs/1703.03864
29. Cornelius V. Braun, Robert T. Lange, Marc Toussaint (2024). Stein Variational Evolution Strategies. URL https://arxiv.org/abs/2410.10390
