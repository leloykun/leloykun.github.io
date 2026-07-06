---
title: "LoRA-Muon-OGD: Spectral Orthogonal Gradient Projection on the Low-Rank Manifold for LLM Continual Learning"
date: 2026-06-25
tags: ["Machine Learning", "Optimizers", "LoRA", "Low-Rank Adaptation", "Muon", "Continual Learning"]
author: ["Franz Louis Cesista"]
description: "Generalizing Orthogonal Gradient Projection to the low-rank case and to steepest descent under a larger family of norms."
summary: "Generalizing Orthogonal Gradient Projection to the low-rank case and to steepest descent under a larger family of norms."
---

## 1. Introduction

As LLMs are deployed in more diverse and longer-horizon settings, they need to continually acquire new skills without forgetting old ones. But when finetuned on new tasks, LLMs rapidly "forget" previously learned skills. Imagine a person immediately forgetting how to ride a bike as soon as they learn how to catch fish. This phenomenon is called, "catastrophic forgetting" ([Goodfellow et al., 2013](https://arxiv.org/abs/1312.6211), [Kirkpatrick et al., 2017](https://www.pnas.org/doi/10.1073/pnas.1611835114)).

One way to mitigate the catastrophic forgetting issue is to project gradients away from past-task directions $\{ C_i \}_{1 \leq i \leq k}$ via Orthogonal Gradient Descent (OGD) ([Farajtabar et al., 2020](https://proceedings.mlr.press/v108/farajtabar20a.html)). That is, we want our weight updates $\Delta W$ to satisfy,
$$
\begin{equation}
    \langle \Delta W, C_i \rangle = 0 \qquad \text{for all } 1 \leq i \leq k. \label{eq:non-interference}
\end{equation}
$$
I would argue this is somewhat hacky as we may want the LLM to *refine* previously learned skills as it chugs through problems, but it works and simple-enough to be bitter-lesson-pilled. [Lu et al., 2026](https://arxiv.org/abs/2605.08949) recently derived Muon-OGD which takes the maximal updates under the spectral-norm geometry in $\mathbb{R}^{m \times n}$ while satisfying the non-interference constraint in $\eqref{eq:non-interference}$. They report SOTA results on continual learning tasks, but the algorithm requires materializing the dense gradient matrix which makes it unsuitable for low-rank finetuning.

In this work, we derive LoRA-Muon-OGD which takes the maximal updates under the spectral-norm, but on the low-rank manifold $\mathcal{M}_r = \{ W = A B^\top | A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{n \times r}, \operatorname{rank}(A) = \operatorname{rank}(B) = r \}$ (with gauge redundancies, $(A, B) \sim (AR, BR^{-1}) \text{ for all } R \in \operatorname{GL}(r)$) while still satisfying the non-interference constraint in $\eqref{eq:non-interference}$. We also show that the derivation is natural and generalizes to steepest descent under arbitrary unitary-invariant norm.

> Note: After publishing this artcile, I've realized that Muon-OGD is a special case of my work in [Ponder: Steepest Descent on Finsler-Structured (Matrix) Geometries via Dual Ascent](../steepest-descent-finsler-dual-ascent/). We just need to set $L(A) = U^T A V$, $b = 0$, and $K = \{ 0 \}$ in [Section 3.1](../steepest-descent-finsler-dual-ascent/#31-general-strategy). What is new in this article is the low-rank versions of these optimizers and generalization to arbitrary smooth parametrizations.

## 2. Problem setting

Let $f: \mathcal{W} \to \mathbb{R}$ be a differentiable and bounded below objective function defined on a finite-dimensional manifold $\mathcal{W}$ equipped with a norm $\| \cdot \|$.
Let $G_W := \nabla_W f(W)$ be its "Euclidean gradient" at $W \in \mathcal{W}$. In the LoRA setting where $\mathcal{W} = \mathcal{M}_r$, let $G_A := G_W B$ and $G_B := G_W^\top A$ be the "Euclidean gradients" w.r.t. the $A$ and $B$ LoRA factors, respectively.
In practice, when doing LoRA finetuning, backpropagation only gives us access to $G_A$ and $G_B$, not $G_W$, and constructing the full dense 'gradient' matrix is often compute and memory extensive.

Our derivations here are made a lot simpler by the observation that the low-rank constraint and OGD's non-interference constraint, intuitively speaking, commute as optimizer-producing actions.
Starting from the Muon optimizer ([Keller et al., 2024](https://kellerjordan.github.io/posts/muon/)) and applying the low-rank constraint first yields LoRA-Muon ([Cesista et al., 2026](https://arxiv.org/abs/2606.12921))
If we instead apply the non-interference constraint first, we instead get Muon-OGD as in [Lu et al., 2026](https://arxiv.org/abs/2605.08949).
But either way, applying the remaining constraint then yields LoRA-Muon-OGD.

The following trust-region problems yield the four optimizers discussed here.

$$\begin{array}{ccc}
\begin{array}{c}
    \text{Muon:} \\
    \begin{aligned}
        \min_{\Delta W} \hspace{0.5em}
            &\langle G_W, \Delta W \rangle \\
        \text{s.t. }
            &\| \Delta W \|_{2 \to 2} \leq \eta
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon:} \\
    \begin{aligned}
        \min_{\Delta A, \Delta B} \hspace{0.5em}
            &\langle G_W, \Delta A B^\top + A \Delta B^\top \rangle \\
        \text{s.t. }
            &\| \Delta A B^\top \|_{2 \to 2} \leq \frac{\eta}{2}, \\
            &\| A \Delta B^\top \|_{2 \to 2} \leq \frac{\eta}{2}
    \end{aligned}
\end{array}
\\[1.5em]
\Big\downarrow\ {\scriptstyle \text{OGD constraint} }
&
&
\Big\downarrow\ {\scriptstyle \text{OGD constraint} }
\\[1.5em]
\begin{array}{c}
    \text{Muon-OGD:} \\
    \begin{aligned}
        \min_{\Delta W} \hspace{0.5em}
            & \langle G_W, \Delta W \rangle \\
        \text{s.t. }
            & \| \Delta W \|_{2 \to 2} \leq \eta, \\
            & U^\top (\Delta W) V = 0
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon-OGD:} \\
    \begin{aligned}
        \min_{\Delta A, \Delta B} \hspace{0.5em}
            & \langle G_W, \Delta A B^\top + A \Delta B^\top \rangle \\
        \text{s.t. }
            & \| \Delta A B^\top \|_{2 \to 2} \leq \frac{\eta}{2}, \\
            & \| A \Delta B^\top \|_{2 \to 2} \leq \frac{\eta}{2}, \\
            & U^\top (\Delta A B^\top + A \Delta B^\top) V = 0
    \end{aligned}
\end{array}
\end{array}$$
where $U \in \mathbb{R}^{m \times k}$ and $V \in \mathbb{R}^{n \times k}$ are the left- and right- singular vectors of the span past-task directions, $C = \operatorname{span}\left(\{ C_i \}_{1 \leq i \leq k} \right)$.

## 3. Lagrangian formulation

Converting the trust-region problems in the previous section into Lagrangians then allows us to solve them via dual-ascent.

$$\begin{array}{ccc}
\begin{array}{c}
    \text{Muon:} \\
    \begin{aligned}
        &\mathcal{L}_{\text{Muon}}(\Delta W; G_W) \\
            &\qquad= \langle G_W, \Delta W \rangle \\
            &\qquad\quad + \iota_{\mathbb{B}_{\eta}^{m \times n}}(\Delta W)
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon:} \\
    \begin{aligned}
        &\mathcal{L}_{\text{LoRA-Muon}}(\Delta A, \Delta B; G_W) \\
            &\qquad= \langle G_W, \Delta A B^\top + A \Delta B^\top \rangle \\
            &\qquad\quad + \iota_{\mathbb{B}_{\eta/2}^{m \times n}}(\Delta A B^\top) + \iota_{\mathbb{B}_{\eta/2}^{m \times n}}(A \Delta B^\top)
    \end{aligned}
\end{array}
\\[1.5em]
\Big\downarrow\ {\scriptstyle \text{OGD constraint} }
&
&
\Big\downarrow\ {\scriptstyle \text{OGD constraint} }
\\[1.5em]
\begin{array}{c}
    \text{Muon-OGD:} \\
    \begin{aligned}
        &\mathcal{L}_{\text{Muon-OGD}}(\Delta W, \Lambda; G_W) \\
            &\qquad= \langle G_W, \Delta W \rangle \\
              &\qquad\quad + \langle U^\top (\Delta W) V, \Lambda \rangle \\
              &\qquad\quad + \iota_{\mathbb{B}_{\eta}^{m \times n}}(\Delta W)
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon-OGD:} \\
    \begin{aligned}
        &\mathcal{L}_{\text{LoRA-Muon-OGD}}(\Delta A, \Delta B, \Lambda; G_W) \\
            &\qquad= \langle G_W, \Delta A B^\top + A \Delta B^\top \rangle \\
              &\qquad\quad + \langle U^\top (\Delta A B^\top + A \Delta B^\top) V, \Lambda \rangle \\
              &\qquad\quad + \iota_{\mathbb{B}_{\eta/2}^{m \times n}}(\Delta A B^\top) + \iota_{\mathbb{B}_{\eta/2}^{m \times n}}(A \Delta B^\top)
    \end{aligned}
\end{array}
\end{array}$$
where $\iota_S$ is the indicator function of set $S$ defined as,
$$\iota_S(X) = \begin{cases}
    0 & X \in S \\
    +\infty & X \notin S
\end{cases},$$
and $\mathbb{B}_{\rho}^{m \times n} = \{ X \in \mathbb{R}^{m \times n} : \| X \|_{2 \to 2} \leq \rho \}$.

Now, from the cyclic property of the trace, we have the identity,
$$
\begin{equation}
    \langle U^\top X V, \Lambda \rangle
        = \langle U \Lambda V^\top, X \rangle \quad\text{for any} \quad X \in \mathbb{R}^{m \times n}.
\end{equation}
$$
Thus we can rewrite the Lagrangian of the Muon-OGD and LoRA-Muon-OGD optimizers as follows:
$$\begin{array}{ccc}
\begin{array}{c}
    \text{Muon-OGD:} \\
    \begin{aligned}
        &\mathcal{L}_{\text{Muon-OGD}}(\Delta W, \Lambda; G_W) \\
            &\qquad= \langle G_W + U \Lambda V^\top, \Delta W \rangle \\
              &\qquad\quad + \iota_{\mathbb{B}_{\eta}^{m \times n}}(\Delta W)
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon-OGD:} \\
    \begin{aligned}
        &\mathcal{L}_{\text{LoRA-Muon-OGD}}(\Delta A, \Delta B, \Lambda; G_W) \\
            &\qquad= \langle G_W + U \Lambda V^\top, \Delta A B^\top + A \Delta B^\top \rangle \\
              &\qquad\quad + \iota_{\mathbb{B}_{\eta/2}^{m \times n}}(\Delta A B^\top) + \iota_{\mathbb{B}_{\eta/2}^{m \times n}}(A \Delta B^\top)
    \end{aligned}
\end{array}
\end{array}$$

Thus, the OGD-versions of the Muon and LoRA-Muon optimizers are similar to the originals, but with a shifted gradient $G_W \mapsto G_W + U \Lambda V^\top$:
$$\begin{aligned}
    \mathcal{L}_{\text{Muon-OGD}}(\Delta W, {\color{darkblue}{\Lambda; G_W}})
        &= \mathcal{L}_{\text{Muon}}(\Delta W; {\color{darkblue}{G_W + U \Lambda V^\top}}) \\
    \mathcal{L}_{\text{LoRA-Muon-OGD}}(\Delta A, \Delta B, {\color{darkblue}{\Lambda; G_W}})
        &= \mathcal{L}_{\text{LoRA-Muon}}(\Delta A, \Delta B; {\color{darkblue}{G_W + U \Lambda V^\top}})
\end{aligned}$$

## 4. Deriving the update rules

For Muon and LoRA-Muon, their respective trust-region problems in [Section 2](#2-trust-region-problems) are already equivalent to minimizing $\mathcal{L}_{\text{Muon}}$ and $\mathcal{L}_{\text{LoRA-Muon}}$ w.r.t. $\Delta W$ or $(\Delta A, \Delta B)$. Solving these problems then yields their update rules. For Muon-OGD and LoRA-Muon-OGD, one can then check that their respective trust-region problems are equivalent to the saddle point problems we construct by taking their Lagrangians in [Section 3](#3-lagrangian-formulation) and minimizing it w.r.t. the differentials $\Delta W$ or $(\Delta A, \Delta B)$ and maximizing w.r.t. $\Lambda$. From Sion's minimax theorem, we can swap the order of the $\min$ and $\max$ here. That is, we have:
$$
\begin{aligned}
    \min_{\Delta W} \max_{\Lambda} \mathcal{L}_{\text{Muon-OGD}}
        &= \max_{\Lambda} \min_{\Delta W} \mathcal{L}_{\text{Muon-OGD}} \\
    \min_{\Delta A, \Delta B} \max_{\Lambda} \mathcal{L}_{\text{LoRA-Muon-OGD}}
        &= \max_{\Lambda} \min_{\Delta A, \Delta B} \mathcal{L}_{\text{LoRA-Muon-OGD}}
\end{aligned}
$$
Solving these minimization and maximization subproblems and alternating these updates then yields the update rules for Muon-OGD and LoRA-Muon-OGD. I've summarized the results below:

$$\begin{array}{ccc}
\begin{array}{c}
    \text{Muon:} \\
    \Delta W^* = -\eta \cdot \operatorname{msign}(G_W)
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon:} \\
    \begin{aligned}
        \Delta A^* &= -\frac{\eta}{2} \operatorname{msign}(\underbrace{G_A}_{=G_W B} S_B^{-1/2}) S_B^{-1/2} \\
        \Delta B^* &= -\frac{\eta}{2} \operatorname{msign}(\underbrace{G_B}_{=G_W^\top A} S_A^{-1/2}) S_A^{-1/2} 
    \end{aligned}
\end{array}
\\[1.5em]
\Big\downarrow\ {\scriptstyle \text{OGD dual shift} }
&
&
\Big\downarrow\ {\scriptstyle \text{OGD dual shift} }
\\[1.5em]
\begin{array}{c}
    \text{Muon-OGD:} \\
    \begin{aligned}
        \Delta W^{(j)}
            &= -\eta \cdot \operatorname{msign}(G_W {\color{darkblue}{+ U \Lambda^{(j-1)} V^\top}}) \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} U^\top (\Delta W^{(j)}) V
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-Muon-OGD:} \\
    \begin{aligned}
        \Delta A^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign} \left( (G_W{\color{darkblue}{ + U \Lambda^{(j-1)} V^\top}}) B S_B^{-1/2} \right) S_B^{-1/2} \\
        \Delta B^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign} \left( (G_W{\color{darkblue}{ + U \Lambda^{(j-1)} V^\top}})^\top A S_A^{-1/2} \right) S_A^{-1/2} \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} U^\top (\underbrace{\Delta A^{(j)} B^\top + A (\Delta B^{(j)})^\top}_{\Delta W^{(j)}}) V \\[0.5em]
        &\qquad\text{or, equivalently,} \\[0.5em]
        \Delta A^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign} \left( G_A S_B^{-1/2}{\color{darkblue}{ + (U \Lambda^{(j-1)}) (V^\top B) S_B^{-1/2}}} \right) S_B^{-1/2} \\
        \Delta B^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign} \left( G_B S_B^{-1/2}{\color{darkblue}{ + (V (\Lambda^{(j-1)})^\top) (U^\top A) S_A^{-1/2}}} \right) S_A^{-1/2} \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} \left[ (U^\top \Delta A^{(j)}) (V^\top B)^\top + (U^\top A) (V^\top \Delta B^{(j)})^\top \right]
    \end{aligned}
\end{array}
\end{array}$$
where $\operatorname{msign}(X)$ is the matrix sign function which maps non-zero singular values of $X$ to $1$, $S_A = A^\top A$, and $S_B = B^\top B$.

### 4.1. Generalizing to LMO-OGD and LoRA-LMO-OGD

As we discussed in [LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/abs/2606.12921), the LoRA-Muon derivation generalizes to steepest descent under unitary-invariant norms. Thus, if we let $\operatorname{LMO}_{\| \cdot \|}$ be the Linear Minimization Oracle (LMO) of any unitary invariant norm $\| \cdot \|$, we'll get the more general OGD update rules:

$$\begin{array}{ccc}
\begin{array}{c}
    \text{LMO-OGD:} \\
    \begin{aligned}
        \Delta W^{(j)}
            &= \eta \cdot \operatorname{LMO}_{\| \cdot \|}(G_W + U \Lambda^{(j-1)} V^\top) \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} U^\top (\Delta W^{(j)}) V
    \end{aligned}
\end{array}
&
\xrightarrow{\quad \text{LoRA split} \quad}
&
\begin{array}{c}
    \text{LoRA-LMO-OGD:} \\
    \begin{aligned}
        \Delta A^{(j)}
            &= \frac{\eta}{2} \operatorname{LMO}_{\| \cdot \|} \left( G_A S_B^{-1/2} + (U \Lambda^{(j-1)}) (V^\top B) S_B^{-1/2} \right) S_B^{-1/2} \\
        \Delta B^{(j)}
            &= \frac{\eta}{2} \operatorname{LMO}_{\| \cdot \|} \left( G_B S_A^{-1/2} + (V (\Lambda^{(j-1)})^\top) (U^\top A) S_A^{-1/2} \right) S_A^{-1/2} \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} \left[ (U^\top \Delta A^{(j)}) (V^\top B)^\top + (U^\top A) (V^\top \Delta B^{(j)})^\top \right]
    \end{aligned}
\end{array}
\end{array}$$

## 5. Generalizing to Finsler norms, smooth weight parametrizations, and general linear constraints

Let $\mathcal{W} \subseteq \mathbb{R}^{m \times n}$ be a finite dimensional (matrix) manifold equipped with a Finsler norm $\| \cdot \|_W$ that varies smoothly across $W \in \mathcal{W}$. Let its weights be smoothly parametrized by,
$$
\begin{equation}
    \phi: \Theta \to \mathcal{W}, \qquad W = \phi(\theta),
\end{equation}
$$
where $\Theta$ is some finite-dimensional vector space and $\theta \in \Theta$. E.g., the LoRA parametrization: $\Theta \in \mathbb{R}^{m \times r} \times \mathbb{R}^{n \times r}$, $\theta = (A, B)$, and $\phi(\theta) = \phi(A, B) = AB^\top$. For optimization on Stiefel manifolds, it suffices to set $\Theta = \mathcal{W}$ and $\phi = \operatorname{Id}$.

Let $D_{\phi_{\theta}}: T_{\theta} \Theta \to T_{W} \mathcal{W}$ be the differential of $\phi$ at $\theta$ and $D_{\phi_{\theta}}^*: T_{W}^* \mathcal{W} \to T_{\theta}^* \Theta$ be its adjoint such that,
$$\begin{align}
    \Delta W
        &= D_{\phi_{\theta}}[\Delta \theta], \\
    \langle H, D_{\phi_{\theta}}[\Delta \theta] \rangle_{\mathcal{W}}
        &= \langle D_{\phi_{\theta}}^*[H], \Delta \theta \rangle_{\Theta}
        \quad \text{for all} \quad H \in T_{W}^* \mathcal{W}, \Delta \theta \in T_{\theta} \Theta,
\end{align}$$
and the $\langle \cdot, \cdot \rangle_{\mathcal{W}}: T_{W}^* \mathcal{W} \times T_{W} \mathcal{W} \to \mathbb{R}$ and $\langle \cdot, \cdot \rangle_\Theta: T_{\theta}^* \Theta \times T_{\theta} \Theta \to \mathbb{R}$ operators here are the canonical pairing of cotangent and tangent vectors of $\mathcal{W}$ and $\Theta$ respectively. Throughout, we will use the Frobenius product for these pairings, $\langle X, Y \rangle_F = \operatorname{tr}(XY^\top)$. For LoRA we have, $D_{\phi_{(A, B)}}[\Delta A, \Delta B] = \Delta A B^\top + A \Delta B^\top$ and $D_{\phi_{(A, B)}}^*[H] = (HB, H^\top A)$.

Now let $\mathcal{P}_W : T_{W} \mathcal{W} \to \mathcal{Y}$ be a (point-dependent) linear constraint with adjoint $\mathcal{P}_W^*: \mathcal{Y}^* \to T_{W}^* \mathcal{W}$ such that,
$$\begin{equation}
    \langle \mathcal{P}_W(X), \Lambda \rangle_{\mathcal{Y}}
        = \langle X, \mathcal{P}_W^*(\Lambda) \rangle_{\mathcal{W}}
        \quad \text{for all} \quad X \in T_W \mathcal{W}, \Lambda \in \mathcal{Y}^*,
\end{equation}$$
and the $\langle \cdot, \cdot \rangle_{\mathcal{Y}}: \mathcal{Y} \times \mathcal{Y}^* \to \mathbb{R}$ operator here is the canonical pairing of $\mathcal{Y}$ vectors and $\mathcal{Y}^*$ covectors. For OGD earlier, we have, $\mathcal{P}(X) = U^\top X V$. And for Stiefel manifold optimization, we have, $\mathcal{P}_W(X) = W^\top X + X^\top W$ [(Bernstein, 2025)](https://thinkingmachines.ai/blog/modular-manifolds/).

The problem we then want to solve is,
$$\begin{equation}
    \Delta W^*
        = \arg\min_{\Delta W \in T_W \mathcal{W}} \langle G_W, \Delta W \rangle_{\mathcal{W}}
        \quad \text{s.t.}
            \quad \| \Delta W \|_W \leq \eta,
            \quad \mathcal{P}_W(\Delta W) = 0,
\end{equation}$$
or in terms of $\theta$,
$$\begin{equation}
    \Delta \theta^*
        = \arg\min_{\Delta \theta \in T_\theta \Theta} \langle G_W, D_{\phi_{\theta}}[\Delta \theta] \rangle_{\mathcal{W}}
        \quad \text{s.t.}
            \quad \Delta \theta \in \mathcal{K}_{\theta},
            \quad \mathcal{P}_W(D_{\phi_{\theta}}[\Delta \theta]) = 0,
\end{equation}$$
where $G_W \in T_W^* \mathcal{W}$ and $\mathcal{K}_{\theta} \subseteq \{ \Delta \theta \in T_\theta \Theta : \| D_{\phi_{\theta}}[\Delta \theta] \|_W \leq \eta \}$ is some (split) trust region constraint that satisfies the $\| \Delta W \|_W \leq \eta$ constraint. For Muon, set $\mathcal{K}_{\theta} = \mathbb{B}_{\eta}^{m \times n}$. And for LoRA-Muon, we use the split spectral constraint in [Section 2](#2-problem-setting).

Let $\Lambda \in \mathcal{Y}^*$. The Lagrangian then is,
$$\begin{align}
    \mathcal{L}(\Delta \theta, \theta; G_W)
        &= \langle G_W, D_{\phi_{\theta}}[\Delta \theta] \rangle_{\mathcal{W}}
            + \langle \mathcal{P}_W(D_{\phi_{\theta}}[\Delta \theta]), \Lambda \rangle_{\mathcal{Y}}
            + \iota_{\mathcal{K}_{\theta}}(\Delta \theta) \\
        &= \langle G_W, D_{\phi_{\theta}}[\Delta \theta] \rangle_{\mathcal{W}}
            + \langle D_{\phi_{\theta}}[\Delta \theta], \mathcal{P}_W^*(\Lambda) \rangle_{\mathcal{W}}
            + \iota_{\mathcal{K}_{\theta}}(\Delta \theta) \label{eq:lagragian-p-adjoint} \\
        &= \langle G_W + \mathcal{P}_W^*(\Lambda), D_{\phi_{\theta}}[\Delta \theta] \rangle_{\mathcal{W}}
            + \iota_{\mathcal{K}_{\theta}}(\Delta \theta) \\
        &= \langle D_{\phi_{\theta}}^*[G_W + \mathcal{P}_W^*(\Lambda)], \Delta \theta \rangle_{\Theta}
            + \iota_{\mathcal{K}_{\theta}}(\Delta \theta) \label{eq:lagragian-diff-adjoint} \\
        &= \langle D_{\phi_{\theta}}^*[G_W] + D_{\phi_{\theta}}^*[\mathcal{P}_W^*(\Lambda)], \Delta \theta \rangle_{\Theta}
            + \iota_{\mathcal{K}_{\theta}}(\Delta \theta), \label{eq:final-lagragian}
\end{align}$$
minimization of which can be solved factor-wise. Note that we used the adjoint of $P_W$ in Equation $\eqref{eq:lagragian-p-adjoint}$, the adjoint differential in Equation $\eqref{eq:lagragian-diff-adjoint}$, and the lineary of the adjoint differential in Equation $\eqref{eq:final-lagragian}$.

The 'commutation' we discussed rather loosely in [Section 2](#2-problem-setting) then directly follows from Equations $\eqref{eq:lagragian-diff-adjoint}$ and $\eqref{eq:final-lagragian}$. We can either (1) apply the duals shift first, $\xi \mapsto \xi + \mathcal{P}_W^*(\Lambda)$, then the reparametrization, $\mathcal{W} \mapsto \Theta$, or (2) apply the reparametrization first then the pullback dual shift, $\zeta \mapsto \zeta + D_{\phi_\theta}^*[\mathcal{P}_W^*(\Lambda)]$, and end up with the same Lagrangian and thereby the same optimizer.

$$\begin{array}{ccc}
\begin{array}{c}
    \text{LMO:} \\
    \langle G_W, \Delta W \rangle_{\mathcal{W}}
        + \iota_{\mathbb{B}_{\eta}}(\Delta W)
\end{array}
&
\xrightarrow{\quad \text{reparametrization} \quad}
&
\begin{array}{c}
    \text{LMO + Smooth Parametrization:} \\
    \langle D_{\phi_{\theta}}^*[G_W], \Delta \theta \rangle_{\Theta}
        + \iota_{\mathcal{K}_{\theta}}(\Delta \theta)
\end{array}
\\[1.5em]
\Big\downarrow\ {\scriptstyle \text{dual shift} }
&
&
\Big\downarrow\ {\scriptstyle \text{pullback dual shift} }
\\[1.5em]
\begin{array}{c}
    \text{LMO + Linear Constraint:} \\
    \langle G_W + \mathcal{P}_W^*(\Lambda), \Delta W \rangle_{\mathcal{W}}
        + \iota_{\mathbb{B}_{\eta}}(\Delta W)
\end{array}
&
\xrightarrow{\quad \text{reparametrization} \quad}
&
\begin{array}{c}
    \text{LMO + Linear Constraint + Smooth Parametrization:} \\
    \langle D_{\phi_{\theta}}^*[G_W] + D_{\phi_{\theta}}^*[\mathcal{P}_W^*(\Lambda)], \Delta \theta \rangle_{\Theta}
        + \iota_{\mathcal{K}_{\theta}}(\Delta \theta)
\end{array}
\end{array}$$

## 6. Optimizer zoo

|                                                                                   | Norm                              | Parametrization |            Linear Constraint            |
| --------------------------------------------------------------------------------- | --------------------------------- | :-------------: | :-------------------------------------: |
| SGD                                                                               | $\| \cdot \|_{\text{vec},2}$      |        -        |                    -                    |
| [SignSGD](https://arxiv.org/abs/1802.04434)                                       | $\| \cdot \|_{\text{vec},\infty}$ |        -        |                    -                    |
| [Schatten-$p$ GD](https://leloykun.github.io/ponder/steepest-descent-schatten-p/) | $\| \cdot \|_{S_p}$               |        -        |                    -                    |
| [Muon](https://kellerjordan.github.io/posts/muon/)                                | $\| \cdot \|_{2 \to 2}$           |        -        |                    -                    |
| [Stiefel-Muon](https://thinkingmachines.ai/blog/modular-manifolds/)               | $\| \cdot \|_{2 \to 2}$           |        -        | $W^\top \Delta W + \Delta W^\top W = 0$ |
| [Spectral-Sphere-Optimizer](https://arxiv.org/abs/2601.08393)                     | $\| \cdot \|_{2 \to 2}$           |        -        |      $ u_1^\top \Delta W v_1 = 0$       |
| [Muon-OGD](https://arxiv.org/abs/2605.08949)                                      | $\| \cdot \|_{2 \to 2}$           |        -        |         $U^\top \Delta W V = 0$         |
| [LoRA-Muon](https://arxiv.org/abs/2606.12921)                                     | $\| \cdot \|_{2 \to 2}$           | $W = A B^\top$  |                    _                    |
| LoRA-Muon-OGD                                                                     | $\| \cdot \|_{2 \to 2}$           | $W = A B^\top$  |         $U^\top \Delta W V = 0$         |

where $(u_1, v_1)$ are the principal left and right singular vectors of $W$.

## How to Cite

```bibtex
@article{cesista2026loramuonogd,
  title = {{LoRA-Muon-OGD}: Spectral Orthogonal Gradient Projection on the Low-Rank Manifold for LLM Continual Learning},
  author = {Franz Louis Cesista},
  journal = {leloykun.github.io},
  year = {2026},
  month = {June},
  day = {25},
  url = {https://leloykun.github.io/ponder/lora-muon-ogd/},
}
```

> If you find this post useful, please consider supporting my work by sponsoring me on GitHub: [![Sponsor on GitHub][sponsor-badge]][sponsor-link]

[sponsor-badge]: https://img.shields.io/badge/🤝-Sponsor%20me-1da1f2?logo=github&style=flat-square
[sponsor-link]: https://github.com/sponsors/leloykun

## References

1. James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.
2. Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. arXiv preprint arXiv:1312.6211, 2013.
3. Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. In Silvia Chiappa and Roberto Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 3762–3773. PMLR, 26–28 Aug 2020. URL https://proceedings.mlr.press/v108/farajtabar20a.html.
4. Binghang Lu, Zheyuan Deng, Runyu Zhang, Bing Hu, Yunhan Zhao, Yuan Tian, Changhong Mou, Guang Lin, Xiaomin Li (2026). Muon-OGD: Muon-based Spectral Orthogonal Gradient Projection for LLM Continual Learning. URL https://arxiv.org/abs/2605.08949
5. Franz Louis Cesista (2025). Rethinking Maximal Update Parametrization: Steepest Descent on Finsler-Structured (Matrix) Geometries via Dual Ascent. URL https://leloykun.github.io/ponder/steepest-descent-finsler-dual-ascent/
6. Franz Louis Cesista, Katherine Crowson, Cédric Simal, Stella Biderman (2026). LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold. URL https://arxiv.org/abs/2606.12921
7. Franz Louis Cesista (2025). Steepest Descent Under Schatten-p Norms. URL https://leloykun.github.io/ponder/steepest-descent-schatten-p/
8. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/.
9. Jeremy Bernstein, "Modular Manifolds", Thinking Machines Lab: Connectionism, Sep 2025.
10. Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Anima Anandkumar (2018). signSGD: Compressed Optimisation for Non-Convex Problems. URL https://arxiv.org/abs/1802.04434
11. Tian Xie, Haoming Luo, Haoyu Tang, Yiwen Hu, Jason Klein Liu, Qingnan Ren, Yang Wang, Wayne Xin Zhao, Rui Yan, Bing Su, Chong Luo, Baining Guo (2026). Controlled LLM Training on Spectral Sphere. URL https://arxiv.org/abs/2601.08393
