---
title: "LoRA-Muon-OGD: Spectral Orthogonal Gradient Projection on the Low-Rank Manifold for LLM Continual Learning"
date: 2026-06-25
tags: ["Machine Learning", "Optimizers", "LoRA", "Low-Rank Adaptation", "Muon", "Continual Learning"]
author: ["Franz Louis Cesista"]
description: "Generalizing Orthogonal Gradient Projection to the low-rank case and to steepest descent under a larger family of norms."
summary: "Generalizing Orthogonal Gradient Projection to the low-rank case and to steepest descent under a larger family of norms."
---


## 1. Introduction

The more LLMs are deployed in more diverse, longer-horizon tasks, the more they need to continually learn and acquire skills 'on-the-go', ideally without forgetting past skills. But that is precisely the central challenge in the field at the moment: when finetuned on new tasks, LLMs rapidly "forget" previously learned skills. Imagine a person immediately forgetting how to ride a bike as soon as they learn how to catch fish by the river. This phenomenon is called, "catastrophic forgetting" ([Goodfellow et al., 2013](https://arxiv.org/abs/1312.6211), [Kirkpatrick et al., 2017](https://www.pnas.org/doi/10.1073/pnas.1611835114)).

One way to mitigate the catastrophic forgetting issue is to project gradients away from past-task directions $\{ C_i \}_{1 \leq i \leq K}$ via Orthogonal Gradient Descent (OGD) ([Farajtabar et al., 2020](https://proceedings.mlr.press/v108/farajtabar20a.html)). That is, we want our weight updates $\Delta W$ to satisfy,
$$
\begin{equation}
    \langle \Delta W, C_i \rangle = 0 \qquad \text{for all } 1 \leq i \leq K. \label{eq:non-interference}
\end{equation}
$$
I would argue this is somewhat hacky as we may want the LLM to *refine* previously learned skills as it chugs through problems, but it works and simple-enough to be bitter-lesson-pilled. [Lu et al., 2026](https://arxiv.org/abs/2605.08949) recently derived Muon-OGD which takes the maximal updates under the spectral-norm geometry in $\mathbb{R}^{m \times n}$ while satisfying the non-interference constraint in $\eqref{eq:non-interference}$. They report SOTA results on continual learning tasks, but the algorithm requires materializing the dense gradient matrix $G_W$ which makes it unsuitable for low-rank finetuning.

In this work, we derive LoRA-Muon-OGD which takes the maximal updates under the spectral-norm, but on the low-rank manifold $\mathcal{M}_r = \{ W = A B^\top | A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{n \times r}, \operatorname{rank}(A) = \operatorname{rank}(B) = r \}$ while still satisfying the non-interference constraint in $\eqref{eq:non-interference}$. We also show that the derivation is natural and generalizes to steepest descent under arbitrary unitary-invariant norm.

## 2. Problem setting

Let $f: \mathcal{W} \mapsto \mathbb{R}$ be a differentiable and bounded below objective function defined on a finite-dimensional manifold $\mathcal{W}$ equipped with a norm $\| \cdot \|$. Let $G_W := \nabla_W f(W)$ be its differential at $W \in \mathcal{W}$. In the LoRA setting where $\mathcal{W} = \mathcal{M}_r$, let $G_A := G_W B$ and $G_B := G_W^\top A$ be the differentials w.r.t. the $A$ and $B$ LoRA factors, respectively. In practice, when doing LoRA finetuning, backpropagation only gives us access to $G_A$ and $G_B$, not $G_W$, and constructing the full dense 'gradient' matrix is often expensive in terms of compute and memory.

Our derivations here are made a lot simpler by the observation that the low-rank constraint and OGD's non-interferance constraint, intuitively speaking, commute as optimizer-producing actions.
Starting from the Muon optimizer ([Keller et al., 2024](https://kellerjordan.github.io/posts/muon/)) and applying the low-rank constraint first yields LoRA-Muon ([Cesista et al., 2026](https://arxiv.org/abs/2606.12921))
If we instead apply the non-interference constraint first, we instead get Muon-OGD as in [Lu et al., 2026](https://arxiv.org/abs/2605.08949).
But, either way, applying the other constraint then yields LoRA-Muon-OGD.

The following are trust-region problems solving which yield the four optimizers we discuss here.

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
where $U$ and $V$ are the left- and right- singular basis vectors of the past-task directions, $C_i = C_{\alpha \beta} = U_{\alpha} V_{\beta}^\top$.

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
$$\langle U^\top X V, \Lambda \rangle = \langle U \Lambda V^\top, X \rangle \quad\text{for any} \quad X \in \mathbb{R}^{m \times n}.$$
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

For Muon and LoRA-Muon, their respective trust-region problems in [Section 2](#2-trust-region-problems) is already equivalent to minimizing $\mathcal{L}_{\text{Muon}}$ and $\mathcal{L}_{\text{LoRA-Muon}}$ w.r.t. $\Delta W$ or $(\Delta A, \Delta B)$. Solving these problems then yields their update rules. For Muon-OGD and LoRA-Muon-OGD, one can then check that their respective trust-region problems are equivalent to the saddle point problems we construct by taking their Lagrangian in [Section 3](#3-lagrangian-formulation) and minimizing it w.r.t. the differentials $\Delta W$ or $(\Delta A, \Delta B)$ and maximizing w.r.t. $\Lambda$. From Sion's minimax theorem, we can swap the order of the $\min$ and $\max$ here. That is, we have:
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
        \Delta A^* &= -\frac{\eta}{2} \operatorname{msign}(\underbrace{G_W B}_{G_A} S_B^{-1/2}) S_B^{-1/2} \\
        \Delta B^* &= -\frac{\eta}{2} \operatorname{msign}(\underbrace{G_W^\top A}_{G_B} S_A^{-1/2}) S_A^{-1/2} 
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
            &= -\eta \cdot \operatorname{msign}(G_W + U \Lambda^{(j-1)} V^\top) \\
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
            &= -\frac{\eta}{2} \operatorname{msign}((G_W + U \Lambda^{(j-1)} V^\top) B S_B^{-1/2}) S_B^{-1/2} \\
        \Delta B^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign}((G_W + U \Lambda^{(j-1)} V^\top)^\top A S_A^{-1/2}) S_A^{-1/2} \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} U^\top (\underbrace{\Delta A^{(j)} B^\top + A (\Delta B^{(j)})^\top}_{\Delta W^{(j)}}) V \\[0.5em]
        &\qquad\text{or, equivalently,} \\[0.5em]
        \Delta A^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign}((G_A + U \Lambda^{(j-1)} (V^\top B)) S_B^{-1/2}) S_B^{-1/2} \\
        \Delta B^{(j)}
            &= -\frac{\eta}{2} \operatorname{msign}((G_B + V (\Lambda^{(j-1)})^\top (U^\top A)) S_A^{-1/2}) S_A^{-1/2} \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} [(U^\top \Delta A^{(j)}) (V^\top B)^\top + (U^\top A) (V^\top \Delta B^{(j)})^\top]
    \end{aligned}
\end{array}
\end{array}$$
where $\operatorname{msign}(X) = X (X^\top X)^{-1/2}$ is the matrix sign function which maps non-zero singular values of a matrix $X$ to $1$, $S_A = A^\top A$, and $S_B = B^\top B$.

### 4.1. Generalizing to LMO-OGD and LoRA-LMO-OGD

As we discussed in [LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/abs/2606.12921), the maths behind LoRA-Muon generalizes to steepest descent under unitary-invariant norms. Thus, if we let $\operatorname{LMO}_{\| \cdot \|}$ be the Linear Minimization Oracle (LMO) of an arbitrary unitary-invariant norm $\| \cdot \|$, we'll get the more general OGD update rules:

$$\begin{array}{ccc}
\begin{array}{c}
    \text{LMO-OGD:} \\
    \begin{aligned}
        \Delta W^{(j)}
            &= \eta \cdot \operatorname{LMO}(G_W + U \Lambda^{(j-1)} V^\top) \\
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
            &= \frac{\eta}{2} \operatorname{LMO}((G_A + U \Lambda^{(j-1)} (V^\top B)) S_B^{-1/2}) S_B^{-1/2} \\
        \Delta B^{(j)}
            &= \frac{\eta}{2} \operatorname{LMO}((G_B + V (\Lambda^{(j-1)})^\top (U^\top A)) S_A^{-1/2}) S_A^{-1/2} \\
        \Delta \Lambda^{(j)}
            &= \sigma_{\Lambda} [(U^\top \Delta A^{(j)}) (V^\top B)^\top + (U^\top A) (V^\top \Delta B^{(j)})^\top]
    \end{aligned}
\end{array}
\end{array}$$

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
5. Franz Louis Cesista, Katherine Crowson, Cédric Simal, Stella Biderman (2026). LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold. URL https://arxiv.org/abs/2606.12921
6. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/.
