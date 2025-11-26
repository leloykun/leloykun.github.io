---
title: "Critical Batch Size for Steepest Descent Under Arbitrary Norms"
date: 2025-11-22
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "First-order optimization under arbitrary norms with Nesterov momentum (and weight decay) yields the same critical batch size."
summary: "First-order optimization under arbitrary norms with Nesterov momentum (and weight decay) yields the same critical batch size."
# cover:
#     image: lr_transfer_pdhg_stiefel_spectral.png
#     alt: "Cover"
#     relative: true
# editPost:
#     URL: "https://x.com/leloykun/status/1958915061793075549"
#     Text: "Crossposted from X (formerly Twitter)"
citation:
    title: "Critical Batch Size for Steepest Descent Under Arbitrary Norms"
    author:
        - "Franz Louis Cesista"
    publication_date: "2025/11/22"
---

## Preliminaries

$$ G_{\xi_t}(W_t) := \nabla f(W_t) - \xi_t$$
$$ \nabla f_{S_t}(W_t) := \frac{1}{b}\sum_{i=1}^{b} G_{\xi_{t,i}}(W_t) $$
$$ \xi_{S_t} := \nabla f(W_t) - \nabla f_{S_t}(W_t) $$

### Assumptions

> **Assumption 1 (Unbiased gradient noise)**
$$\begin{equation} \mathbb{E}\left[\xi_t\right] = 0 \end{equation}$$

> **Assumption 2 (Bounded gradient noise variance)** There exists $\widetilde{\sigma} > 0$ such that for all $t$ and arbitrary norm $\| \cdot \|$,
$$\begin{equation} \mathbb{E}\left[\| \xi_t \|^{\dagger 2}\right] = \widetilde{\sigma}^2 \end{equation}$$
Equivalently, from the universality of norms on finite-dimensional vector spaces, there exists $\kappa > 0$ such that,
$$\begin{equation} \mathbb{E}\left[\| \xi_t \|_F^{2}\right] \leq \kappa^2 \widetilde{\sigma}^2 =: \sigma^2 \end{equation}$$
where $\sigma := \kappa \widetilde{\sigma}$.

> **Assumption 3 (L-smoothness of $f$ under the pair $(\| \cdot \|, \| \cdot \|^{\dagger})$)** There exists $\widetilde{L} > 0$ such that for all $X, Y \in \mathcal{W}$,
$$\begin{equation} \| \nabla f(Y) - \nabla f(X) \|^{\dagger} \leq \widetilde{L} \| Y - X \| \end{equation}$$
Equivalently, from the universality of norms on finite-dimensional vector spaces, there exists $\kappa > 0$ such that,
$$\begin{equation} \| \nabla f(Y) - \nabla f(X) \|_F \leq \kappa \widetilde{L} \| Y - X \|_F = L \| Y - X \|_F \end{equation}$$
where $L := \kappa \widetilde{L}$.

### Nesterov momentum

For a given momentum hyperparameter $\beta \in (0, 1)$, Nesterov momentum is defined as,
$$\begin{align}
    M_t &= \beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t) \nonumber \\
    C_t &= \beta M_t + (1 - \beta) \nabla f_{S_t}(W_t) \nonumber \\
\end{align}$$
We then use $C_t$ to compute the steepest descent update direction.

### Linear Minimization Oracles (LMOs) and dual norms

$$\begin{align}
    A_t^*
        &:= \arg\min_{A \in \mathbb{R}^{m \times n}} \langle C_t, A \rangle_F \quad \text{ s.t. } \quad \| A \| \leq 1 \nonumber \\
        &= \texttt{LMO}_{\| \cdot \|}(C_t) \nonumber
\end{align}$$
such that,
$$\begin{align}
    \| A_t^* \|
        &= 1 \label{eq:lmo-norm} \\
    \langle C_t, A_t^* \rangle_F
        &= \langle C_t, \texttt{LMO}_{\| \cdot \|}(C_t) \rangle_F \nonumber \\
        &= \arg\min_{A \leq 1} \langle C_t, A \rangle_F \nonumber \\
        &= -\arg\max_{A \leq 1} \langle C_t, A \rangle_F \nonumber \\
        &= - \| C_t \|^{\dagger} \label{eq:lmo-inner-product}
\end{align}$$

The update rule for steepest descent with step size $\eta > 0$ and weight decay term $\lambda \geq 0$ is then given by,
$$\begin{equation}
    W_{t+1} = (1 - \lambda\eta) W_t + \eta A_t^* \label{eq:updateweightdecay}
\end{equation}$$

## Convergence bound for steepest descent under arbitrary norms without weight decay

### Gradient noise and momentum error bounds

> **Proposition 4 (Bounded minibatch gradient noise variance)**
$$\begin{equation}
    \mathbb{E}\left[ \| \xi_{S_t} \|_F^{2} \right] \leq \frac{\sigma^2}{b}
\end{equation}$$

**Proof.**
$$\begin{align}
    \mathbb{E}\left[ \| \xi_{S_t} \|_F^{2} \right]
        &= \mathbb{E}\left[ \left\| \nabla f(W_t) - \frac{1}{b} \sum_{i=1}^{b} G_{\xi_{t,i}}(W_t) \right\|_F^{2} \right] \nonumber \\
        &= \mathbb{E}\left[ \left\| \frac{1}{b} \sum_{i=1}^{b} (\nabla f(W_t) - G_{\xi_{t,i}}(W_t)) \right\|_F^{2} \right] \nonumber \\
        &= \mathbb{E}\left[ \left\| \frac{1}{b} \sum_{i=1}^{b} \xi_{t,i} \right\|_F^{2} \right] \nonumber \\
        &\leq \frac{1}{b^2} \sum_{i=1}^{b} \mathbb{E}\left[ \| \xi_{t,i} \|_F^{2} \right] \nonumber \\
        &= \frac{\sigma^2}{b} \quad\blacksquare \nonumber
\end{align}$$

---

We then bound the average first and second moments of the momentum error term, $E_t := \nabla f(W_t) - M_t$, and the Nesterov momentum error term, $\nabla f(W_t) - C_t$.

> **Proposition 5 (Average first and second moments of the momentum error term)**
$$\begin{align}
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[ \| E_t \|_F \right]
        &\leq \frac{2\sqrt{2}}{1 - \beta}\frac{1}{T} \| \nabla f(W_0) - M_0 \|_F
            + \frac{2}{1 - \beta} L \eta
            + \sqrt{2 (1 - \beta)} \frac{\sigma}{\sqrt{b}} \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|^{2}_F\right]
        &\leq \frac{2}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) - M_0 \|^{2}_F
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2
            + 2 (1 - \beta) \frac{\sigma^2}{b}
\end{align}$$

**Proof.** Notice that,
$$\begin{align}
    E_t
        &= \nabla f(W_t) - M_t \nonumber \\
        &= \nabla f(W_t) - (\beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta (\nabla f(W_t) - M_{t-1}) + (1 - \beta)\xi_{S_t} \nonumber \\
        &= \beta (\nabla f(W_t) - \nabla f(W_{t-1}) + \nabla f(W_{t-1}) - M_{t-1}) + (1 - \beta)\xi_{S_t} \nonumber \\
        &= \beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1})) + (1 - \beta) \xi_{S_t} \nonumber
\end{align}$$

Taking norms, expectations, and using Peter-Paul inequality, Assumption (1), Assumption (3), and Proposition (4) then yields,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|_F^2 \right]
        &= \mathbb{E}\left[ \| \beta E_{t-1} + \beta (\nabla f(W_t) - \nabla f(W_{t-1})) + (1 - \beta) \xi_{S_t} \|_F^2 \right] \nonumber \\
        &= \beta^2 \mathbb{E}\left[ \| E_{t-1} \|_F^2 \right]
            + \beta^2 \mathbb{E}\left[ \| \nabla f(W_t) - \nabla f(W_{t-1}) \|_F^2 \right]
            + (1 - \beta)^2 \mathbb{E}\left[ \| \xi_{S_t} \|_F^2 \right] \nonumber \\
            &\quad+ 2 \beta^2 \mathbb{E}\left[ \langle E_{t-1}, \nabla f(W_t) - \nabla f(W_{t-1}) \rangle_F \right] \nonumber \\
            &\quad+ \cancel{2 \beta (1 - \beta) \mathbb{E}\left[ \langle E_{t-1}, \xi_{S_t} \rangle_F \right]
                + 2 \beta (1 - \beta) \mathbb{E}\left[ \langle \nabla f(W_t) - \nabla f(W_{t-1}), \xi_{S_t} \rangle_F \right]} \nonumber \\
        &= \beta^2 (1 + \epsilon) \mathbb{E}\left[ \| E_{t-1} \|_F^2 \right]
            + \beta^2 \left(1 + \frac{1}{\epsilon}\right) L^2 \eta^2
            + (1 - \beta)^2 \frac{\sigma^2}{b} \nonumber
\end{align}$$
for some $\epsilon > 0$. Setting $\epsilon = \frac{1 - \beta}{2}$ and unrolling the recurrence then gives,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|^{2}_F \right]
        &\leq \frac{\beta^2(3 - \beta)}{2} \mathbb{E}\left[ \| E_{t-1} \|^{2}_F \right]
            + \frac{\beta^2 (3 - \beta)}{1 - \beta} L^2 \eta^2
            + (1 - \beta)^2 \frac{\sigma^2}{b} \nonumber \\
        &\leq \frac{1 + \beta}{2} \mathbb{E}\left[ \| E_{t-1} \|^{2}_F \right]
            + \frac{2}{1 - \beta} L^2 \eta^2 
            + (1 - \beta)^2 \frac{\sigma^2}{b} \label{eq:approxub} \\
        &\leq \left( \frac{1 + \beta}{2} \right)^t \mathbb{E}\left[ \| E_{0} \|^{2}_F \right]
            + \left(\frac{2}{1 - \beta} L^2 \eta^2 
            + (1 - \beta)^2 \frac{\sigma^2}{b}\right) \sum_{k=0}^{t-1} \left(\frac{1 + \beta}{2}\right)^k \nonumber \\
        &\leq \left( \frac{1 + \beta}{2} \right)^t \| E_{0} \|^{2}_F
            + \left(\frac{2}{1 - \beta} L^2 \eta^2 
            + (1 - \beta)^2 \frac{\sigma^2}{b}\right) \frac{2}{1 - \beta} \nonumber \\
        &\leq \left( \frac{1 + \beta}{2} \right)^t \| E_{0} \|^{2}_F
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2 
            + 2(1 - \beta)\frac{\sigma^2}{b} \nonumber \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|^{2}_F\right]
        &\leq \frac{2}{1 - \beta} \frac{1}{T} \| E_{0} \|^{2}_F
            + \frac{4}{(1 - \beta)^2} L^2 \eta^2 
            + 2 (1 - \beta) \frac{\sigma^2}{b} \nonumber
\end{align}$$
where we use the crude upper bound $\beta^2 (3 - \beta) \leq 1 + \beta < 2$ in Equation $\eqref{eq:approxub}$ to simplify the algebra.

Applying Jensen's inequality and the fact that $\sqrt{a + b + c} \leq \sqrt{a} + \sqrt{b} + \sqrt{c}$ for $a, b, c > 0$ then yields,
$$\begin{align}
    \mathbb{E}\left[ \| E_t \|_F \right]
        &\leq \sqrt{\mathbb{E}\left[ \| E_t \|^{2}_F \right]} \nonumber \\
        &\leq \left( \sqrt{\frac{1 + \beta}{2}} \right)^t \| E_{0} \|_F
            + \frac{2}{1 - \beta} L \eta 
            + \sqrt{2 (1 - \beta)} \frac{\sigma}{\sqrt{b}} \nonumber \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| E_t \|_F\right]
        &\leq \frac{2\sqrt{2}}{1 - \beta} \frac{1}{T} \| E_{0} \|_F
            + \frac{2}{1 - \beta} L \eta 
            + \sqrt{2 (1 - \beta)} \frac{\sigma}{\sqrt{b}} \quad\blacksquare \nonumber \\
\end{align}$$

---

As for the Nesterov momentum error term, we have,

> **Corollary 6 (Average first and second moments of the Nesterov momentum error term)**
$$\begin{align}
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| \nabla f(W_t) - C_t \|_F \right]
        &\leq \frac{2\sqrt{2}\beta}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) - M_0 \|_F
            + \frac{2 \beta}{1 - \beta} L \eta \nonumber \\
        &\quad+ \left(\sqrt{2 (1 - \beta)}\beta + (1 - \beta)\right) \frac{\sigma}{\sqrt{b}} \\
    \frac{1}{T} \sum_{t = 0}^{T-1} \mathbb{E}\left[\| \nabla f(W_t) - C_t \|_F^2\right]
        &\leq \frac{2\beta}{1 - \beta} \frac{1}{T} \| \nabla f(W_0) - M_0 \|_F^2
            + \frac{4\beta}{(1 - \beta)^2} L^2 \eta^2 \nonumber \\
        &\quad+ (2\beta + 1) (1 - \beta) \frac{\sigma^2}{b}
\end{align}$$

**Proof.** Observe that,
$$\begin{align}
    \nabla f(W_t) - C_t
        &= \nabla f(W_t) - (\beta M_t + (1 - \beta) \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta (\nabla f(W_t) - M_t) + (1 - \beta) (\nabla f(W_t) - \nabla f_{S_t}(W_t)) \nonumber \\
        &= \beta E_t + (1 - \beta) \xi_{S_t} \nonumber
\end{align}$$
Thus, since $x \mapsto \| x \|$ and $x \mapsto \| x \|^2$ are convex, we have,
$$\begin{align}
    \| \nabla f(W_t) - C_t \|_F^{2}
        &\leq \beta \| E_t \|_F^{2} + (1 - \beta) \| \xi_{S_t} \|_F^{2} \nonumber \\
    \| \nabla f(W_t) - C_t \|_F
        &\leq \beta \| E_t \|_F + (1 - \beta) \| \xi_{S_t} \|_F \nonumber
\end{align}$$
The result then follows from Proposition (4) and Proposition (5). $\quad\blacksquare$

---

### Convergence bounds without weight decay

> **Theorem 7 (Convergence bound without weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda = 0$ (i.e., weight decay is disabled) and step size $\eta > 0$. Then for an arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, there exist constants $X, Y, Z > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F] \leq \frac{X}{T} + \frac{Y}{b} + Z
\end{equation}$$
where $T$ is the total number of time steps, $b$ is the batch size, and
$$Y = \frac{(2 \beta + 1)(1 - \beta)}{2} \sigma^2.$$

**Proof.** From the universality of norms in finite-dimensional vector spaces, there exist constants $\kappa_1 > 0, \kappa_2 > 0$ such that for all $X \in \mathbb{R}^{m \times n}$,
$$ \kappa_1 \| X \|_F \leq \| X \|^{\dagger} \leq \kappa_2 \| X \|_F $$
For Muon, we have $\| X \|^{\dagger} = \| X \|_{\text{nuc}}$ (the nuclear norm), and so $\kappa_1 = 1, \kappa_2 = \sqrt{\text{rank}(X)} \leq \sqrt{\min{(m, n)}}$.

Let us first disable weight decay, i.e., set $\lambda = 0$. Since $f$ is $L$-smooth, the descent lemma, Equation $\eqref{eq:lmo-inner-product}$, and Equation $\eqref{eq:lmo-norm}$ yields,
$$\begin{align}
    f(W_{t+1})
        &\leq f(W_t) + \langle \nabla f(W_t), W_{t+1} - W_t \rangle + \frac{L}{2} \| W_{t+1} - W_t \|^2 \label{eq:descentlemma} \\
        &\leq f(W_t) + \langle \nabla f(W_t), \eta A_t^* \rangle + \frac{L}{2} \| \eta A_t^* \|^2 \nonumber \\
        &\leq f(W_t) + \langle \nabla f(W_t) - C_t + C_t, \eta A_t^* \rangle_F + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t) + \langle C_t, \eta A_t^* \rangle_F + \langle \nabla f(W_t) - C_t, \eta A_t^* \rangle_F + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t) - \eta \| C_t \|^{\dagger} + \left(\frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2} + \frac {\eta^2}{2 \epsilon} \| A_t^* \|^2\right) + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t) - \eta \left(\| \nabla f(W_t) \|^{\dagger} - \| \nabla f(W_t) - C_t \|^{\dagger}\right) \nonumber \\
            &\qquad+ \frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac{(1/\epsilon + L)\eta^2}{2} \nonumber \\
        &\leq f(W_t) - \eta \| \nabla f(W_t) \|^{\dagger} + \eta \| \nabla f(W_t) - C_t \|^{\dagger} \nonumber \\
            &\qquad+ \frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac{(1/\epsilon + L)\eta^2}{2} \label{eq:descentlemma-final}
\end{align}$$
Note that the $\langle \cdot, \cdot \rangle$ operator in Equation $\eqref{eq:descentlemma}$ is *not* an inner product, but the canonical pairing between cotangent and tangent spaces ($\nabla f(W_t) \in T_{W_t}^* \mathcal{W}$ while $A_t^* \in T_{W_t}\mathcal{W}$). Under the standard basis of $\mathbb{R}^{m \times n}$, however, it *behaves like* the Frobenius inner product.

Thus, rearranging Equation $\eqref{eq:descentlemma-final}$ and setting $\epsilon = \frac{\kappa_1 \eta}{\kappa_2^2}$ yields,
$$\begin{align}
    \| \nabla f(W_t) \|^{\dagger}
        &\leq \frac{f(W_t) - f(W_{t+1})}{\eta} + \| \nabla f(W_t) - C_t \|^{\dagger} + \frac{\epsilon}{2\eta} \| \nabla f(W_t) - C_t \|^{\dagger 2} + \frac{(1/\epsilon + L)\eta^2}{2} \nonumber \\
    \| \nabla f(W_t) \|_F
        &\leq \frac{f(W_t) - f(W_{t+1})}{\eta\kappa_1} + \frac{\kappa_2}{\kappa_1}\| \nabla f(W_t) - C_t \|_F + \frac{1}{2} \| \nabla f(W_t) - C_t \|^{2}_F + \frac{(\kappa_2^2/\kappa_1 + L\eta)\eta}{2\kappa_1} \nonumber \\
\end{align}$$

And after taking expectations, and averaging, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F]
        &\leq \frac{f(W_0) - f(W_T)}{\eta \kappa_1 T}  + \frac{(\kappa_2^2/\kappa_1 + L\eta)\eta}{2\kappa_1} \nonumber \\
        &\quad+ \frac{1}{T}\frac{\kappa_2}{\kappa_1}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|_F]
            + \frac{1}{2T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) - C_t \|^{2}_F] \nonumber \\
        &\leq \frac{f(W_0) - f(W_T)}{\eta \kappa_1 T}  + \frac{(\kappa_2^2/\kappa_1 + L\eta)\eta}{2\kappa_1} \nonumber \\
        &\quad+ \frac{\kappa_2}{\kappa_1} \left(\frac{2\sqrt{2}\beta}{1 - \beta}\frac{1}{T} \| E_0 \|_F
            + \frac{2 \beta}{1 - \beta} L \eta
            + \left(\sqrt{2(1 - \beta)}\beta + (1 - \beta) \right) \frac{\sigma}{\sqrt{b}} \right) \nonumber \\
        &\quad+ \frac{1}{2} \left(\frac{2\beta}{1 - \beta}\frac{1}{T} \| E_{0} \|^{2}_F
            + \frac{4 \beta}{(1 - \beta)^2} L^2 \eta^2
            + (2 \beta + 1) (1 - \beta) \frac{\sigma^2}{b} \right) \nonumber \\
        &\leq \frac{X}{T} + \frac{Y}{b} + Z
\end{align}$$
where,
$$\begin{align}
    X
        &:= \frac{f(W_0) - f^*}{\eta\kappa_1}
            + \frac{2\sqrt{2}\beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} \| \nabla f(W_0) - M_0 \|_F
            + \frac{\beta}{1 - \beta} \| \nabla f(W_0) - M_0 \|_F^{2} \nonumber \\
    Y
        &:= \frac{(2 \beta + 1)(1 - \beta)}{2} \sigma^2 \nonumber \\
    Z
        &:= \frac{(\kappa_2^2/\kappa_1 + L\eta)\eta}{2\kappa_1}
            + \frac{2 \beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} L \eta
            + \frac{2\beta}{(1 - \beta)^2} L^2 \eta^2 \nonumber \\
            &\qquad+ \left(\sqrt{2 (1 - \beta)} + (1 - \beta)\right) \frac{\kappa_2}{\kappa_1} \frac{\sigma}{\sqrt{b}} \quad\blacksquare \nonumber
\end{align}$$

## Convergence bound for steepest descent under arbitrary norms with weight decay

### Weight, gradient, and momentum size bounds

> **Proposition 8 (Weight size upper bound).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda > 0$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$ and $\| W_0 \| \leq \frac{1}{\lambda}$. Then, for all $t \geq 0$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{equation}
    \| W_t \| \leq \frac{1}{\lambda}
\end{equation}$$

**Proof.** Let us unroll the recurrence above,
$$\begin{align}
    W_t
        &= (1 - \lambda\eta) W_{t-1} + \eta A_{t-1}^* \nonumber \\
        &= (1 - \lambda\eta)^2 W_{t-2} + \eta (1 - \lambda\eta) A_{t-2}^* + \eta A_{t-1}^* \nonumber \\
        &\;\vdots \nonumber \\
        &= (1 - \lambda\eta)^t W_0 + \eta \sum_{i=0}^{t-1} (1 - \lambda\eta)^i A_{t-1-i}^* \nonumber
\end{align}$$
Taking norms and using the triangle inequality then gives,
$$\begin{align}
    \| W_t \|
        &\leq (1 - \lambda\eta)^t \| W_0 \| + \eta \sum_{i=0}^{t-1} (1 - \lambda\eta)^i \| A_{t-1-i}^* \| \nonumber \\
        &\leq (1 - \lambda\eta)^t \| W_0 \| + \eta \sum_{i=0}^{t-1} (1 - \lambda\eta)^i \nonumber \\
        &\leq (1 - \lambda\eta)^t \| W_0 \| + \frac{\eta}{\lambda\eta} (1 - (1 - \lambda\eta)^t) \nonumber \\
        &\leq \frac{1}{\lambda} \quad\blacksquare \nonumber
\end{align}$$

---

> **Proposition 9 (Sample gradient variance and (Nesterov) momentum size bound).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda > 0$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then, for all $t \geq 0$ and arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|_F^2 \right]
        &\leq \frac{\sigma^2}{b} \\
    \mathbb{E}\left[ \| M_t \right]
        &\leq \frac{\sigma^2}{b} \\
    \mathbb{E}\left[ \| C_t \right]
        &\leq \frac{\sigma^2}{b}
\end{align}$$

**Proof.** From Assumption (1) and Proposition (4), we have,
$$\begin{align}
    \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) - \nabla f(W_t) \|_F^2 \right]
        &= \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|_F^2 \right]
            - \cancel{2\mathbb{E}\left[ \langle \nabla f_{S_t}(W_t), \nabla f(W_t) \rangle \right]}
            + \mathbb{E}\left[ \| \nabla f(W_t) \|_F^2 \right] \nonumber \\
    \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|_F^2 \right]
        &= \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) - \nabla f(W_t) \|_F^2 \right]
            - \mathbb{E}\left[ \| \nabla f(W_t) \|_F^2 \right] \nonumber \\
        &\leq \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) - \nabla f(W_t) \|_F^2 \right] \nonumber \\
        &\leq \frac{\sigma^2}{b} \nonumber
\end{align}$$

Then, let us unroll the momentum recurrence,
$$\begin{align}
    \mathbb{E}\left[ \| M_t \|_F^2 \right]
        &= \mathbb{E}\left[ \| \beta M_{t-1} + (1 - \beta) \nabla f_{S_t}(W_t) \|_F^2 \right] \nonumber \\
        &\leq \beta \mathbb{E}\left[ \| M_{t-1} \|_F^2 \right] + (1 - \beta) \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|_F^2 \right] \nonumber \\
        &\leq \beta^t \| M_0 \|_F^2 + (1 - \beta) \sum_{i=0}^{t-1} \frac{\sigma^2}{b} \beta^i \nonumber \\
        &\leq \frac{\sigma^2}{b} \nonumber
\end{align}$$

As for the Nesterov momentum term, we have,
$$\begin{align}
    \mathbb{E}\left[ \| C_t \|_F^2 \right]
        &= \mathbb{E}\left[ \| \beta M_t + (1 - \beta) \nabla f_{S_t}(W_t) \|_F^2 \right] \nonumber \\
        &\leq \beta \mathbb{E}\left[ \| M_t \|_F^2 \right] + (1 - \beta) \mathbb{E}\left[ \| \nabla f_{S_t}(W_t) \|_F^2 \right] \nonumber \\
        &\leq \frac{\sigma^2}{b} \quad\blacksquare \nonumber
\end{align}$$

---

> **Theorem 10 (Convergence bound with weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then for an arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, there exist constants $X, Y, Z > 0$ such that,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F] \leq \frac{X}{T} + \frac{Y}{b} + Z
\end{equation}$$
where $T$ is the total number of time steps, $b$ is the batch size, and
$$Y = \frac{(2 \beta + 1)(1 - \beta) + \lambda}{2} \sigma^2.$$

We follow a similar proof strategy as Theorem (7).

$$\begin{align}
    f(W_{t+1})
        &\leq f(W_t) + \langle \nabla f(W_t), W_{t+1} - W_t \rangle + \frac{L}{2} \| W_{t+1} - W_t \|^2 \nonumber \\
        &\leq f(W_t) + \langle \nabla f(W_t), \eta A_t^* - \lambda\eta W_{t} \rangle + \frac{L}{2} \| \eta A_t^* - \lambda\eta W_{t} \|^2 \nonumber \\
        &\leq f(W_t) + \langle \nabla f(W_t) - C_t + C_t, \eta A_t^* - \lambda\eta W_{t} \rangle_F + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t) + \langle C_t, \eta A_t^* \rangle_F + \lambda\eta \langle C_t, -W_{t} \rangle_F + \langle \nabla f(W_t) - C_t, \eta A_t^* - \lambda\eta W_{t} \rangle_F + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \| C_t \|^{\dagger}
            + \lambda\eta \left(\frac{\epsilon'}{2} \| C_t \|^{\dagger 2} + \frac{1}{2\epsilon'} \| -W_t \|^2 \right) \nonumber \\
            &\qquad+ \left(\frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac {\eta^2}{2 \epsilon} \| A_t^* - \lambda\eta W_{t} \|^2\right)
            + \frac{L \eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \left(\| \nabla f(W_t) \|^{\dagger} - \| \nabla f(W_t) - C_t \|^{\dagger}\right)
            + \frac{\lambda\eta\epsilon'}{2} \| C_t \|^{\dagger 2}
            + \frac{\lambda\eta}{2\epsilon'} \| W_t \|^2
            \nonumber \\
            &\qquad+ \frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2}
                + \frac {\eta^2}{2 \epsilon} \left(2\| A_t^* \| + 2\lambda\eta \| W_{t} \|^2 \right)
                + \frac{L\eta^2}{2} \nonumber \\
        &\leq f(W_t)
            - \eta \| \nabla f(W_t) \|^{\dagger}
            + \eta \| \nabla f(W_t) - C_t \|^{\dagger}
            + \frac{\epsilon}{2}\| \nabla f(W_t) - C_t \|^{\dagger 2} \nonumber \\
            &\qquad + \frac{\lambda\eta\epsilon'}{2} \| C_t \|^{\dagger 2} 
                + \frac{\lambda\eta(2\eta^2/\epsilon + 1/\epsilon')}{2} \| W_t \|^2
                + \frac{(2/\epsilon + L)\eta^2}{2} \label{eq:descentlemma-weightdecay}
\end{align}$$

Again, rearranging Equation $\eqref{eq:descentlemma-weightdecay}$ with $\epsilon = \epsilon' = \frac{\kappa_1 \eta}{\kappa_2^2}$, and using Proposition (8) and Proposition (9), we have,
$$\begin{align}
    \| \nabla f(W_t) \|^{\dagger}
        &\leq \frac{f(W_t) - f(W_{t+1})}{\eta}
            + \| \nabla f(W_t) - C_t \|^{\dagger}
            + \frac{\epsilon}{2\eta} \| \nabla f(W_t) - C_t \|^{\dagger 2} \nonumber \\
        &\quad
            + \frac{\lambda\epsilon'}{2} \| C_t \|^{\dagger 2}
            + \frac{\lambda(2\eta^2/\epsilon + 1/\epsilon')}{2} \| W_t \|^2
            + \frac{(2/\epsilon + L)\eta^2}{2} \nonumber \\
    \| \nabla f(W_t) \|_F
        &\leq \frac{f(W_t) - f(W_{t+1})}{\eta\kappa_1}
            + \frac{\kappa_2}{\kappa_1}\| \nabla f(W_t) - C_t \|_F
            + \frac{1}{2} \| \nabla f(W_t) - C_t \|^{2}_F \nonumber \\
        &\quad
            + \frac{\lambda}{2} \frac{\sigma^2}{b}
            + \frac{\kappa_2^2}{\kappa_1}\frac{2\eta^2 + 1}{2\lambda \eta}
            + \frac{(2 \kappa_2^2/\kappa_1 + L\eta)\eta}{2\kappa_1} \nonumber \\
\end{align}$$

Thus, after taking expectations and averaging, we have,
$$\begin{align}
    \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F]
        &\leq \frac{X}{T} + \frac{Y}{b} + Z
\end{align}$$
where,
$$\begin{align}
    X
        &:= \frac{f(W_0) - f^*}{\eta\kappa_1}
            + \frac{2\sqrt{2}\beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} \| \nabla f(W_0) - M_0 \|_F
            + \frac{\beta}{1 - \beta} \| \nabla f(W_0) - M_0 \|_F^{2} \nonumber \\
    Y
        &:= \frac{(2 \beta + 1)(1 - \beta) + \lambda}{2} \sigma^2 \nonumber \\
    Z
        &:= \frac{2 (\kappa_2^2/\kappa_1 + L\eta)\eta}{2\kappa_1}
            + \frac{2 \beta}{1 - \beta} \frac{\kappa_2}{\kappa_1} L \eta
            + \frac{2\beta}{(1 - \beta)^2} L^2 \eta^2 \nonumber \\
        &\qquad
            + \frac{\kappa_2^2}{\kappa_1}\frac{2\eta^2 + 1}{2\lambda \eta}
            + \left(\sqrt{2 (1 - \beta)} + (1 - \beta)\right) \frac{\kappa_2}{\kappa_1} \frac{\sigma}{\sqrt{b}} \quad\blacksquare \nonumber
\end{align}$$

---

## Deriving the critical batch size

> **Theorem 11 (Critical batch size for steepest descent under arbitrary norms with (Nesterov) momentum and weight decay).** Let $W_t$ be the weight at time step $t$ updated according to Equation $\eqref{eq:updateweightdecay}$ with weight decay parameter $\lambda$ and step size $\eta > 0$ such that $\lambda \eta \leq 1$, $\| W_0 \| \leq \frac{1}{\lambda}$, and $M_0 = 0$. Then for an arbitrary norm pair $(\| \cdot \|, \| \cdot \|^{\dagger})$, the critical batch size $b_{crit}$ that minimizes the total number of tokens processed to reach convergence according to the criterion in Equation $\eqref{eq:convergence-criterion}$ is given by,
$$\begin{equation}
    b_{crit} = \left( (2\beta + 1)(1 - \beta) + \lambda \right) \frac{\sigma^2}{\epsilon'}
\end{equation}$$

**Proof.** We consider the steepest descent iteration process to have converged at time step $T$ when, for some $\epsilon > 0$,
$$\begin{equation}
    \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\| \nabla f(W_t) \|_F] \leq \frac{X}{T} + \frac{Y}{b} + Z \leq \epsilon \label{eq:convergence-criterion}
\end{equation}$$
Since $Z$ is a constant independent of $T$ and $b$, we can simply fold it into $\epsilon$ by defining $\epsilon' := \epsilon - Z > 0$. Simple algebra then yields the number of iterations to satisfy the convergence criterion in Equation $\eqref{eq:convergence-criterion}$ as,
$$\begin{align}
    \frac{X}{T} + \frac{Y}{b} + Z &\leq \epsilon \nonumber \\
    \frac{X}{T} + \frac{Y}{b} &\leq \epsilon - Z =: \epsilon' \nonumber \\
    \frac{Xb}{T} + Y &\leq \epsilon' b \nonumber \\
    \frac{Xb}{\epsilon' b - Y} &\leq T \nonumber \\
    \frac{Xb}{\epsilon' b - Y} &=: T(b)
\end{align}$$
Note that we also have to constraint $b > \frac{Y}{\epsilon'}$ to ensure that $T(b) > 0$. Taking the first and second derivatives then yields,
$$\begin{align}
    \frac{dT(b)}{db} &= -\frac{XY}{(\epsilon' b - Y)^2} \leq 0 \nonumber \\
    \frac{d^2T(b)}{db^2} &= \frac{2XY\epsilon'}{(\epsilon' b - Y)^3} \geq 0 \nonumber
\end{align}$$
Thus, $T(b)$ is a monotonically decreasing and convex function for $b > \frac{Y}{\epsilon'}$.

Now, the number of tokens we need to process to reach convergence is roughly proportional to,
$$b \cdot T(b) = \frac{Xb^2}{\epsilon' b - Y}$$
Taking the first and second derivatives again yields,
$$\begin{align}
    \frac{d(b \cdot T(b))}{db} &= \frac{Xb(\epsilon' b - 2Y)}{(\epsilon' b - Y)^2} \nonumber \\
    \frac{d^2(b \cdot T(b))}{db^2} &= \frac{2XY^2}{(\epsilon' b - Y)^3} \geq 0 \nonumber
\end{align}$$
Thus, $b \cdot T(b)$ is a convex function for $b > \frac{Y}{\epsilon'}$, with a minimizer $b^* = \frac{2Y}{\epsilon'}$. This gives us the critical batch size,
$$\begin{equation}
    b_{crit} = \frac{2Y}{\epsilon'} = \left( (2\beta + 1)(1 - \beta) + \lambda \right) \frac{\sigma^2}{\epsilon'}
\end{equation}$$
