---
title: "Adam with Agressive Gradient Clipping ≈ Smoothed SignSGD/NormSGD"
date: 2025-07-03
tags: ["Machine Learning", "Optimizers"]
author: "Franz Louis Cesista"
description: "Why does Adam with aggressive gradient value/norm clipping have sparse updates and do well with higher learning rates? Here we show that it is essentially equivalent to a smoothed version of SignSGD/NormSGD."
summary: "Why does Adam with aggressive gradient value/norm clipping have sparse updates and do well with higher learning rates? Here we show that it is essentially equivalent to a smoothed version of SignSGD/NormSGD."
# cover:
#     image: cover.jpg
#     alt: "Cover"
#     relative: true
# editPost:
#     URL: "https://x.com/leloykun/status/1883634169902952655"
#     Text: "Crossposted from X (formerly Twitter)"
---

Here we will show that Adam with aggressive gradient value/norm clipping is essentially equivalent to a smoothed version of SignSGD/NormSGD. We will also explain why the commulative updates are sparse and why it does well with higher learning rates.

## Smoothed SignSGD and Smoothed NormSGD

Unlike Signum which applies momentum first before taking the sign, Smoothed SignSGD applies the sign first before applying momentum.
> **Definition 1 (Smoothed SignSGD).** Let $\eta > 0$ be the learning rate, $\beta \in [0, 1)$ be the momentum coefficient, and $G_t$ be the gradient at time step $t$. The update rule for Smoothed SignSGD is given by:
> $$\begin{align}
M_{t}^{\text{sssgd}} &= \beta M_{t-1}^{\text{sssgd}} + (1 - \beta) \text{sign}(G_t) \\\\
W_{t+1} &= W_t - \eta M_{t}^{\text{sssgd}}\nonumber
\end{align}$$
> where $M_t$ is the momentum at time step $t$, $W_t$ is the model parameters at time step $t$, and $\text{sign}(\cdot)$ is the element-wise sign function.

Note that unfolding the recurrence in Equation (1) gives us,
$$\begin{equation}M_{t}^{\text{sssgd}} = (1 - \beta)\sum_{k=0}^{t-1}\beta^k\text{sign}(G_{t-k})\end{equation}$$

Likewise, we can define Smoothed NormSGD as follows:
> **Definition 2 (Smoothed NormSGD).** Let $\eta > 0$ be the learning rate, $\beta \in [0, 1)$ be the momentum coefficient, $||\cdot||$ be a norm chosen a priori, and $G_t$ be the gradient at time step $t$. The update rule for Smoothed NormSGD is given by:
> $$\begin{align}
M_{t}^{\text{snsgd}} &= \beta M_{t-1}^{\text{snsgd}} + (1 - \beta) \frac{G_t}{||G_t||} \\\\
W_{t+1} &= W_t - \eta M_{t}^{\text{snsgd}}\nonumber
\end{align}$$
> where $M_t$ is the momentum at time step $t$, $W_t$ is the model parameters at time step $t$.

And again, unfolding the recurrence in Equation (3) gives us,
$$\begin{equation}M_{t}^{\text{snsgd}} = (1 - \beta)\sum_{k=0}^{t-1}\beta^k\frac{G_{t-k}}{||G_{t-k}||}\end{equation}$$

## Adam with aggressive gradient *value* clipping is equivalent to Smoothed SignSGD

Here we apply gradient clipping element-wise with threshold $\alpha > 0$:
$$G\_{t,i,j}^{\text{clipped}} = \text{clip}\_{[-\alpha, \alpha]}(G\_{t,i,j})
= \begin{cases}
    \alpha\cdot\text{sign}(G\_{t,i,j}) & \text{if } |G\_{t,i,j}| \geq \alpha \\\\
    G\_{t,i,j} & \text{if } |G\_{t,i,j}| < \alpha
\end{cases}
$$
With *aggressive gradient value clipping* (i.e., $\alpha \to 0$), we can make the simplifying assumtion that $|G\_{t,i,j}| \geq \alpha$ for all $t, i, j$. Thus we have,
$$G\_{t}^{\text{clipped}} = \alpha\cdot\text{sign}(G\_{t})$$

Passing this through Adam's update rule, we get:

$$\begin{align*}
M_{t}^{\text{adam}}
    &= \beta_1 M_{t-1}^{\text{adam}} + (1 - \beta_1)G\_{t}^{\text{clipped}} \\\\
    &= \beta_1 M_{t-1}^{\text{adam}} + (1 - \beta_1)\alpha\cdot\text{sign}(G_t) \\\\
    &= \alpha(1 - \beta_1)\sum_{k=0}^{t-1}\beta_1^k\text{sign}(G_{t-k}) \\\\
M_{t}^{\text{adam}} &= \alpha M_{t}^{\text{sssgd}}
\end{align*}$$

and,

$$\begin{align*}
V_{t}^{\text{adam}}
    &= \beta_2 V_{t-1}^{\text{adam}} + (1 - \beta_2)(G\_{t}^{\text{clipped}})^2 \\\\
    &= \beta_2 V_{t-1}^{\text{adam}} + (1 - \beta_2)(\alpha\cdot\text{sign}(G_t))^2 \\\\
    &= \alpha^2(1 - \beta_2)\sum_{k=0}^{t-1}\beta_2^k\text{sign}(G_{t-k})^2 \\\\
    &= \alpha^2(1 - \beta_2)\sum_{k=0}^{t-1}\beta_2^k\mathbb{1}\qquad\qquad\text{from the assumption that } |G_{t-k}| \geq \alpha \\\\
V_{t}^{\text{adam}}
    &=\alpha^2 (1 - \beta_2^t)\mathbb{1}
\end{align*}$$

Thus the update direction becomes,
$$\begin{align*}
U_t &= \frac{M_t^{\text{adam}} / (1 - \beta_1^t)}{\sqrt{V_t^{\text{adam}} / (1 - \beta_2^t)}} \\\\
    &= \frac{\alpha M_t^{\text{sssgd}} / (1 - \beta_1^t)}{\sqrt{\alpha^2 (1 - \beta_2^t)\mathbb{1} / (1 - \beta_2^t)}} \\\\
U_t &= \frac{1}{(1 - \beta_1^t)} M_t^{\text{sssgd}}
\end{align*}$$
Note that the $\alpha$ terms cancel out. And as $t \to \infty$, we have $\beta_1^t \to 0$. Thus,
$$U_t \to M_t^{\text{sssgd}}\qquad\text{as}\qquad t \to \infty$$

Hence Adam with aggressive gradient value clipping is just Smoothed SignSGD!

### Why are Smoothed SignSGD updates sparse?

Let's go back to Equation (2):
$$M_{t}^{\text{sssgd}} = (1 - \beta)\sum_{k=0}^{t-1}\beta^k\text{sign}(G_{t-k})$$
and let's pick an arbitrary entry $G_{t,i,j}$. Notice that if the signs of the recent $G_{t,i,j}$s flip too much, then the $\beta^0$, $\beta^1$, $\beta^2$, ... terms effectively cancel each other out. Thus that entry will not contribute to the update. On the other hand, if the signs of the recent $G_{t,i,j}$s are aligned, then $M_{t,i,j}^{\text{sssgd}} \to \pm 1$. What this means is that for a given entry, the weights only get updated if the signs of the recent gradients are aligned.

## Adam with aggressive gradient *norm* clipping is essentially equivalent Smoothed NormSGD

Unlike the previous case, here we apply the clipping on the norm of the gradient. That is, for a given threshold $\alpha > 0$, we have:
$$G\_{t}^{\text{clipped}} = \begin{cases}
    \frac{\alpha}{||G\_{t}||}G\_{t} & \text{if } ||G\_{t,i,j}|| \geq \alpha \\\\
    G\_{t,i,j} & \text{if } ||G\_{t,i,j}|| < \alpha
\end{cases}$$
And with *aggressive gradient norm clipping*, we can assume that $||G_{t}|| \geq \alpha$ for all $t$.

And like before, passing this through Adam's update rule, we get:
$$\begin{align*}
M_{t}^{\text{adam}}
    &= \beta_1 M_{t-1}^{\text{adam}} + (1 - \beta_1)G\_{t}^{\text{clipped}} \\\
    &= \beta_1 M_{t-1}^{\text{adam}} + (1 - \beta_1)\frac{\alpha}{||G_{t}||}G_{t} \\\\
    &= \alpha(1 - \beta_1)\sum_{k=0}^{t-1}\beta_1^k \frac{G_{t-k}}{||G_{t}||} \\\\
M_{t}^{\text{adam}} &= \alpha M_{t}^{\text{snsgd}}
\end{align*}$$
and,
$$\begin{align*}
V_{t}^{\text{adam}}
    &= \beta_2 V_{t-1}^{\text{adam}} + (1 - \beta_2)(G\_{t}^{\text{clipped}})^2 \\\\
    &= \beta_2 V_{t-1}^{\text{adam}} + (1 - \beta_2)\left(\frac{\alpha}{||G_{t}||}G_{t}\right)^2 \\\\
    &= \alpha^2(1 - \beta_2)\sum_{k=0}^{t-1}\beta_2^k\left(\frac{G_{t-k}}{||G_{t-k}||}\right)^2 \\\\
V_{t}^{\text{adam}}
    &= \alpha^2(1 - \beta_2^t) S_t
\end{align*}$$
where
$$S_t = \frac{1 - \beta_2}{1 - \beta_2^t} \sum_{k=0}^{t-1}\beta_2^k\left(\frac{G_{t-k}}{||G_{t-k}||}\right)^2$$

For an arbitrary index $i,j$, notice that $G_{t-k,i,j} / ||G_{t-k}|| \leq 1$ for all $k$. The (entrywise) sum in the RHS then is minimized when $G_{t-k,i,j} = 0$ for all $k$. And the sum is maximized when $G_{t-k,i,j} / ||G_{t-k}|| = 1$ for all $k$. Together, we have:
$$0 \leq S_{t,i,j} \leq \frac{1 - \beta_2}{1 - \beta_2^t} \sum_{k=0}^{t-1}\beta_2^k = \frac{1 - \beta_2}{1 - \beta_2^t} \left( \frac{1 - \beta_2^t}{1 - \beta_2} \right) = 1$$

The (Adam) update direction then becomes:
$$\begin{align*}
U_t &= \frac{M_t^{\text{adam}} / (1 - \beta_1^t)}{\sqrt{V_t^{\text{adam}} / (1 - \beta_2^t)}} \\\\
    &= \frac{\alpha M_t^{\text{snsgd}} / (1 - \beta_1^t)}{\sqrt{\alpha^2(1 - \beta_2^t) S_t / (1 - \beta_2^t)}} \\\\
U_t &= \frac{1}{(1 - \beta_1^t)S_t} M_t^{\text{snsgd}}
\end{align*}$$
And if we make the further assumption that the gradients are isotopic, or more intuitively speaking, the gradients statistically do not 'change' over time, then we can treat $S_t$ as a constant. Thus,
$$U_t \to \text{constant}\cdot M_t^{\text{snsgd}}\qquad\text{as}\qquad t \to \infty$$
Hence Adam with aggressive gradient norm clipping is essentially just Smoothed NormSGD.

### Why are Smoothed NormSGD updates sparse?

Roughly the same reasoning applies as in the case of Smoothed SignSGD. The main difference is that the weights to the betas are, in a sense, "denser":
$$\text{sign}(G_{t-k,i,j}) \in \\{-1, 0, 1\\}\qquad\text{ vs. }\qquad\frac{G_{t-k,i,j}}{||G_{t-k}||} \in [-1, 1]$$

And so, we do have to be careful of the case where the gradient norms have a lot of variance. But in practice, the norms of the gradients are usually stable, except for some gradient spikes here and there, so it's a non-issue. That said, let's go back to Equation (4):
$$M_{t}^{\text{snsgd}} = (1 - \beta)\sum_{k=0}^{t-1}\beta^k\frac{G_{t-k}}{||G_{t-k}||}$$
and let's pick an arbitrary entry $G_{t,i,j}$. Like before, if the signs of the recent $G_{t,i,j}$s flip too much, then the $\beta^0$, $\beta^1$, $\beta^2$, ... terms effectively cancel each other out (we use the assumption that the norms are stable here). Otherwise, if they're aligned, then $M_{t,i,j}^{\text{snsgd}} \to \pm 1$. Thus, as before, the weights only get updated if the recent gradients are aligned.

### Why do Smoothed SignSGD and Smoothed NormSGD do well with higher learning rates?

(Smoothed) SignSGD and NormSGD destroy the magnitude information of the gradients. So picking a higher learning rate is, in a sense, a way to compensate for the lost magnitude information. The momentum also stabilizes the updates, so we can afford to pick a higher learning rate without worrying about overshooting that much.
