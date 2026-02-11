---
title: "Error-Compensating Optimizers: Handling Weight Decay and Matrix LMOs"
date: 2026-02-10
tags: ["Machine Learning", "Optimizers", "Muon"]
author: ["Franz Louis Cesista"]
description: "-"
summary: "-"
draft: true
---

## 1. Introduction

To optimize inference in production settings, we typically need to quantize the model weights first, and then do all subsequent computation in low precision. But training in low precision natively is difficult, and so we often pretrain in high precision, quantize, then fine-tune to recover any lost performance. This wastes compute and we often do not fully recover the original performance.

Error-Compensating Optimizers (ECO), on the other hand, allows the (idealized) master weights to evolve in high precision while only materializing quantized weights ([Nikdan et. al, 2026](https://arxiv.org/abs/2601.22101)). This prevents performance degradation from quantization, and eliminates the need for a separate fine-tuning phase. The crux is to (1) compute the gradients with respect to the quantized weights, and (2) 'pull' the quantization 'error' back into the momentum buffer for use in the next step. 

Here we shall discuss how to handle weight decay, matrix LMOs, and federated learning in the ECO framework.

## 2. ECO with weight decay

### 2.1. ECO for SGD with momentum and weight decay

Let $q(\cdot)$ be the quantization function, $W_t^*$ and $M_t^*$ be the (idealized)master weights and momentum, and $\widehat{W}_t$ and $M_t$ be the (materialized) quantized weights and (unquantized) momentum at step $t$. Then, the master-weight SGD update with momentum and weight decay is given by,
$$\begin{align}
    \widehat{W}_t &= q(W_t^*) \\
    G_t &= \nabla L(\widehat{W}_t) \\
    M_{t+1}^* &= \beta M_t^* + (1-\beta) G_t \label{eq:sgdm_m_update} \\
    W_{t+1}^* &= (1 - \eta \lambda) W_t^* - \eta M_{t+1}^* \label{eq:sgdm_w_update}
\end{align}$$
where $\beta, \eta, \lambda$ are the momentum, learning rate, and weight decay hyperparameters, respectively.

However, we do not want to materialize the master weights $W_t^*$, but only the quantized weights $\widehat{W}_t$, hence the ECO-style update of the form,
$$\begin{align}
    G_t
        &= \nabla L(\widehat{W}_t) \\
    \widetilde{M}_{t+1}
        &= \beta M_{t} + (1-\beta) G_t \label{eq:eco_sgdm_m_update} \\
    \widetilde{W}_{t+1}
        &= (1 - \eta \lambda) \widehat{W}_t - \eta \widetilde{M}_{t+1} \label{eq:eco_sgdm_w_update} \\
    \widehat{W}_{t+1}
        &= q(\widetilde{W}_{t+1}) \\
    E_{t+1}
        &= \widetilde{W}_{t+1} - \widehat{W}_{t+1} \label{eq:eco_sgdm_error} \\
    M_{t+1}
        &= g(\widetilde{M}_{t+}, E_{t+1}),
\end{align}$$
where $\widetilde{M}_{t+}$ and $\widetilde{W}_{t+1}$ are intermediate variables, and $g$ is the error-compensation function that 'pulls' the quantization 'error' back into the momentum buffer for use in the next step. We derive $g$ as follows.

First, since $\widetilde{W}_t = \widehat{W}_t + E_t$, it is sensible to set,
$$\begin{align}
    W_t^*
        &:= \widetilde{W}_t = \widehat{W}_t + E_t.
\end{align}$$

Combining Equations $\eqref{eq:sgdm_w_update}$, $\eqref{eq:eco_sgdm_w_update}$, and $\eqref{eq:eco_sgdm_error}$, we have,
$$\begin{align}
    W_{t+1}^*
        &= \widetilde{W}_{t+1} \nonumber \\
    (1 - \eta \lambda) W_t^* - \eta M_{t+1}^*
        &= (1 - \eta \lambda) \widehat{W}_t - \eta \widetilde{M}_{t+1} \nonumber \\
    (1 - \eta \lambda) (\widehat{W}_t + E_t) - \eta M_{t+1}^*
        &= (1 - \eta \lambda) \widehat{W}_t - \eta \widetilde{M}_{t+1} \nonumber \\
    M_{t+1}^*
        &= \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta} E_t \label{eq:sgdm_m_update_expanded}
\end{align}$$

Now let $\alpha_t := M_t^* - M_t$. Then combining Equations $\eqref{eq:sgdm_m_update}$, $\eqref{eq:eco_sgdm_m_update}$, and $\eqref{eq:sgdm_m_update_expanded}$, we have,
$$\begin{align}
    \beta M_t^* + (1-\beta) G_t
        &= \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta} E_t \nonumber \\
    \beta (M_t + \alpha_t) + (1-\beta) G_t
        &= \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta} E_t \nonumber \\
    \cancel{\widetilde{M}_{t+1}} + \beta \alpha_t
        &= \cancel{\widetilde{M}_{t+1}} + \frac{1 - \eta \lambda}{\eta} E_t \nonumber \\
    \alpha_t
        &= \frac{1 - \eta \lambda}{\beta \eta} E_t
\end{align}$$
and since this holds for all $t$, we have,
$$\begin{align}
    \alpha_{t+1}
        &= \frac{1 - \eta \lambda}{\beta \eta} E_{t+1} \nonumber \\
    M_{t+1}
        &= M_{t+1}^* - \frac{1 - \eta \lambda}{\beta \eta} E_{t+1} \nonumber \\
        &= \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta} E_t - \frac{1 - \eta \lambda}{\beta \eta} E_{t+1}
\end{align}$$

Lastly, we can eliminate the dependence on $E_t$ using the heuristic $E_t \approx E_{t+1}$ observed in practice, which gives us the final error-compensating momentum update rule:
$$\begin{align}
    M_{t+1}
        &\approx \widetilde{M}_{t+1} + \frac{\color{red}{1 - \eta \lambda}}{\eta}\left(1 - \frac{1}{\beta}\right) E_{t+1} \label{eq:sgdm_error_compensation}
\end{align}$$
where the red-colored term is the difference from Algorithm 2 in the ECO paper.

### 2.2. ECO for steepest descent with LMOs of the form $\texttt{LMO}(X) = Xh(X)$

Steepest descent under a norm $\| \cdot \|$ with Linear Minimization Oracle (LMO) $\texttt{LMO}(X)$ has the update rule,
$$\begin{align}
    \widehat{W}_t &= q(W_t^*) \\
    G_t &= \nabla L(\widehat{W}_t) \\
    M_{t+1}^* &= \beta M_t^* + (1-\beta) G_t \label{eq:sgdm_m_update_2} \\
    U_{t+1}^* &= \texttt{LMO}(M_{t+1}^*) \\
    W_{t+1}^* &= (1 - \eta \lambda) W_t^* - \eta U_{t+1}^* \label{eq:sgdm_w_update_2}
\end{align}$$

As in the earlier section, the ECO-style update is given by,
$$\begin{align}
    G_t
        &= \nabla L(\widehat{W}_t) \\
    \widetilde{M}_{t+1}
        &= \beta M_{t} + (1-\beta) G_t \label{eq:eco_sgdm_m_update_2} \\
    U_{t+1}
        &= \texttt{LMO}(\widetilde{M}_{t+1}) \\
    \widetilde{W}_{t+1}
        &= (1 - \eta \lambda) \widehat{W}_t - \eta U_{t+1} \label{eq:eco_sgdm_w_update_2} \\
    \widehat{W}_{t+1}
        &= q(\widetilde{W}_{t+1}) \\
    E_{t+1}
        &= \widetilde{W}_{t+1} - \widehat{W}_{t+1} \label{eq:eco_sgdm_error_2} \\
    M_{t+1}
        &= g(\widetilde{M}_{t+}, E_{t+1}),
\end{align}$$

Setting $W_t^* := \widetilde{W}_t$ as before, we have,
$$\begin{align}
    W_{t+1}^*
        &= \widetilde{W}_{t+1} \nonumber \\
    (1 - \eta \lambda) W_t^* - \eta U_{t+1}^*
        &= (1 - \eta \lambda) \widehat{W}_t - \eta U_{t+1} \nonumber \\
    (1 - \eta \lambda) (\widehat{W}_t + E_t) - \eta U_{t+1}^*
        &= (1 - \eta \lambda) \widehat{W}_t - \eta U_{t+1} \nonumber \\
    U_{t+1}^*
        &= U_{t+1} + \frac{1 - \eta \lambda}{\eta} E_t
\end{align}$$

We then make the following first-order approximation of the LMO:
$$\begin{align}
    \texttt{LMO}(X + \Delta X)
        &\approx \texttt{LMO}(X) + \Delta X h(X)
\end{align}$$

Thus, for LMOs of the form $\texttt{LMO}(X) = X h(X)$ with invertible $h(X)$, we have,
$$\begin{align}
    U_{t+1}^*
        &= \texttt{LMO}(\widetilde{M}_{t+1} + (M_{t+1}^* - \widetilde{M}_{t+1})) \nonumber \\
        &\approx U_{t+1} + (M_{t+1}^* - \widetilde{M}_{t+1}) h(\widetilde{M}_{t+1}) \nonumber \\
    M_{t+1}^*
        &\approx \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta} E_t h^{-1}(\widetilde{M}_{t+1})
\end{align}$$

Following the same steps as before then yields,
$$\begin{align}
    M_{t+1}
        &\approx \widetilde{M}_{t+1} + \frac{\color{red}{1 - \eta \lambda}}{\eta}\left(1 - \frac{1}{\beta}\right) E_{t+1} \color{red}{h^{-1}(\widetilde{M}_{t+1})}
\end{align}$$

Specializing to steepest descent under the spectral norm (as in the Muon optimizer), we have,
$$\begin{align}
    \texttt{LMO}(X)
        &= \texttt{msign}(X) = X \underbrace{(X^T X)^{-1/2}}_{h(X)} \nonumber \\
    M_{t+1}
        &\approx \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta}\left(1 - \frac{1}{\beta}\right) E_{t+1} (\widetilde{M}_{t+1}^T \widetilde{M}_{t+1})^{1/2} \\
    M_{t+1}
        &\approx \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta}\left(1 - \frac{1}{\beta}\right) E_{t+1} (\widetilde{M}_{t+1}^T \widetilde{M}_{t+1})(\widetilde{M}_{t+1}^T \widetilde{M}_{t+1})^{-1/2},
\end{align}$$
which we can compute via right-multiplication with a matrix r-th root as in [Ponder: Shampoo-PRISM: Kronecker-Factored Optimization via Anisotropic Spectral Shaping](../shampoo-prism/).

### 2.3. ECO for steepest descent with LMOs of the form $\texttt{LMO}(X) = X \odot h(X)$

Following the steps above for LMOs of the form $\texttt{LMO}(X) = X \odot h(X)$, we instead have,
$$\begin{align}
    M_{t+1}^*
        &\approx \widetilde{M}_{t+1} + \frac{1 - \eta \lambda}{\eta} \frac{1}{h(\widetilde{M}_{t+1})} \odot E_t \\
    M_{t+1}
        &\approx \widetilde{M}_{t+1} + \frac{\color{red}{1 - \eta \lambda}}{\eta}\left(1 - \frac{1}{\beta}\right) {\color{red}{\frac{1}{h(\widetilde{M}_{t+1})} \odot}} E_{t+1}
\end{align}$$

For AdamW, we have, $\texttt{LMO}(M_t) = \frac{M_t / (1 - \beta_1^t)}{\sqrt{V_t} / (1 - \beta_2^t) + \epsilon}$, where $V_t$ is the second moment accumulator, and so we have,
$$\begin{align}
    M_{t+1}
        &\approx \widetilde{M}_{t+1} + \frac{{\color{red}{(1 - \eta \lambda)}}(1 - \beta_1^{t+1})}{\eta} \left( 1 - \frac{1}{\beta} \right) \left( \frac{\sqrt{V_{t+1}}}{1 - \beta_2^{t+1}} + \epsilon \right) \odot E_{t+1}
\end{align}$$

## How to cite

```bibtex
@misc{cesista2026eco,
  author = {Franz Louis Cesista},
  title = {Error-Compensating Optimizers: Handling Weight Decay and Matrix LMOs},
  year = {2026},
  month = {February},
  day = {10},
  url = {https://leloykun.github.io/ponder/eco/},
}
```


## References

1. Mahdi Nikdan, Amir Zandieh, Dan Alistarh, Vahab Mirrokni (2026). ECO: Quantized Training without Full-Precision Master Weights. URL https://arxiv.org/abs/2601.22101
2. Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher (2025). Training Deep Learning Models with Norm-Constrained LMOs. URL https://arxiv.org/abs/2502.07529