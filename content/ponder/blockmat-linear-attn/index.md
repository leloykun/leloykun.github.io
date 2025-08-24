---
title: "Blocked Matrix Formulation of Linear Attention Mechanisms"
date: 2025-03-16
tags: ["Machine Learning", "Linear Attention", "Test-Time Regression"]
author: "Franz Louis Cesista"
description: "The blocked matrix formulation of linear attention mechanisms, multi-step online gradient descent at inference time, and chunk-wise parallelism."
summary: "The blocked matrix formulation of linear attention mechanisms, multi-step online gradient descent at inference time, and chunk-wise parallelism."
# cover:
#     image: cover.jpg
#     alt: "Cover"
editPost:
    URL: "https://x.com/leloykun/status/1901267939267162351"
    Text: "Crossposted from X (formerly Twitter)"
---

In the [previous post](../test-time-regression/), we derived several linear attention mechanisms from scratch by formulating them as test-time online regression problems. Here, we'll discuss a more intuitive way to represent the update rules of the internal states of these linear attention mechanisms using a blocked matrix formulation. Then, we'll discuss how to use it to (1) derive the update rules for linear attention mechanisms that take multiple gradient descent steps per token and (2) derive the update rules for chunk-wise parallelism of already-existing linear attention mechanisms.

## Recap: Linear Attention Mechanisms

Linear attention mechanisms typically have an update rule of the form:
$$S_i = S_{i-1}A_i + B_i$$
where $S\_{i-1}$ is the (old) state after processing the first $i-1$ tokens, $S_i$ is the (new) state after processing the first $i$ tokens, and $A_i$ and $B_i$ are update matrices. Think of $A_i$ as an operation that *modifies* some information already stored in the state while $B_i$ *adds* new information to the state. In most cases where $A_i \neq I$, $A_i$ typically *removes* some (old) information from the state. But if we allow $A_i$ to have negative eigenvalues, then we can also think of it as an operation that, in a sense, *inverts* information instead.

Here are a couple of examples:

| Linear Attention Mechanism |                                                                          **$A_i$** |                     **$B_i$** |
| -------------------------- | ---------------------------------------------------------------------------------: | ----------------------------: |
| Vanilla Linear Attention   |                                                                                $I$ |         $\bm{v}_i \bm{k}_i^T$ |
| Mamba 2                    |                                              $\text{diag}\left(\alpha\_i I\right)$ |         $\bm{v}_i \bm{k}_i^T$ |
| DeltaNet                   |                                                  $I - \beta_i \bm{k}_i \bm{k}_i^T$ | $\beta_i \bm{v}_i \bm{k}_i^T$ |
| Gated DeltaNet             |                                        $\alpha_i(I - \beta_i \bm{k}_i \bm{k}_i^T)$ | $\beta_i \bm{v}_i \bm{k}_i^T$ |
| RWKV-7                     | $\text{diag}(\bm{w}_i) - \bm{\hat{\kappa}}_i(\bm{a}_i \odot\bm{\hat{\kappa}}_i^T)$ |         $\bm{v}_i \bm{k}_i^T$ |

where $\bm{k}_i  \in \mathbb{R}^{d_k}$ and $\bm{v}_i \in \mathbb{R}^{d_v}$ are the corresponding key-value pair for the $i$-th token; $\alpha_i \in [0, 1]$ can be thought of as a data-dependent weight decay that controls how much of the previous state to keep or forget; and $\beta_i \in [0, 1]$ can be thought of as a data-dependent learning rate that controls how much of the new information to add to the state.

If we let $\alpha_i \in [-1, 1]$ for Mamba 2 and $\beta_i \in [0, 2]$ for (Gated) DeltaNet, then $A_i$ can have negative eigenvalues while still having norm $\|\|A_i\|\| \leq 1$. This allows the models to learn more complex patterns while maintaining training stability (Grazzi et al., 2025).

## Blocked Matrix Formulation of Linear Attention Mechanisms

Notice that we can rewrite the update rule above as,

$$
\begin{align*}
    S_i &= S_{i-1}A_i + B_i\\\\
    S_{i} &=
        \begin{bmatrix}
            S_{i-1} & I
        \end{bmatrix}
        \begin{bmatrix}
            A_i \\\\
            B_i
        \end{bmatrix}
\end{align*}
$$
or, equivalently,
$$
\begin{bmatrix}
    S_{i} & I
\end{bmatrix} =
\begin{bmatrix}
    S_{i-1} & I
\end{bmatrix}
\begin{bmatrix}
    A_i & 0 \\\\
    B_i & I
\end{bmatrix}
$$

At training time, we need *all* of the intermediary states, not just the final state. Thus, we need an efficient way to compute $S_N$ for all token indices $N$. To do this, let's unroll the recurrence above:

$$
\begin{align*}
\begin{bmatrix}
    S_{N} & I
\end{bmatrix} &=
\begin{bmatrix}
    S_{N-1} & I
\end{bmatrix}
\begin{bmatrix}
    A_N & 0 \\\\
    B_N & I
\end{bmatrix}\\\\
\begin{bmatrix}
    S_{N} & I
\end{bmatrix} &=
\begin{bmatrix}
    S_{N-2} & I
\end{bmatrix}
\begin{bmatrix}
    A\_{N-1} & 0 \\\\
    B\_{N-1} & I
\end{bmatrix}
\begin{bmatrix}
    A_N & 0 \\\\
    B_N & I
\end{bmatrix}\\\\
&\vdots\\\\
\begin{bmatrix}
    S_{N} & I
\end{bmatrix} &=
\begin{bmatrix}
    S_{0} & I
\end{bmatrix}
\begin{bmatrix}
    A_1 & 0 \\\\
    B_1 & I
\end{bmatrix}
\begin{bmatrix}
    A_2 & 0 \\\\
    B_2 & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_N & 0 \\\\
    B_N & I
\end{bmatrix}\\\\
    S_N &=
\begin{bmatrix}
    S_{0} & I
\end{bmatrix}
\begin{bmatrix}
    A_1 & 0 \\\\
    B_1 & I
\end{bmatrix}
\begin{bmatrix}
    A_2 & 0 \\\\
    B_2 & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_N & 0 \\\\
    B_N & I
\end{bmatrix}
\begin{bmatrix}
    I \\\\
    0
\end{bmatrix}
\end{align*}
$$

In practice, we usually initialize $S_0$ as the zero matrix. Thus,

$$
\begin{align}
    S_N &=
\begin{bmatrix}
    0 & I
\end{bmatrix}
\begin{bmatrix}
    A_1 & 0 \\\\
    B_1 & I
\end{bmatrix}
\begin{bmatrix}
    A_2 & 0 \\\\
    B_2 & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_N & 0 \\\\
    B_N & I
\end{bmatrix}
\begin{bmatrix}
    I \\\\
    0
\end{bmatrix}\\\\
    S_N &=
\begin{bmatrix}
    0 & I
\end{bmatrix}
\begin{bmatrix}
    \prod_{i=1}^{N} A_i & 0 \\\\
    \sum_{i=1}^{N} \left(B_i \prod_{j=i+1}^{N} A_j\right) & I
\end{bmatrix}
\begin{bmatrix}
    I \\\\
    0
\end{bmatrix}\\\\
    S_N &= \sum_{i=1}^{N} \left(B_i \prod_{j=i+1}^{N} A_j\right)
\end{align}
$$
where $(1) \rightarrow (2)$ can be proven by induction.

Equation $(1)$ makes it obvious *why* and *how* we can parallelize computation of $S_N$, for all $N$, at training time: the updates are merely (blocked) matrix multiplications; matrix multiplications are associative; thus, we can use the (fully-parallel) associative scan algorithm to compute all the intermediary states in $O(N)$ time!

## One-Step Online Gradient Descent per Token

Let's derive $S_N$ for each of the linear attention mechanisms in the table above.

### Vanilla Linear Attention

{{< collapse summary="Show derivation of $S_N$" openByDefault=true >}}
$$A_i = I \quad\quad B_i = \bm{v}_i \bm{k}_i^T$$
From Equation $(3)$ above, we get:
$$
\begin{align*}
    S_N &= \sum\_{i=1}^{N} \left(\bm{v}\_i \bm{k}\_i^T \prod\_{j=i+1}^{N} I\right)\\\\
    S_N &= \sum\_{i=1}^{N} \bm{v}\_i \bm{k}\_i^T
\end{align*}
$$
{{< /collapse >}}

### Mamba 2

{{< collapse summary="Show derivation of $S_N$" >}}
$$A_i = \text{diag}\left(\alpha_i I\right) \quad\quad B_i = \bm{v}_i \bm{k}_i^T$$
Thus,
$$
\begin{align*}
    S_N &= \sum\_{i=1}^{N} \left(\bm{v}\_i \bm{k}\_i^T \prod\_{j=i+1}^{N} \text{diag}\left(\alpha_j I\right)\right)\\\\
    S_N &= \sum\_{i=1}^{N} \left( \prod\_{j=i+1}^{N} \alpha_j \right) \bm{v}\_i \bm{k}\_i^T
\end{align*}
$$
{{< /collapse >}}

### DeltaNet

{{< collapse summary="Show derivation of $S_N$" >}}
$$A_i = I - \beta_i \bm{k}_i \bm{k}_i^T \quad\quad B_i = \beta_i \bm{v}_i \bm{k}_i^T$$
Thus,
$$S_N = \sum\_{i=1}^{N} \left(\beta_i \bm{v}\_i \bm{k}\_i^T \prod\_{j=i+1}^{N} \left(I - \beta_j \bm{k}_j \bm{k}_j^T\right)\right)$$
{{< /collapse >}}

### Gated DeltaNet

{{< collapse summary="Show derivation of $S_N$" >}}
$$A_i = \alpha_i(I - \beta_i \bm{k}_i \bm{k}_i^T) \quad\quad B_i = \beta_i \bm{v}_i \bm{k}_i^T$$
Thus,
$$
\begin{align*}
    S_N &= \sum\_{i=1}^{N} \left(\beta_i \bm{v}\_i \bm{k}\_i^T \prod\_{j=i+1}^{N} \alpha_j \left(I - \beta_j \bm{k}_j \bm{k}_j^T\right)\right)\\\\
    S_N &= \sum\_{i=1}^{N} \left(\left(\beta_i \prod\_{j=i+1}^{N} \alpha_j \right) \bm{v}\_i \bm{k}\_i^T \prod\_{j=i+1}^{N} \left(I - \beta_j \bm{k}_j \bm{k}_j^T\right)\right)
\end{align*}
$$
{{< /collapse >}}

### RWKV-7

{{< collapse summary="Show derivation of $S_N$" >}}
$$A_i = \text{diag}(\bm{w}_i) - \bm{\hat{\kappa}}_i(\bm{a}_i \odot\bm{\hat{\kappa}}_i^T) \quad\quad B_i = \bm{v}_i \bm{k}_i^T$$
Thus,
$$S_N = \sum\_{i=1}^{N} \left(\bm{v}\_i \bm{k}\_i^T \prod\_{j=i+1}^{N} \left(\text{diag}(\bm{w}_j) - \bm{\hat{\kappa}}_j(\bm{a}_j \odot\bm{\hat{\kappa}}_j^T)\right)\right)$$
{{< /collapse >}}

Easy!

---

## Multi-Step Online Gradient Descent per Token

Now, what if we take $n_h$ gradient descent steps per token?

To do this, we can follow the procedure outlined in the DeltaProduct (Siems et al., 2025) paper where they: 

1. Recurrently generate $n_h$ key-value pairs for each input token,
2. Update the state using the $n_h$ key-value pairs, and
3. Keep only the final key-value pair and discard the rest.

In our formulation, this is equivalent to replacing each update with a product of $n_h$ updates:

$$
\begin{bmatrix}
    A_{i} & 0 \\\\
    B_{i} & I
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
    A_{i,1} & 0 \\\\
    B_{i,1} & I
\end{bmatrix}
\begin{bmatrix}
    A_{i,2} & 0 \\\\
    B_{i,2} & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_{i,n_h} & 0 \\\\
    B_{i,n_h} & I
\end{bmatrix}
$$
where $A_{i,j}$ and $B_{i,j}$ are the update matrices for the $j$-th gradient descent step for the $i$-th token.

Thus, Equation $(1)$ becomes:
$$
\begin{align}
S_N =
\begin{bmatrix}
    0 & I
\end{bmatrix}
\begin{bmatrix}
    A_{1,1} & 0 \\\\
    B_{1,1} & I
\end{bmatrix}
\begin{bmatrix}
    A_{1,2} & 0 \\\\
    B_{1,2} & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_{1,n_h} & 0 \\\\
    B_{1,n_h} & I
\end{bmatrix}
\begin{bmatrix}
    A_{2,1} & 0 \\\\
    B_{2,1} & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_{N, n_h} & 0 \\\\
    B_{N, n_h} & I
\end{bmatrix}
\begin{bmatrix}
    I \\\\
    0
\end{bmatrix}
\end{align}
$$

And if we reindex this as $[\cdot]\_k = [\cdot]\_{\lceil k/n\_h \rceil,\space (k-1) \\% n\_h + 1}$, then from equation $(3)$ above, we get:
$$
\begin{align}
S\_N = \sum\_{k=1}^{Nn\_h} \left( B_k \prod\_{k'=k+1}^{Nn\_h} A_{k'}\right)
\end{align}
$$

Alternatively, we can also combine the updates for each token into a single update matrix first before multiplying them together:

$$
\begin{align}
    \begin{bmatrix}
        A'\_{i} & 0 \\\\
        B'\_{i} & I
    \end{bmatrix}
    = \prod\_{j=1}^{n_h}
    \begin{bmatrix}
        A\_{i,j} & 0 \\\\
        B\_{i,j} & I
    \end{bmatrix}
    = \begin{bmatrix}
        \prod\_{j=1}^{n_h} A\_{i,j} & 0 \\\\
        \sum\_{j=1}^{n_h} \left(B\_{i,j} \prod\_{j'=j+1}^{n_h} A\_{i,j'}\right) & I
    \end{bmatrix}
\end{align}
$$

$$
\begin{align}
    S_N &=
        \begin{bmatrix}
        0 & I
        \end{bmatrix}
        \begin{bmatrix}
        A'\_1 & 0 \\\\
        B'\_1 & I
        \end{bmatrix}
        \begin{bmatrix}
        A'_2 & 0 \\\\
        B'_2 & I
        \end{bmatrix}
        \cdots
        \begin{bmatrix}
        A'_N & 0 \\\\
        B'_N & I
        \end{bmatrix}
        \begin{bmatrix}
        I \\\\
        0
        \end{bmatrix}\\\\
    S_N &=
        \begin{bmatrix}
        0 & I
        \end{bmatrix}
        \begin{bmatrix}
        \prod\_{i=1}^N A'\_i & 0 \\\\
        \sum\_{i=1}^N \left( B'\_i \prod\_{i'=i+1}^N A'\_{i'} \right) & I
        \end{bmatrix}
        \begin{bmatrix}
        I \\\\
        0
        \end{bmatrix}\\\\
    S_N &= \sum\_{i=1}^N \left( B'\_i \prod\_{i'=i+1}^N A'\_{i'} \right)\\\\
    S_N &= \sum\_{i=1}^N \sum\_{j=1}^{n_h} \left( B\_{i,j} \underline{\left(\prod\_{j'=j+1}^{n_h} A\_{i,j'}\right) \left(\prod\_{i'=i+1}^N \prod\_{j'=1}^{n_h} A\_{i',j'} \right)}\right)
\end{align}
$$

which, again, if we reindex this as $[\cdot]\_k = [\cdot]\_{\lceil k/n\_h \rceil,\space (k-1) \\% n\_h + 1}$, we get:

$$S\_N = \sum\_{k=1}^{Nn\_h} \left( B_k \prod\_{k'=k+1}^{Nn\_h} A_{k'}\right)$$
as expected.

---

Now, let's derive the $S_N$ for the linear attention mechanisms in the table above, but this time, with $n_h$ gradient descent steps per token.

### MambaSum*

{{< collapse summary="Show derivation of $S_N$" openByDefault=true >}}
$$A\_{i,j} = \text{diag}\left(\alpha\_{i,j} I\right) \quad\quad B\_{i,j} = \bm{v}\_{i,j} \bm{k}\_{i,j}^T$$
Thus, from Equation $(10)$ above,
$$
\begin{align*}
S_N &= \sum\_{i=1}^N \sum\_{j=1}^{n_h} \left( \bm{v}\_{i,j} \bm{k}\_{i,j}^T \left(\prod\_{j'=j+1}^{n_h} \text{diag}\left(\alpha\_{i,j'} I\right)\right) \left(\prod\_{i'=i+1}^N \prod\_{j'=1}^{n_h} \text{diag}\left(\alpha\_{i',j'} I\right) \right)\right)\\\\
S_N &= \sum\_{i=1}^N \sum\_{j=1}^{n_h} \left(\underline{\left( \prod\_{j'=j+1}^{n_h} \alpha\_{i,j'}\right) \left(\prod\_{i'=i+1}^N \prod\_{j'=1}^{n_h} \alpha\_{i',j'} \right)} \right) \bm{v}\_{i,j} \bm{k}\_{i,j}^T\\\\
S_N &= \sum\_{k=1}^{Nn\_h} \left(\prod\_{k'=k+1}^{Nn\_h} \alpha\_{k'}\right) \bm{v}\_k \bm{k}\_k^T
\end{align*}
$$
{{< /collapse >}}

> *I'm not actually sure if MambaSum already exists under a different name. If it does, please let me know!

### DeltaProduct

{{< collapse summary="Show derivation of $S_N$" >}}
$$A\_{i,j} = I - \beta\_{i,j} \bm{k}\_{i,j} \bm{k}\_{i,j}^T \quad\quad B\_{i,j} = \beta\_{i,j} \bm{v}\_{i,j} \bm{k}\_{i,j}^T$$
Thus,
$$
\begin{align*}
S_N &= \sum\_{i=1}^N \sum\_{j=1}^{n_h} \left( \beta\_{i,j} \bm{v}\_{i,j} \bm{k}\_{i,j}^T \underline{\left(\prod\_{j'=j+1}^{n_h} \left(I - \beta\_{i,j'} \bm{k}\_{i,j'} \bm{k}\_{i,j'}^T\right)\right) \left(\prod\_{i'=i+1}^N \prod\_{j'=1}^{n_h} \left(I - \beta\_{i',j'} \bm{k}\_{i',j'} \bm{k}\_{i',j'}^T\right) \right)}\right)\\\\
S_N &= \sum\_{k=1}^{Nn\_h} \left(\beta\_k \bm{v}\_k \bm{k}\_k^T \prod\_{k'=k+1}^{Nn\_h} \left(I - \beta\_{k'} \bm{k}\_{k'} \bm{k}\_{k'}^T\right)\right)
\end{align*}
$$
{{< /collapse >}}

### Gated DeltaProduct

{{< collapse summary="Show derivation of $S_N$" >}}
$$A\_{i,j} = \alpha\_{i,j}(I - \beta\_{i,j} \bm{k}\_{i,j} \bm{k}\_{i,j}^T) \quad\quad B\_{i,j} = \beta\_{i,j} \bm{v}\_{i,j} \bm{k}\_{i,j}^T$$
Thus,
$$
\begin{align*}
S_N &= \sum\_{i=1}^N \sum\_{j=1}^{n_h} \left( \beta\_{i,j} \bm{v}\_{i,j} \bm{k}\_{i,j}^T \underline{\left(\prod\_{j'=j+1}^{n_h} \alpha\_{i,j'} \left(I - \beta\_{i,j'} \bm{k}\_{i,j'} \bm{k}\_{i,j'}^T\right)\right) \left(\prod\_{i'=i+1}^N \prod\_{j'=1}^{n_h} \alpha\_{i',j'} \left(I - \beta\_{i',j'} \bm{k}\_{i',j'} \bm{k}\_{i',j'}^T\right) \right)}\right)\\\\
S_N &= \sum\_{k=1}^{Nn\_h} \left(\beta\_k \bm{v}\_k \bm{k}\_k^T \prod\_{k'=k+1}^{Nn\_h} \alpha\_{k'} \left(I - \beta\_{k'} \bm{k}\_{k'} \bm{k}\_{k'}^T\right)\right)\\\\
S_N &= \sum\_{k=1}^{Nn\_h} \left(\left( \beta\_k \prod\_{k'=k+1}^{Nn\_h} \alpha\_{k'} \right) \bm{v}\_k \bm{k}\_k^T \prod\_{k'=k+1}^{Nn\_h} \left(I - \beta\_{k'} \bm{k}\_{k'} \bm{k}\_{k'}^T\right)\right)
\end{align*}
$$
{{< /collapse >}}

### RWKV-7P

{{< collapse summary="Show derivation of $S_N$" >}}
$$A\_{i,j} = \text{diag}(\bm{w}\_{i,j}) - \bm{\hat{\kappa}}\_{i,j}(\bm{a}\_{i,j} \odot\bm{\hat{\kappa}}\_{i,j}^T) \quad\quad B\_{i,j} = \bm{v}\_{i,j} \bm{k}\_{i,j}^T$$
Thus,
$$
\begin{align*}
S_N &= \sum\_{i=1}^N \sum\_{j=1}^{n_h} \left( \bm{v}\_{i,j} \bm{k}\_{i,j}^T \underline{\left(\prod\_{j'=j+1}^{n_h} \left(\text{diag}(\bm{w}\_{i,j'}) - \bm{\hat{\kappa}}\_{i,j'}(\bm{a}\_{i,j'} \odot\bm{\hat{\kappa}}\_{i,j'}^T)\right)\right) \left(\prod\_{i'=i+1}^N \prod\_{j'=1}^{n_h} \left(\text{diag}(\bm{w}\_{i',j'}) - \bm{\hat{\kappa}}\_{i',j'}(\bm{a}\_{i',j'} \odot\bm{\hat{\kappa}}\_{i',j'}^T)\right) \right)}\right)\\\\
S_N &= \sum\_{k=1}^{Nn\_h} \left(\bm{v}\_k \bm{k}\_k^T \prod\_{k'=k+1}^{Nn\_h} \left(\text{diag}(\bm{w}\_k') - \bm{\hat{\kappa}}\_k'(\bm{a}\_k' \odot\bm{\hat{\kappa}}\_k'^T)\right)\right)
\end{align*}
$$
{{< /collapse >}}

---

## Chunk-wise Parallelism [Section Flagged for Review]

<div align="center">
    <img src="../test-time-regression/linear-attn-comp-forms.png" style="width:75%; height:75%;" />
</div>

Since the update operations of linear attention mechanisms we discussed above are associative--i.e., the order in which we "combine" the updates doesn't matter--we can perform the computations in multiple ways:
1. The **Fully Recurrent Form** where we update the state as we loop through the tokens/update matrices one by one,
2. The **Fully-Parallel Associative Scan Form** where we hierarchically combine the updates in a tree-like structure, and
3. The **Chunk-wise Parallel Form** (Hua et al., 2022; Sun et al., 2023) which is a compromise between the two where we divide the sequence into chunks first, combine intra-chunk updates in parallel, and then combine the chunk-level updates in a recurrent manner.

At inference time, the recurrent form works best*. But at training time, we have to be more hardware-aware to squeeze out as much performance as possible. We will discuss more about this in a separate post. But for now, there are two important things to keep in mind:
1. The **GPU Memory Hierarchy**. NVIDIA GPUs have a "global", high-bandwidth memory (HBM) that all threads in all processing units can access, and a smaller, shared memory (SRAM) that threads in the same processing unit can access. The shared memory, being more "local", has a much lower latency than the HBM. Thus, as much as possible, we want to limit communications between the processing units and the HBM and use the SRAM instead.
2. The **Tensor Cores**. Modern NVIDIA GPUs have tensor cores that can perform matrix multiplications much faster. Thus, ideally, we want to maximize the use of matrix multiplications and limit other operations.

Now, parallel associative scan might seem the best choice, and indeed it already suffices for some architectures like Mamba 1. However, it requires a lot more (shared) memory and communication between the processing units (and therefore materialization to the HBM). And it also doesn't fully utilize the tensor cores. But with chunk-wise parallelism, we only need to store the current state in the shared memory, and use matrix multiplications to compute the next chunk-level state. This way, we don't have to materialize the $S_N$s to the HBM at all, and we can fully utilize the tensor cores. Hence why most flash linear attention kernels use chunk-wise parallelism.

> *At inference time, we need to process the input tokens first before generating outputs. This is called the "pre-filling" stage. And chunk-wise parallelism works better here. After that, we can then use the recurrent form to generate the outputs.

---

A better way to think of chunk-wise parallelism is as multi-step online gradient descent, but instead of updating the state $n_h$ times per token, we update the state $n_c$ times per chunk where $n_c = N/C$ is the number of tokens per chunk and $C$ is the number of chunks. Thus, we just reuse our results from the previous section!

To make the connection more explicit, let's reindex Equation $(1)$ as $[\cdot]\_i = [\cdot]\_{\lceil i/n\_c \rceil,\space (i-1) \\% n\_c + 1}$:
$$
\begin{align*}
    S_N &=
        \begin{bmatrix}
        0 & I
        \end{bmatrix}
        \begin{bmatrix}
        A_{1} & 0 \\\\
        B_{1} & I
        \end{bmatrix}
        \begin{bmatrix}
        A_{2} & 0 \\\\
        B_{2} & I
        \end{bmatrix}
        \cdots
        \begin{bmatrix}
        A_{n_c} & 0 \\\\
        B_{n_c} & I
        \end{bmatrix}
        \begin{bmatrix}
        A_{n_c + 1} & 0 \\\\
        B_{n_c + 1} & I
        \end{bmatrix}
        \cdots
        \begin{bmatrix}
        A_{N} & 0 \\\\
        B_{N} & I
        \end{bmatrix}
        \begin{bmatrix}
        I \\\\
        0
        \end{bmatrix}\\\\
    S_N &=
        \begin{bmatrix}
        0 & I
        \end{bmatrix}
        \begin{bmatrix}
        A_{1,1} & 0 \\\\
        B_{1,1} & I
        \end{bmatrix}
        \begin{bmatrix}
        A_{1,2} & 0 \\\\
        B_{1,2} & I
        \end{bmatrix}
        \cdots
        \begin{bmatrix}
        A_{1,n_c} & 0 \\\\
        B_{1,n_c} & I
        \end{bmatrix}
        \begin{bmatrix}
        A_{2,1} & 0 \\\\
        B_{2,1} & I
        \end{bmatrix}
        \cdots
        \begin{bmatrix}
        A_{C, n_c} & 0 \\\\
        B_{C, n_c} & I
        \end{bmatrix}
        \begin{bmatrix}
        I \\\\
        0
        \end{bmatrix}\\\\
\end{align*}
$$
where $A_{c,i}$ and $B_{c,i}$ are now the update matrices for the $i$-th token within the $c$-th chunk.

And by combining the updates for each chunk as in Equation $(6)$ above, we get:
$$
\begin{align}
\begin{bmatrix}
    A^\*\_{c} & 0 \\\\
    B^\*\_{c} & I
\end{bmatrix}
= \prod\_{i=1}^{n_c} \begin{bmatrix}
    A\_{c,i} & 0 \\\\
    B\_{c,i} & I
\end{bmatrix}
= \begin{bmatrix}
    \prod\_{i=1}^{n_c} A\_{c,i} & 0 \\\\
    \sum\_{i=1}^{n_c} \left(B\_{c,i} \prod\_{i'=i+1}^{n_c} A\_{c,i'}\right) & I
\end{bmatrix}
\end{align}
$$
$$
S_C =
    \underline{
        \begin{bmatrix}
        0 & I
        \end{bmatrix}
        \begin{bmatrix}
        A^\*\_1 & 0 \\\\
        B^\*\_1 & I
        \end{bmatrix}
        \begin{bmatrix}
        A^\*_2 & 0 \\\\
        B^\*_2 & I
        \end{bmatrix}
        \cdots
        \begin{bmatrix}
        A^\*\_{C-1} & 0 \\\\
        B^\*\_{C-1} & I
        \end{bmatrix}
    }
    \begin{bmatrix}
    A^\*_C & 0 \\\\
    B^\*_C & I
    \end{bmatrix}
    \begin{bmatrix}
    I \\\\
    0
    \end{bmatrix}
$$
which has the equivalent cross-chunk recurrent form:
$$
\begin{align}
\begin{bmatrix}
    S\_{C} & I
\end{bmatrix} &=
\begin{bmatrix}
    S\_{C-1} & I
\end{bmatrix}
\begin{bmatrix}
    A^\*_C & 0 \\\\
    B^\*_C & I
\end{bmatrix}\\\\
S_C &= S\_{C-1}A^\*_C + B^\*_C
\end{align}
$$

---

Now, let's derive the $S_C$ for the linear attention mechanisms in the table above.

### Chunk-wise Mamba 2

{{< collapse summary="Show derivation of $S_C$" openByDefault=true >}}
$$
\begin{align*}
    A\_{c,i} &= \text{diag}\left(\alpha\_{c,i} I\right) & B\_{c,i} &= \bm{v}\_{c,i} \bm{k}\_{c,i}^T\\\\
    A^\*_C &= \prod\_{i=1}^{n_c} \text{diag}\left(\alpha\_{C,i} I\right) \quad & B^\*_C &= \sum\_{i=1}^{n_c} \left(\bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \text{diag}\left(\alpha\_{C,i'} I\right)\right)
\end{align*}
$$
Thus, from Equation $(13)$ above,
$$
\begin{align*}
    S_C &= S\_{C-1}A^\*_C + B^\*_C\\\\
    S_C &= S\_{C-1} \prod\_{i=1}^{n_c} \text{diag}\left(\alpha\_{C,i} I\right) + \sum\_{i=1}^{n_c} \left(\bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \text{diag}\left(\alpha\_{C,i'} I\right)\right)\\\\
    S_C &= S\_{C-1} \prod\_{i=1}^{n_c} \alpha\_{C,i} + \sum\_{i=1}^{n_c} \left(\prod\_{i'=i+1}^{n_c} \alpha\_{C,i'}\right) \bm{v}\_{C,i} \bm{k}\_{C,i}^T
\end{align*}
$$
{{< /collapse >}}

### Chunk-wise DeltaNet

{{< collapse summary="Show derivation of $S_C$" >}}
$$
\begin{align*}
    A\_{c,i} &= I - \beta\_{c,i} \bm{k}\_{c,i} \bm{k}\_{c,i}^T & B\_{c,i} &= \beta\_{c,i} \bm{v}\_{c,i} \bm{k}\_{c,i}^T\\\\
    A^\*_C &= \prod\_{i=1}^{n_c} \left(I - \beta\_{C,i} \bm{k}\_{C,i} \bm{k}\_{C,i}^T\right) \quad & B^\*_C &= \sum\_{i=1}^{n_c} \left(\beta\_{C,i} \bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \left(I - \beta\_{C,i'} \bm{k}\_{C,i'} \bm{k}\_{C,i'}^T\right)\right)
\end{align*}
$$
Thus,
$$
\begin{align*}
    S_C &= S\_{C-1}A^\*_C + B^\*_C\\\\
    S_C &= S\_{C-1} \prod\_{i=1}^{n_c} \left(I - \beta\_{C,i} \bm{k}\_{C,i} \bm{k}\_{C,i}^T\right) + \sum\_{i=1}^{n_c} \left(\beta\_{C,i} \bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \left(I - \beta\_{C,i'} \bm{k}\_{C,i'} \bm{k}\_{C,i'}^T\right)\right)
\end{align*}
$$
{{< /collapse >}}

### Chunk-wise Gated DeltaNet

{{< collapse summary="Show derivation of $S_C$" >}}
$$
\begin{align*}
    A\_{c,i} &= \alpha\_{c,i}(I - \beta\_{c,i} \bm{k}\_{c,i} \bm{k}\_{c,i}^T) & B\_{c,i} &= \beta\_{c,i} \bm{v}\_{c,i} \bm{k}\_{c,i}^T\\\\
    A^\*_C &= \prod\_{i=1}^{n_c} \alpha\_{C,i} \left(I - \beta\_{C,i} \bm{k}\_{C,i} \bm{k}\_{C,i}^T\right) \quad & B^\*_C &= \sum\_{i=1}^{n_c} \left(\beta\_{C,i} \bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \alpha\_{C,i'} \left(I - \beta\_{C,i'} \bm{k}\_{C,i'} \bm{k}\_{C,i'}^T\right)\right)
\end{align*}
$$
Thus,
$$
\begin{align*}
    S_C &= S\_{C-1}A^\*_C + B^\*_C\\\\
    S_C &= S\_{C-1} \prod\_{i=1}^{n_c} \alpha\_{C,i} \left(I - \beta\_{C,i} \bm{k}\_{C,i} \bm{k}\_{C,i}^T\right) + \sum\_{i=1}^{n_c} \left(\beta\_{C,i} \bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \alpha\_{C,i'} \left(I - \beta\_{C,i'} \bm{k}\_{C,i'} \bm{k}\_{C,i'}^T\right)\right)\\\\
    S_C &= S\_{C-1} \left(\prod\_{i=1}^{n_c} \alpha\_{C,i} \right) \left(\prod\_{i=1}^{n_c} \left(I - \beta\_{C,i} \bm{k}\_{C,i} \bm{k}\_{C,i}^T\right)\right) + \sum\_{i=1}^{n_c} \left(\left(\beta\_{C,i} \prod\_{i'=i+1}^{n_c} \alpha\_{C,i'} \right) \bm{v}\_{C,i} \bm{k}\_{C,i}^T  \prod\_{i'=i+1}^{n_c} \left(I - \beta\_{C,i'} \bm{k}\_{C,i'} \bm{k}\_{C,i'}^T\right)\right)
\end{align*}
$$
{{< /collapse >}}

### Chunk-wise RWKV-7

{{< collapse summary="Show derivation of $S_C$" >}}
$$
\begin{align*}
    A\_{c,i} &= \text{diag}\left(\bm{w}\_{c,i}\right) - \bm{\hat{\kappa}}\_{c,i}(\bm{a}\_{c,i} \odot\bm{\hat{\kappa}}\_{c,i}^T) & B\_{c,i} &= \bm{v}\_{c,i} \bm{k}\_{c,i}^T\\\\
    A^\*_C &= \prod\_{i=1}^{n_c} \left(\text{diag}\left(\bm{w}\_{C,i}\right) - \bm{\hat{\kappa}}\_{C,i}(\bm{a}\_{C,i} \odot\bm{\hat{\kappa}}\_{C,i}^T)\right) \quad & B^\*_C &= \sum\_{i=1}^{n_c} \left(\bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \left(\text{diag}\left(\bm{w}\_{C,i'}\right) - \bm{\hat{\kappa}}\_{C,i'}(\bm{a}\_{C,i'} \odot\bm{\hat{\kappa}}\_{C,i'}^T)\right)\right)
\end{align*}
$$
Thus,
$$
\begin{align*}
    S_C &= S\_{C-1}A^\*_C + B^\*_C\\\\
    S_C &= S\_{C-1} \prod\_{i=1}^{n_c} \left(\text{diag}\left(\bm{w}\_{C,i}\right) - \bm{\hat{\kappa}}\_{C,i}(\bm{a}\_{C,i} \odot\bm{\hat{\kappa}}\_{C,i}^T)\right) + \sum\_{i=1}^{n_c} \left(\bm{v}\_{C,i} \bm{k}\_{C,i}^T \prod\_{i'=i+1}^{n_c} \left(\text{diag}\left(\bm{w}\_{C,i'}\right) - \bm{\hat{\kappa}}\_{C,i'}(\bm{a}\_{C,i'} \odot\bm{\hat{\kappa}}\_{C,i'}^T)\right)\right)
\end{align*}
$$
{{< /collapse >}}

## Multi-Step Online Gradient Descent per Token with Chunk-wise Parallelism

Let's combine the two techniques we've discussed so far: multi-step online gradient descent per token and chunk-wise parallelism.

### The strategy

We can do this either way, but suppose we chunk the updates first then expand the each of the updates within the chunks into a product of $n_h$ updates. I.e., we have:

$$
\begin{bmatrix}
    A_{(c-1)*n_c + i} & 0 \\\\
    B_{(c-1)*n_c + i} & I
\end{bmatrix}
\xrightarrow{\text{reindex}}
\begin{bmatrix}
    A_{c,i} & 0 \\\\
    B_{c,i} & I
\end{bmatrix}
\xrightarrow{\text{expand}}
\begin{bmatrix}
    A_{c,i,1} & 0 \\\\
    B_{c,i,1} & I
\end{bmatrix}
\begin{bmatrix}
    A_{c,i,2} & 0 \\\\
    B_{c,i,2} & I
\end{bmatrix}
\cdots
\begin{bmatrix}
    A_{c,i,n_h} & 0 \\\\
    B_{c,i,n_h} & I
\end{bmatrix}
$$
where $A_{c,i,j}$ and $B_{c,i,j}$ are the update matrices for the $j$-th gradient descent step for the $i$-th token within the $c$-th chunk.

And from equations $(6)$, $(10)$, and $(11)$, we have:
$$
\begin{align*}
    \begin{bmatrix}
        A^\*\_{c} & 0 \\\\
        B^\*\_{c} & I
    \end{bmatrix}
    &= \prod\_{i=1}^{n_c} \begin{bmatrix}
        A'\_{c,i} & 0 \\\\
        B'\_{c,i} & I
    \end{bmatrix}
    = \prod\_{i=1}^{n_c} \prod\_{j=1}^{n_h} \begin{bmatrix}
        A\_{c,i,j} & 0 \\\\
        B\_{c,i,j} & I
    \end{bmatrix}\\\\
    \begin{bmatrix}
        A^\*\_{c} & 0 \\\\
        B^\*\_{c} & I
    \end{bmatrix}
    &= \begin{bmatrix}
        \prod\_{i=1}^{n_c} \prod\_{j=1}^{n_h} A\_{c,i,j} & 0 \\\\
        \sum\_{i=1}^{n_c}\sum\_{j=1}^{n_h} \left( B\_{c,i,j} \left(\prod\_{j'=j+1}^{n_h} A\_{c,i,j'}\right) \left(\prod\_{i'=i+1}^{n_c} \prod\_{j=1}^{n_h} A\_{c,i,j}\right)\right) & I
    \end{bmatrix}
\end{align*}
$$
Thus,
$$
\begin{align*}
    A^\*\_{c} &= \prod\_{i=1}^{n_c} \prod\_{j=1}^{n_h} A\_{c,i,j} \\\\
    B^\*\_{c} &= \sum\_{i=1}^{n_c}\sum\_{j=1}^{n_h} \left( B\_{c,i,j} \left(\prod\_{j'=j+1}^{n_h} A\_{c,i,j'}\right) \left(\prod\_{i'=i+1}^{n_c} \prod\_{j=1}^{n_h} A\_{c,i,j}\right)\right)
\end{align*}
$$
which we can then plug into Equation $(13)$ to get the cross-chunk recurrence:

$$
\begin{align*}
    S_C &= S\_{C-1}A^\*_C + B^\*_C\\\\
    S_C &= S\_{C-1} \prod\_{i=1}^{n_c} \prod\_{j=1}^{n_h} A\_{C,i,j} + \sum\_{i=1}^{n_c}\sum\_{j=1}^{n_h} \left( B\_{C,i,j} \left(\prod\_{j'=j+1}^{n_h} A\_{C,i,j'}\right) \left(\prod\_{i'=i+1}^{n_c} \prod\_{j=1}^{n_h} A\_{C,i,j}\right)\right)
\end{align*}
$$

or, if we reindex this as $[\cdot]\_{C,k} = [\cdot]\_{C,\space \lceil k/n\_h \rceil,\space (k-1) \\% n\_h + 1}$, we get:

$$
\begin{align*}
    S_C &= S\_{C-1} \prod\_{k=1}^{n_c n_h} A\_{C,k} + \sum\_{k=1}^{n_c n_h} \left( B\_{C,k} \prod\_{k'=k+1}^{n_c n_h} A\_{C,k'}\right)
\end{align*}
$$

---

As an exercise, try deriving the cross-chunk recurrence for MambaSum, DeltaProduct, Gated DeltaProduct, and RWKV-7P.

---

## Conclusion

And that's it!

Not only is the blocked matrix formulation of linear attention mechanisms intuitive, it also makes the connections between different algorithms and computational forms much more obvious. I'd even go as far as to say that we now have the proper abstraction to do an evolutionary search for new linear attention mechanisms ;)

---

In the next post, we'll talk about faster ways to calculate $A^\*\_{c}$ and $B^\*\_{c}$ for diagonal and diagonal-plus-low-rank $A^\*\_{c}$ using the WY Representations and the UT Transform. Stay tuned!

## Acknowledgements

Big thanks to Songlin Yang, Julien Siems, and @Smerky, @BeeGass, @safelix, and @jacobbuckman for their feedback and discussions!

## How to Cite

```bibtex
@misc{cesista2025blockmatlinearattn,
  author = {Franz Louis Cesista},
  title = {Blocked Matrix Formulation of Linear Attention Mechanisms},
  year = {2025},
  month = {March},
  day = {16},
  url = {https://leloykun.github.io/ponder/blockmat-linear-attn/},
}
```

## References

1. Riccardo Grazzi, Julien Siems, Jörg K.H. Franke, Arber Zela, Frank Hutter, Massimiliano Pontil (2025). Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. URL https://arxiv.org/abs/2411.12537
2. Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, Riccardo Grazzi (2025). DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders. URL https://arxiv.org/abs/2502.10297
3. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 2020b. URL http://proceedings.mlr.press/v119/katharopoulos20a.html.
4. Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Proceedings of the 41st International Conference on MachineLearning, volume 235 of Proceedingsof Machine Learning Research, pp. 10041–10071. PMLR, 2024b. URL https://proceedings.mlr.press/v235/dao24a.html.
5. Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim (2025). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. URL https://arxiv.org/abs/2406.06484
6. Songlin Yang, Jan Kautz, Ali Hatamizadeh (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. URL https://arxiv.org/abs/2412.06464
7. Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V. Le. Transformer quality in linear time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 9099–9117. PMLR, 2022b. URL https://proceedings.mlr.press/v162/hua22a.html.
8. Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. ArXiv preprint, abs/2307.08621, 2023. URL https://arxiv.org/abs/2307.08621.
9. Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Haowen Hou, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, Christian Zhou-Zheng (2025). RWKV-7 "Goose" with Expressive Dynamic State Evolution. URL https://arxiv.org/abs/2503.14456
