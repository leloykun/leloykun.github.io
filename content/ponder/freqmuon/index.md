---
title: "Frequency Domain Muon for Convolutional Neural Networks: Simplified"
date: 2026-03-06
tags: ["Machine Learning", "Optimizers", "Muon", "Convolutional Neural Networks", "FFT"]
author: ["Franz Louis Cesista"]
description: "The 'correct' way to apply Muon to CNNs"
summary: "The 'correct' way to apply Muon to CNNs"
editPost:
    URL: "https://github.com/leloykun/ortho-conv-cifar10"
    Text: "Github Repository"
---

## 1. Frequency Domain Muon

The Muon optimizer only makes sense when applied to linear operators (e.g. MLP weights) ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/)). Convolutions in Convolutional Neural Networks (CNNs) are not linear operators in the pixel space, but they become one after a 'coordinate change' to the frequency domain via the Fast Fourier Transform (FFT). As such, it is there where we should apply Muon's orthogonalization logic. This builds on top of [Ji-Ha Kim's recent work on FreqMuon](https://jiha-kim.github.io/posts/frequency-domain-muon-for-conv-filters/).

The core algorithm goes as follows:
1. Compute $G := \nabla f(W_{\text{CNN}})$ in pixel space via backpropagation.
2. FFT to the frequency domain: $\widehat{G} := \text{FFT}(G)$.
3. Apply Muon's orthogonalization to $\widehat{G}$, treating each frequency bin as a separate linear operator: $\widehat{U} := \texttt{msign}(\widehat{G})$.
4. Inverse FFT back to the spatial domain: $U := \text{FFT}^{-1}(\widehat{U})$.
5. Update CNN weights with $U$: $W_{\text{CNN}} \leftarrow W_{\text{CNN}} - \eta U$.

### 1.1. Sample implementation

```python
@dataclass
class FreqMuonCfg:
    fft_size: int = 8
    ns_steps: int = 2
    eps: float = 1e-7

def _zeropower_ns5_complex(
    X: torch.Tensor,  # [B, m, n], complex64
    *,
    steps: int,
    eps: float,
) -> torch.Tensor:
    """
    Muon-style NS5 polynomial on native complex tensors.
    Returns approx polar factor / orthogonalized update.
    """
    assert X.ndim == 3 and X.is_complex()
    _, m, n = X.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    norm = torch.linalg.matrix_norm(X, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X / norm
    if transpose := m > n:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.mH      # [B, r, r]
        X = a * X + (b * A + c * A @ A) @ X
    if transpose:
        X = X.transpose(-2, -1)
    return X

def _freq_muon_conv_update_batched(g32: torch.Tensor, cfg: FreqMuonCfg) -> torch.Tensor:
    """
    g32: [P, Cout, Cin, kH, kW] float32
    returns: [P, Cout, Cin, kH, kW] float32
    """
    assert g32.ndim == 5 and g32.dtype == torch.float32
    P, Cout, Cin, kH, kW = g32.shape
    M = cfg.fft_size

    if kH > M or kW > M:
        raise ValueError(f"Kernel {kH}x{kW} > fft_size {M}. Increase --fft_size.")

    # Full FFT on an MxM logical grid; PyTorch pads/trims automatically.
    Khat = torch.fft.fft2(g32, s=(M, M), dim=(-2, -1), norm="ortho")  # [P, Cout, Cin, M, M]

    # Apply Muon orthogonalization to each Cout x Cin matrix in the frequency domain
    Kflat = Khat.permute(0, 3, 4, 1, 2).reshape(P * M * M, Cout, Cin)
    Kflat = _zeropower_ns5_complex(Kflat, steps=cfg.ns_steps, eps=cfg.eps)
    Khat2 = Kflat.reshape(P, M, M, Cout, Cin).permute(0, 3, 4, 1, 2)

    # Back to spatial domain, then crop to original kernel support
    upd_pad = torch.fft.ifft2(Khat2, s=(M, M), dim=(-2, -1), norm="ortho")
    upd = upd_pad.real[:, :, :, :kH, :kW]
    return upd.to(dtype=g32.dtype)

# _zeropower_ns5_complex = torch.compile(_zeropower_ns5_complex, mode="max-autotune")
_freq_muon_conv_update_batched = torch.compile(_freq_muon_conv_update_batched, mode="max-autotune")
```

## 2. Results

| Method                            | Validation Accuracy (%) |
| --------------------------------- | ----------------------: |
| Baseline                          |                   91.44 |
| FreqMuon (fft_size=8, ns_steps=2) |                   **93.51** |

We evaluate FreqMuon on the CIFAR-10 Airbench benchmark, training a highly-optimized CNN from scratch for 7 epochs. We find that FreqMuon beats the SOTA baseline optimizer by a large margin, achieving 93.51% validation accuracy vs the baseline's 91.44%.

> Note: these results are preliminary and we have not yet performed a hyperparameter sweep for FreqMuon.

## How to cite

```bibtex
@misc{cesista2026freqmuon,
  author = {Franz Louis Cesista},
  title = {Frequency Domain Muon for Convolutional Neural Networks: Simplified},
  year = {2026},
  url = {https://github.com/leloykun/freqmuon/},
}
```

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein (2024). Muon: An optimizer for hidden layers in neural networks. Available at: https://kellerjordan.github.io/posts/muon/
2. Ji-Ha Kim (2026). Frequency-Domain Muon for Conv Filters - Orthogonalizing the Operator. URL https://jiha-kim.github.io/posts/frequency-domain-muon-for-conv-filters/
