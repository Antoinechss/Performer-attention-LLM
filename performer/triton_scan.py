"""
Triton CUDA kernel for the FAVOR+ causal sequential scan.

Replaces the Python `for i in range(N)` loop in PerformerAttentionCore.forward()
with a single fused GPU kernel launch, keeping the running S[M,D] and z[M] state
in on-chip registers instead of bouncing through global memory on every step.

Why this is faster than the Python loop:
- Python loop: ~512 kernel launches at N=512, each with ~5µs CUDA launch overhead
- Triton kernel: 1 kernel launch, all N steps run inside the GPU without Python involvement

Requirements:
- pip install triton    (already a PyTorch dependency on CUDA builds)
- CUDA GPU             (no native macOS/MPS support)

Usage is automatic: PerformerAttentionCore.forward() dispatches here when
   device.type == "cuda"  AND  triton is installed.
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _favor_scan_kernel(
        phi_q_ptr, phi_k_ptr, v_ptr, out_ptr,
        N,
        # Strides for phi_q / phi_k tensors: shape [B, H, N, M]
        stride_qk_b, stride_qk_h, stride_qk_n,
        # Strides for v / out tensors: shape [B, H, N, D]
        stride_v_b, stride_v_h, stride_v_n,
        H:       tl.constexpr,
        BLOCK_M: tl.constexpr,  # num random features M (must be power of 2)
        BLOCK_D: tl.constexpr,  # head_dim D (must be power of 2)
    ):
        """
        One GPU program handles one (batch, head) pair.
        Grid = (B * H,).

        Maintains S[M, D] and z[M] in registers (no global memory per step).
        Streams N tokens sequentially inside the GPU — no Python loop overhead.
        """
        bh = tl.program_id(0)
        b  = bh // H
        h  = bh  % H

        # Base pointers for this (b, h) slice
        base_qk = b * stride_qk_b + h * stride_qk_h
        base_v  = b * stride_v_b  + h * stride_v_h

        m_idx = tl.arange(0, BLOCK_M)  # [M]
        d_idx = tl.arange(0, BLOCK_D)  # [D]

        # Running state — lives in registers, never written to global memory
        S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)  # kv accumulator
        z = tl.zeros((BLOCK_M,),         dtype=tl.float32)  # k  accumulator

        for n in range(N):
            # Load phi_k[b, h, n, :] and v[b, h, n, :]
            phi_k_n = tl.load(phi_k_ptr + base_qk + n * stride_qk_n + m_idx)  # [M]
            v_n     = tl.load(v_ptr     + base_v  + n * stride_v_n  + d_idx)  # [D]

            # Causal update: include token n in the state before querying it
            S = S + phi_k_n[:, None] * v_n[None, :]  # [M, D]  outer product
            z = z + phi_k_n                           # [M]

            # Load phi_q[b, h, n, :] and compute output
            phi_q_n = tl.load(phi_q_ptr + base_qk + n * stride_qk_n + m_idx)  # [M]
            num   = tl.sum(phi_q_n[:, None] * S, axis=0)   # [D]   phi_q @ S
            denom = tl.sum(phi_q_n * z) + 1e-6              # scalar
            tl.store(out_ptr + base_v + n * stride_v_n + d_idx, num / denom)


def triton_scan_forward(phi_q: torch.Tensor, phi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    FAVOR+ causal sequential scan — single fused GPU kernel.

    Args:
        phi_q : [B, H, N, M] float32, contiguous — random-feature queries
        phi_k : [B, H, N, M] float32, contiguous — random-feature keys
        v     : [B, H, N, D] float32, contiguous — values

    Returns:
        out   : [B, H, N, D] float32
    """
    assert _TRITON_AVAILABLE, (
        "triton package is not installed.\n"
        "Install it with:  pip install triton\n"
        "Note: Triton requires a CUDA GPU — not available on macOS."
    )
    assert phi_q.device.type == "cuda", "Triton kernel requires a CUDA device"

    # Ensure contiguous layout for pointer arithmetic
    phi_q = phi_q.contiguous()
    phi_k = phi_k.contiguous()
    v     = v.contiguous()

    B, H, N, M = phi_q.shape
    D = v.shape[-1]

    out = torch.empty(B, H, N, D, dtype=phi_q.dtype, device=phi_q.device)

    grid = (B * H,)
    _favor_scan_kernel[grid](
        phi_q, phi_k, v, out,
        N,
        phi_q.stride(0), phi_q.stride(1), phi_q.stride(2),
        v.stride(0),     v.stride(1),     v.stride(2),
        H=H,
        BLOCK_M=M,
        BLOCK_D=D,
    )
    return out
