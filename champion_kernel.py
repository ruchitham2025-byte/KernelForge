import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_fused_kernel(
    A_ptr,          # (M, N) float16
    B_ptr,          # (M, N) float16
    b1_ptr,         # (N,) float16
    b2_ptr,         # (N,) float16
    out_ptr,        # (M, N) float16
    M,              # int
    N,              # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    # 2‑D coordinates
    row = offsets // N
    col = offsets % N

    # Load inputs (guard with mask)
    a = tl.load(A_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Load biases (same for all rows, depend only on column)
    bias1 = tl.load(b1_ptr + col, mask=mask).to(tl.float32)
    bias2 = tl.load(b2_ptr + col, mask=mask).to(tl.float32)

    # Add bias
    a_bias = a + bias1
    b_bias = b + bias2

    # SiLU: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-b_bias))
    silu = b_bias * sig

    # Final multiply
    out = a_bias * silu

    # Store result
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


def triton_swiglu(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                  w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    Computes: out = (x @ w1.T + b1) * silu(x @ w2.T + b2)
    where silu(t) = t * sigmoid(t).

    All inputs are float16 CUDA tensors with shapes:
        x: (4096, 1024)
        w1, w2: (2048, 1024)
        b1, b2: (2048,)
    Output: (4096, 2048) float16.
    """
    assert x.is_cuda and w1.is_cuda and w2.is_cuda
    assert b1.is_cuda and b2.is_cuda
    assert x.dtype == w1.dtype == w2.dtype == b1.dtype == b2.dtype == torch.float16

    M, K = x.shape
    N, K2 = w1.shape
    assert K == K2, "Incompatible feature dimension"
    assert w2.shape == (N, K), "w2 must have same shape as w1"
    assert b1.shape == (N,) and b2.shape == (N,)

    # Single GEMM for both branches: stack weights and multiply once
    W = torch.cat([w1, w2], dim=0)          # (2N, K)
    Y = torch.mm(x, W.t())                  # (M, 2N)

    # Split the result
    A_raw = Y[:, :N].contiguous()           # (M, N)
    B_raw = Y[:, N:].contiguous()           # (M, N)

    out = torch.empty_like(A_raw)           # (M, N) float16

    total_elements = M * N
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)

    _swiglu_fused_kernel[grid](
        A_raw, B_raw, b1, b2, out,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
