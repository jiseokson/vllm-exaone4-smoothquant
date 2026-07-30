# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused SmoothQuant input scaling + per-token INT8 quantization kernel.

Replaces the two-kernel sequence:
  1. x_scaled = x * smooth_scale          (IdentityWithParam)
  2. x_q, x_s = scaled_int8_quant(x_scaled)

with a single Triton kernel that reads x once, applies the per-channel
smooth scale, computes the per-token absmax, and writes INT8 output.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.platforms import current_platform


@triton.jit
def _smooth_scale_quant_int8_kernel(
    x_ptr,       # [M, K] fp16/bf16 input
    scale_ptr,   # [K]    fp32 smooth scale (IdentityWithParam.weight)
    xq_ptr,      # [M, K] int8 output
    xs_ptr,      # [M]    fp32 per-token output scale
    stride_x,    # row stride of x (== K for contiguous)
    K,
    BLOCK_K: tl.constexpr,
):
    """One Triton program handles one token (row)."""
    row_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    # Load one row of input
    x = tl.load(
        x_ptr + row_id * stride_x + offs, mask=mask, other=0.0
    ).to(tl.float32)

    # Apply per-channel smooth scale (fits in registers / L1)
    s = tl.load(scale_ptr + offs, mask=mask, other=1.0)
    x_scaled = x * s

    # Per-token absmax -> token scale
    abs_max = tl.max(tl.abs(x_scaled))
    abs_max = tl.maximum(abs_max, 1e-10)
    token_scale = abs_max / 127.0

    # Quantize to int8
    x_q = tl.clamp(
        tl.extra.cuda.libdevice.round(x_scaled / token_scale), -128.0, 127.0
    ).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_x + offs, x_q, mask=mask)
    tl.store(xs_ptr + row_id, token_scale)


def smooth_scale_quant_int8(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused smooth-scale application + dynamic per-token INT8 quantization.

    Args:
        x:            Input tensor [..., K], fp16 or bf16.
        smooth_scale: Per-channel scale [K], fp32.

    Returns:
        x_q: Quantized tensor [..., K], int8.
        x_s: Per-token scale  [..., 1], fp32.  Shape matches cutlass expectation.
    """
    assert current_platform.is_cuda(), \
        "smooth_scale_quant_int8 requires CUDA"

    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1]).contiguous()
    M, K = x_2d.shape

    x_q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    x_s_flat = torch.empty(M, dtype=torch.float32, device=x.device)

    BLOCK_K = triton.next_power_of_2(K)
    num_warps = min(max(BLOCK_K // 256, 1), 8)

    _smooth_scale_quant_int8_kernel[(M,)](
        x_2d,
        smooth_scale.to(torch.float32),
        x_q,
        x_s_flat,
        stride_x=x_2d.stride(0),
        K=K,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=1,
    )

    # Restore batch dims and add trailing dim for cutlass (expects [M, 1])
    x_q = x_q.view(orig_shape)
    x_s = x_s_flat.view(*orig_shape[:-1], 1)
    return x_q, x_s
