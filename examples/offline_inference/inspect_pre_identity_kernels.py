# SPDX-License-Identifier: Apache-2.0
"""Inspect the kernels used by an EXAONE pre-identity W8A8 checkpoint.

This is a diagnostic script, not a benchmark. It deliberately uses eager,
single-process execution so that Python-level instrumentation can observe the
quantization and scaled-matmul calls made during one short generation.

Example:
    .venv/bin/python examples/offline_inference/inspect_pre_identity_kernels.py \
        --model ./sq_int8_pre_identity
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

# Keep the model worker in this process. This must be set before importing vLLM.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Explain what quantization is.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from vllm import LLM, SamplingParams, _custom_ops
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.ops import smooth_quant
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_int8 as w8a8_module,
    )

    selected: list[dict[str, Any]] = []
    calls: Counter[str] = Counter()

    original_create_weights = w8a8_module.CompressedTensorsW8A8Int8.create_weights
    original_cutlass_apply = CutlassInt8ScaledMMLinearKernel.apply_weights
    original_triton_apply = TritonInt8ScaledMMLinearKernel.apply_weights
    original_smooth_quant = smooth_quant.smooth_scale_quant_int8
    original_cutlass_mm = _custom_ops.cutlass_scaled_mm

    def traced_create_weights(self, layer, *create_args, **create_kwargs):
        result = original_create_weights(self, layer, *create_args, **create_kwargs)
        selected.append(
            {
                "layer": layer,
                "kernel": type(self.kernel).__name__,
                "logical_widths": tuple(layer.logical_widths),
            }
        )
        return result

    def traced_cutlass_apply(self, layer, x, bias=None):
        calls["CutlassInt8ScaledMMLinearKernel.apply_weights"] += 1
        branch = "smooth" if getattr(layer, "smooth_scale", None) is not None else "plain"
        calls[f"cutlass_branch.{branch}"] += 1
        return original_cutlass_apply(self, layer, x, bias)

    def traced_triton_apply(self, layer, x, bias=None):
        calls["TritonInt8ScaledMMLinearKernel.apply_weights"] += 1
        branch = "smooth" if getattr(layer, "smooth_scale", None) is not None else "plain"
        calls[f"triton_branch.{branch}"] += 1
        return original_triton_apply(self, layer, x, bias)

    def traced_smooth_quant(x, scale):
        calls["smooth_scale_quant_int8"] += 1
        calls[f"smooth_shape.x={tuple(x.shape)},scale={tuple(scale.shape)}"] += 1
        return original_smooth_quant(x, scale)

    def traced_cutlass_mm(*mm_args, **mm_kwargs):
        calls["ops.cutlass_scaled_mm"] += 1
        return original_cutlass_mm(*mm_args, **mm_kwargs)

    # Patch before constructing LLM: create_weights runs during model creation.
    w8a8_module.CompressedTensorsW8A8Int8.create_weights = traced_create_weights
    CutlassInt8ScaledMMLinearKernel.apply_weights = traced_cutlass_apply
    TritonInt8ScaledMMLinearKernel.apply_weights = traced_triton_apply
    smooth_quant.smooth_scale_quant_int8 = traced_smooth_quant
    _custom_ops.cutlass_scaled_mm = traced_cutlass_mm

    print("[1/3] Loading model in eager, single-process mode...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    kernel_counts = Counter(item["kernel"] for item in selected)
    smooth_counts = Counter(
        item["kernel"]
        for item in selected
        if getattr(item["layer"], "smooth_scale", None) is not None
    )

    print("\n[2/3] Kernels selected while creating W8A8 linear layers")
    for kernel, count in sorted(kernel_counts.items()):
        print(f"  {kernel}: {count} layers")
    print("  Layers carrying smooth_scale after checkpoint loading:")
    for kernel, count in sorted(smooth_counts.items()):
        print(f"    {kernel}: {count} layers")

    print("\n  Representative layers")
    for index, item in enumerate(selected[:8]):
        layer = item["layer"]
        scale = getattr(layer, "smooth_scale", None)
        scale_desc = "none" if scale is None else f"{tuple(scale.shape)} {scale.dtype}"
        print(
            f"    #{index}: kernel={item['kernel']}, "
            f"logical_widths={item['logical_widths']}, smooth_scale={scale_desc}"
        )

    # Exclude any model-load warm-up from the actual inference counts.
    calls.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\n[3/3] Running one prefill (max_tokens=1)...")
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0, max_tokens=1),
        use_tqdm=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for name, count in sorted(calls.items()):
        print(f"  {name}: {count}")
    print(f"\n  Generated token: {outputs[0].outputs[0].text!r}")

    print("\nInterpretation")
    print("  cutlass_branch.smooth == smooth_scale_quant_int8 means the")
    print("  pre-identity scale + activation quantization fused path ran.")
    print("  cutlass_branch.plain covers o_proj/down_proj and other W8A8 linears.")
    if calls["triton_branch.smooth"]:
        print("  WARNING: TritonInt8ScaledMMLinearKernel saw smooth_scale, but its")
        print("  current override does not apply that scale.")


if __name__ == "__main__":
    main()
