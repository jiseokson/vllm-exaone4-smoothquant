# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark EXAONE4 SmoothQuant reference and fused input paths.

This benchmark isolates the only operation changed by
``Exaone4ForCausalLMSmoothQuantTritonFused``:

* reference: ``x * smooth_scale`` followed by dynamic per-token INT8 quant
* fused: Triton smooth-scale + dynamic per-token INT8 quant

It can measure the quantization paths alone or include the unchanged CUTLASS
W8A8 GEMM used by the QKV and gate/up projections. All A/B pairs share the
same activation, scale, quantized weight, and weight scale tensors.

Examples:
    .venv/bin/python benchmarks/kernels/benchmark_exaone4_smoothquant.py \
        --model-config ./exaone4_w8a8_smoothquant \
        --tokens 1 8 32 128 512 2048 \
        --mode both

    .venv/bin/python benchmarks/kernels/benchmark_exaone4_smoothquant.py \
        --hidden-size 2048 --intermediate-size 4096 \
        --num-attention-heads 32 --num-key-value-heads 8 --head-dim 64 \
        --tp-size 1 --scope quant --mode eager
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.ops.smooth_quant import smooth_scale_quant_int8
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_INTERMEDIATE_SIZE = 4096
DEFAULT_NUM_ATTENTION_HEADS = 32
DEFAULT_NUM_KEY_VALUE_HEADS = 8


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    tp_size: int
    qkv_output_size: int
    gate_up_output_size: int
    dtype_name: str


@dataclass(frozen=True)
class TimingStats:
    median_us: float
    p20_us: float
    p80_us: float


TensorFn = Callable[[], object]


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.").lower()
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {name!r}; use bfloat16 or float16."
        ) from exc


def _load_hf_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != "exaone4":
        raise ValueError(
            f"Expected an EXAONE4 config, got model_type={config.get('model_type')!r}"
        )
    return config


def _resolve_model_shape(args: argparse.Namespace) -> ModelShape:
    config = _load_hf_config(args.model_config)

    def resolve(arg_name: str, config_name: str, default: Any) -> Any:
        value = getattr(args, arg_name)
        return value if value is not None else config.get(config_name, default)

    hidden_size = int(resolve("hidden_size", "hidden_size", DEFAULT_HIDDEN_SIZE))
    intermediate_size = int(
        resolve("intermediate_size", "intermediate_size", DEFAULT_INTERMEDIATE_SIZE)
    )
    num_attention_heads = int(
        resolve(
            "num_attention_heads",
            "num_attention_heads",
            DEFAULT_NUM_ATTENTION_HEADS,
        )
    )
    num_key_value_heads = int(
        resolve(
            "num_key_value_heads",
            "num_key_value_heads",
            num_attention_heads,
        )
    )
    head_dim = int(resolve("head_dim", "head_dim", hidden_size // num_attention_heads))
    dtype_name = str(resolve("dtype", "dtype", config.get("torch_dtype", "bfloat16")))
    _dtype_from_name(dtype_name)

    tp_size = args.tp_size
    if tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    if num_attention_heads % tp_size != 0:
        raise ValueError("num_attention_heads must be divisible by --tp-size")
    if num_key_value_heads >= tp_size:
        if num_key_value_heads % tp_size != 0:
            raise ValueError(
                "num_key_value_heads must be divisible by --tp-size when "
                "num_key_value_heads >= tp_size"
            )
    elif tp_size % num_key_value_heads != 0:
        raise ValueError(
            "--tp-size must be divisible by num_key_value_heads when KV heads "
            "are replicated"
        )
    if intermediate_size % tp_size != 0:
        raise ValueError("intermediate_size must be divisible by --tp-size")

    local_q_heads = num_attention_heads // tp_size
    local_kv_heads = max(1, num_key_value_heads // tp_size)
    qkv_output_size = (local_q_heads + 2 * local_kv_heads) * head_dim
    gate_up_output_size = 2 * (intermediate_size // tp_size)

    for name, size in (
        ("hidden_size", hidden_size),
        ("qkv_output_size", qkv_output_size),
        ("gate_up_output_size", gate_up_output_size),
    ):
        if size % 16 != 0:
            raise ValueError(
                f"{name}={size} is not divisible by 16, as required by the "
                "CUTLASS W8A8 path"
            )

    return ModelShape(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        tp_size=tp_size,
        qkv_output_size=qkv_output_size,
        gate_up_output_size=gate_up_output_size,
        dtype_name=dtype_name,
    )


def _relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference_f32 = reference.float()
    actual_f32 = actual.float()
    denominator = torch.linalg.vector_norm(reference_f32).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(reference_f32 - actual_f32) / denominator)


def _check_quant_outputs(
    reference_fn: TensorFn,
    fused_fn: TensorFn,
    *,
    max_q_diff: int,
    max_rel_l2: float,
) -> dict[str, float | int]:
    ref_q, ref_scale = reference_fn()  # type: ignore[misc]
    fused_q, fused_scale = fused_fn()  # type: ignore[misc]
    torch.cuda.synchronize()

    q_abs_max = int((ref_q.int() - fused_q.int()).abs().max())
    scale_rel_l2 = _relative_l2(ref_scale, fused_scale)
    ref_dequant = ref_q.float() * ref_scale
    fused_dequant = fused_q.float() * fused_scale
    dequant_rel_l2 = _relative_l2(ref_dequant, fused_dequant)

    if q_abs_max > max_q_diff or dequant_rel_l2 > max_rel_l2:
        raise RuntimeError(
            "Quant correctness check failed: "
            f"max_q_diff={q_abs_max} (limit {max_q_diff}), "
            f"dequant_rel_l2={dequant_rel_l2:.6g} (limit {max_rel_l2})"
        )
    return {
        "max_q_diff": q_abs_max,
        "scale_rel_l2": scale_rel_l2,
        "dequant_rel_l2": dequant_rel_l2,
    }


def _check_linear_outputs(
    reference_fn: TensorFn,
    fused_fn: TensorFn,
    *,
    max_rel_l2: float,
) -> dict[str, float]:
    reference = reference_fn()  # type: ignore[assignment]
    fused = fused_fn()  # type: ignore[assignment]
    torch.cuda.synchronize()
    output_rel_l2 = _relative_l2(reference, fused)
    max_abs_diff = float((reference.float() - fused.float()).abs().max())
    if output_rel_l2 > max_rel_l2:
        raise RuntimeError(
            "Linear correctness check failed: "
            f"output_rel_l2={output_rel_l2:.6g} (limit {max_rel_l2})"
        )
    return {
        "output_rel_l2": output_rel_l2,
        "max_abs_diff": max_abs_diff,
    }


def _capture_cuda_graph(fn: TensorFn, device: torch.device) -> TensorFn:
    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)

    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = fn()

    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    def replay() -> object:
        graph.replay()
        return static_output

    return replay


def _time_once(fn: TensorFn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _summarize(samples_us: list[float]) -> TimingStats:
    samples = torch.tensor(samples_us, dtype=torch.float64)
    quantiles = torch.quantile(
        samples, torch.tensor([0.5, 0.2, 0.8], dtype=samples.dtype)
    )
    return TimingStats(
        median_us=float(quantiles[0]),
        p20_us=float(quantiles[1]),
        p80_us=float(quantiles[2]),
    )


def _benchmark_pair(
    reference_fn: TensorFn,
    fused_fn: TensorFn,
    *,
    device: torch.device,
    mode: str,
    warmup_iters: int,
    bench_iters: int,
    samples: int,
) -> tuple[TimingStats, TimingStats]:
    if mode == "cudagraph":
        reference_fn = _capture_cuda_graph(reference_fn, device)
        fused_fn = _capture_cuda_graph(fused_fn, device)

    # Compile Triton, initialize CUTLASS, and populate allocator pools.
    for iteration in range(warmup_iters):
        if iteration % 2 == 0:
            reference_fn()
            fused_fn()
        else:
            fused_fn()
            reference_fn()
    torch.cuda.synchronize(device)

    timings: dict[str, list[float]] = {"reference": [], "fused": []}
    functions = {"reference": reference_fn, "fused": fused_fn}
    for sample_index in range(samples):
        # Alternating the order gives an A-B-B-A pattern across two samples.
        order = (
            ("reference", "fused") if sample_index % 2 == 0 else ("fused", "reference")
        )
        for name in order:
            timings[name].append(_time_once(functions[name], bench_iters))

    return _summarize(timings["reference"]), _summarize(timings["fused"])


def _make_quant_functions(
    x: torch.Tensor, smooth_scale_f32: torch.Tensor
) -> tuple[TensorFn, TensorFn]:
    # ActivationScaleLayer is created in the model dtype. The fused model moves
    # the same values to an FP32 buffer before passing them to Triton.
    reference_scale = smooth_scale_f32.to(x.dtype)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        x_scaled = x * reference_scale
        x_q, x_scale, _ = ops.scaled_int8_quant(
            x_scaled.contiguous(), scale=None, azp=None, symmetric=True
        )
        return x_q, x_scale

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        return smooth_scale_quant_int8(x, smooth_scale_f32)

    return reference, fused


def _make_linear_functions(
    x: torch.Tensor,
    smooth_scale_f32: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[TensorFn, TensorFn]:
    reference_quant, fused_quant = _make_quant_functions(x, smooth_scale_f32)

    def reference() -> torch.Tensor:
        x_q, x_scale = reference_quant()  # type: ignore[misc]
        return ops.cutlass_scaled_mm(
            x_q,
            weight_q,
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=x.dtype,
        )

    def fused() -> torch.Tensor:
        x_q, x_scale = fused_quant()  # type: ignore[misc]
        return ops.cutlass_scaled_mm(
            x_q,
            weight_q,
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=x.dtype,
        )

    return reference, fused


def _make_activations(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)
    smooth_scale = torch.empty(
        hidden_size, device=device, dtype=torch.float32
    ).uniform_(0.5, 1.5)
    return x, smooth_scale


def _make_weight(
    hidden_size: int, output_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # CutlassInt8ScaledMMLinearKernel transposes [N, K] after loading. Keep the
    # resulting column-major [K, N] view instead of making it contiguous.
    weight_q = torch.randint(
        -127,
        128,
        (output_size, hidden_size),
        device=device,
        dtype=torch.int8,
    ).t()
    weight_scale = torch.empty(
        (1, output_size), device=device, dtype=torch.float32
    ).uniform_(1e-4, 1e-2)
    return weight_q, weight_scale


def _format_error(correctness: dict[str, float | int]) -> str:
    if "dequant_rel_l2" in correctness:
        return f"{correctness['dequant_rel_l2']:.2e}"
    if "output_rel_l2" in correctness:
        return f"{correctness['output_rel_l2']:.2e}"
    return "-"


def _print_header() -> None:
    print(
        f"{'scope':<9} {'M':>6} {'K':>6} {'N':>7} {'mode':<10} "
        f"{'ref us':>10} {'fused us':>10} {'speedup':>9} {'saved us':>10} "
        f"{'rel-L2':>10}"
    )
    print("-" * 100)


def _record_case(
    results: list[dict[str, Any]],
    *,
    scope: str,
    num_tokens: int,
    hidden_size: int,
    output_size: int | None,
    mode: str,
    reference_fn: TensorFn,
    fused_fn: TensorFn,
    correctness: dict[str, float | int],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    reference, fused = _benchmark_pair(
        reference_fn,
        fused_fn,
        device=device,
        mode=mode,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        samples=args.samples,
    )
    speedup = reference.median_us / fused.median_us
    saved_us = reference.median_us - fused.median_us
    output_size_text = "-" if output_size is None else str(output_size)
    print(
        f"{scope:<9} {num_tokens:>6} {hidden_size:>6} {output_size_text:>7} "
        f"{mode:<10} {reference.median_us:>10.3f} {fused.median_us:>10.3f} "
        f"{speedup:>8.3f}x {saved_us:>10.3f} "
        f"{_format_error(correctness):>10}"
    )
    results.append(
        {
            "scope": scope,
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "mode": mode,
            "reference_us": asdict(reference),
            "fused_us": asdict(fused),
            "speedup": speedup,
            "saved_us": saved_us,
            "correctness": correctness,
        }
    )


def _correctness_or_empty(
    args: argparse.Namespace,
    check: Callable[[], dict[str, float | int]],
) -> dict[str, float | int]:
    return {} if args.skip_correctness else check()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU and vLLM CUDA kernels.")
    if args.warmup_iters < 1 or args.bench_iters < 1 or args.samples < 1:
        raise ValueError("warmup-iters, bench-iters, and samples must be positive")
    if any(num_tokens <= 0 for num_tokens in args.tokens):
        raise ValueError("Every value passed to --tokens must be positive")

    shape = _resolve_model_shape(args)
    dtype = _dtype_from_name(shape.dtype_name)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    modes = ["eager", "cudagraph"] if args.mode == "both" else [args.mode]
    tokens = list(dict.fromkeys(args.tokens))
    results: list[dict[str, Any]] = []

    print(f"GPU: {torch.cuda.get_device_name(device)} ({device})")
    print(
        f"dtype={dtype}, TP={shape.tp_size}, K={shape.hidden_size}, "
        f"qkv N={shape.qkv_output_size}, gate-up N={shape.gate_up_output_size}"
    )
    print(
        f"tokens={tokens}, warmup={args.warmup_iters}, "
        f"iterations/sample={args.bench_iters}, samples={args.samples}"
    )
    _print_header()

    if args.scope in ("quant", "both"):
        for num_tokens in tokens:
            x, smooth_scale = _make_activations(
                num_tokens, shape.hidden_size, dtype, device
            )
            reference_fn, fused_fn = _make_quant_functions(x, smooth_scale)
            correctness = _correctness_or_empty(
                args,
                lambda reference_fn=reference_fn, fused_fn=fused_fn: (
                    _check_quant_outputs(
                        reference_fn,
                        fused_fn,
                        max_q_diff=args.max_q_diff,
                        max_rel_l2=args.max_rel_l2,
                    )
                ),
            )
            for mode in modes:
                _record_case(
                    results,
                    scope="quant",
                    num_tokens=num_tokens,
                    hidden_size=shape.hidden_size,
                    output_size=None,
                    mode=mode,
                    reference_fn=reference_fn,
                    fused_fn=fused_fn,
                    correctness=correctness,
                    args=args,
                    device=device,
                )

    if args.scope in ("linear", "both"):
        projection_sizes = {
            "qkv": shape.qkv_output_size,
            "gate-up": shape.gate_up_output_size,
        }
        for projection in args.projections:
            output_size = projection_sizes[projection]
            weight_q, weight_scale = _make_weight(
                shape.hidden_size, output_size, device
            )
            for num_tokens in tokens:
                x, smooth_scale = _make_activations(
                    num_tokens, shape.hidden_size, dtype, device
                )
                reference_fn, fused_fn = _make_linear_functions(
                    x, smooth_scale, weight_q, weight_scale
                )
                correctness = _correctness_or_empty(
                    args,
                    lambda reference_fn=reference_fn, fused_fn=fused_fn: (
                        _check_linear_outputs(
                            reference_fn,
                            fused_fn,
                            max_rel_l2=args.max_rel_l2,
                        )
                    ),
                )
                for mode in modes:
                    _record_case(
                        results,
                        scope=projection,
                        num_tokens=num_tokens,
                        hidden_size=shape.hidden_size,
                        output_size=output_size,
                        mode=mode,
                        reference_fn=reference_fn,
                        fused_fn=fused_fn,
                        correctness=correctness,
                        args=args,
                        device=device,
                    )
            del weight_q, weight_scale
            torch.cuda.empty_cache()

    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "model_shape": asdict(shape),
        "settings": {
            "tokens": tokens,
            "scope": args.scope,
            "projections": args.projections,
            "mode": args.mode,
            "warmup_iters": args.warmup_iters,
            "bench_iters": args.bench_iters,
            "samples": args.samples,
            "seed": args.seed,
        },
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2)
            output_file.write("\n")
        print(f"\nWrote JSON report to {args.output_json}")
    return report


def parse_args() -> argparse.Namespace:
    parser = FlexibleArgumentParser(
        description="Benchmark EXAONE4 SmoothQuant reference and fused kernels."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="EXAONE4 config.json or a model directory containing it.",
    )
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-key-value-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "bf16", "float16", "half", "fp16"],
        help="Overrides the dtype in --model-config.",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048],
        help="Flattened token counts (M dimension) to benchmark.",
    )
    parser.add_argument("--scope", choices=["quant", "linear", "both"], default="both")
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=["qkv", "gate-up"],
        default=["qkv", "gate-up"],
        help="Linear envelopes to benchmark when --scope includes linear.",
    )
    parser.add_argument(
        "--mode", choices=["eager", "cudagraph", "both"], default="both"
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-q-diff", type=int, default=2)
    parser.add_argument(
        "--max-rel-l2",
        type=float,
        default=2e-2,
        help="Maximum relative L2 error allowed by correctness checks.",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
