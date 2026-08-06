# SPDX-License-Identifier: Apache-2.0
"""Compare EXAONE4 SmoothQuant reference and fused implementations.

The public entry point coordinates two isolated worker subprocesses. Each
worker loads one architecture, runs the same one-token prefill, and records
module boundaries plus the activation quantizer outputs. The coordinator then
compares the traces and emits a human-readable summary and an optional JSON
report.

Example:
    .venv/bin/python \
        examples/offline_inference/compare_exaone_smoothquant.py \
        --model ./exaone4_w8a8_smoothquant \
        --prompt "Explain quantization briefly." \
        --output /tmp/exaone-smoothquant-report.json

This is a correctness diagnostic, not a benchmark. Tensor capture copies GPU
outputs to CPU and therefore introduces synchronization overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


REFERENCE_ARCHITECTURE = "Exaone4ForCausalLMSmoothQuantReference"
TARGET_ARCHITECTURE = "Exaone4ForCausalLMSmoothQuantTritonFused"
LAYER_RE = re.compile(r"model\.layers\.(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare intermediate tensors from the EXAONE4 SmoothQuant "
            "reference and fused architectures."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Explain quantization briefly.",
        help="Prompt tokenized by the reference worker.",
    )
    parser.add_argument(
        "--reference-architecture",
        default=REFERENCE_ARCHITECTURE,
    )
    parser.add_argument(
        "--target-architecture",
        default=TARGET_ARCHITECTURE,
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--activation-rtol", type=float, default=1e-2)
    parser.add_argument("--activation-atol", type=float, default=1e-2)
    parser.add_argument("--scale-rtol", type=float, default=1e-6)
    parser.add_argument("--scale-atol", type=float, default=1e-7)
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every captured boundary instead of the concise summary.",
    )
    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="Keep the raw reference.pt and target.pt trace files.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Parent directory for retained traces; implies --keep-traces.",
    )

    # Internal worker arguments. They intentionally remain undocumented in the
    # public help output.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--role",
        choices=("reference", "target"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--architecture", help=argparse.SUPPRESS)
    parser.add_argument("--trace-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--token-ids-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _iter_tensors(value: Any, suffix: str = "") -> Iterator[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield suffix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{suffix}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{suffix}.{key}")


class TensorRecorder:
    def __init__(self) -> None:
        self.tensors: dict[str, list[Any]] = {}

    def record(self, name: str, value: Any) -> None:
        for suffix, tensor in _iter_tensors(value):
            key = f"{name}{suffix}"
            self.tensors.setdefault(key, []).append(
                tensor.detach().contiguous().cpu().clone()
            )


def _first_tensor(value: Any) -> Any | None:
    return next((tensor for _, tensor in _iter_tensors(value)), None)


def _linear_semantic_name(module_name: str | None) -> str | None:
    if module_name is None:
        return None
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.(self_attn\.qkv_proj|mlp\.gate_up_proj)",
        module_name,
    )
    if match is None:
        return None
    layer_index, operation = match.groups()
    branch = "attention" if operation == "self_attn.qkv_proj" else "mlp"
    return f"layer.{layer_index}.{branch}"


def _module_semantic_name(module_name: str) -> str | None:
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", module_name)
    if match is None:
        return None

    layer_index, suffix = match.groups()
    suffix_mapping = {
        "self_attn.qkv_proj": "attention.qkv_output",
        "self_attn.q_norm": "attention.q_norm",
        "self_attn.k_norm": "attention.k_norm",
        "self_attn.rotary_emb": "attention.rope",
        "self_attn.attn": "attention.attn_output",
        "self_attn.o_proj": "attention.o_proj_output",
        "post_attention_layernorm": "attention.post_norm",
        "mlp.gate_up_proj": "mlp.gate_up_output",
        "mlp.act_fn": "mlp.activation_output",
        "mlp.down_proj": "mlp.down_proj_output",
        "post_feedforward_layernorm": "mlp.post_norm",
    }
    semantic_suffix = suffix_mapping.get(suffix)
    if semantic_suffix is None:
        return None
    return f"layer.{layer_index}.{semantic_suffix}"


def _load_worker_model(llm: Any) -> Any:
    try:
        return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError(
            "Unable to access the in-process model. Ensure "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 and tensor parallel size 1."
        ) from exc


def _selected_kernel_counts(model: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        kernel = getattr(quant_method, "kernel", None)
        if kernel is not None:
            counts[type(kernel).__name__] += 1
    return counts


def _validate_smooth_scale_attachments(model: Any) -> list[str]:
    import torch

    errors: list[str] = []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        smooth_scale = getattr(module, "smooth_scale", None)
        if smooth_scale is None:
            continue

        if name.endswith(".self_attn.qkv_proj"):
            scaler_name = name.removesuffix(".self_attn.qkv_proj") + ".attn_scaler"
        elif name.endswith(".mlp.gate_up_proj"):
            scaler_name = name.removesuffix(".mlp.gate_up_proj") + ".ffn_scaler"
        else:
            errors.append(f"Unexpected smooth_scale owner: {name}")
            continue

        scaler = modules.get(scaler_name)
        if scaler is None or not hasattr(scaler, "weight"):
            errors.append(f"Missing scaler {scaler_name} for {name}")
            continue
        if not torch.equal(smooth_scale.float(), scaler.weight.float()):
            errors.append(f"Scale mismatch between {name} and {scaler_name}")
    return errors


def _record_smooth_scales(model: Any, recorder: TensorRecorder) -> None:
    modules = dict(model.named_modules())
    for name, layer in modules.items():
        match = LAYER_RE.fullmatch(name)
        if match is None:
            continue
        layer_index = match.group(1)
        attention_scale = getattr(
            layer.self_attn.qkv_proj, "smooth_scale", layer.attn_scaler.weight
        )
        mlp_scale = getattr(
            layer.mlp.gate_up_proj, "smooth_scale", layer.ffn_scaler.weight
        )
        recorder.record(f"layer.{layer_index}.attention.smooth_scale", attention_scale)
        recorder.record(f"layer.{layer_index}.mlp.smooth_scale", mlp_scale)


def _register_module_hooks(model: Any, recorder: TensorRecorder) -> list[Any]:
    handles: list[Any] = []
    active_decoder_layers: list[str] = []

    def decoder_pre_hook(layer_index: str) -> Callable[..., None]:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            # Decoder args are (positions, hidden_states, residual).
            active_decoder_layers.append(layer_index)
            recorder.record(f"layer.{layer_index}.input", args[1])

        return hook

    def decoder_output_hook(layer_index: str) -> Callable[..., None]:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, tuple):
                recorder.record(f"layer.{layer_index}.output", output[0])
                if len(output) > 1 and output[1] is not None:
                    recorder.record(
                        f"layer.{layer_index}.residual_before_mlp", output[1]
                    )
            else:
                recorder.record(f"layer.{layer_index}.output", output)
            active_decoder_layers.pop()

        return hook

    def canonical_input_hook(semantic: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            value = args[0]
            smooth_scale = getattr(module, "smooth_scale", None)
            if smooth_scale is not None:
                value = value * smooth_scale
            recorder.record(f"{semantic}.canonical_input", value)

        return hook

    def rope_output_hook(
        _module: Any, _args: tuple[Any, ...], output: Any
    ) -> None:
        layer_index = active_decoder_layers[-1]
        recorder.record(f"layer.{layer_index}.attention.rope_q", output[0])
        recorder.record(f"layer.{layer_index}.attention.rope_k", output[1])

    def output_hook(semantic: str) -> Callable[..., None]:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                recorder.record(semantic, tensor)

        return hook

    for name, module in model.named_modules():
        layer_match = LAYER_RE.fullmatch(name)
        if layer_match is not None:
            layer_index = layer_match.group(1)
            handles.append(
                module.register_forward_pre_hook(decoder_pre_hook(layer_index))
            )
            handles.append(
                module.register_forward_hook(decoder_output_hook(layer_index))
            )
            continue

        semantic = _module_semantic_name(name)
        if semantic is None:
            continue
        if name.endswith(".self_attn.rotary_emb"):
            handles.append(module.register_forward_hook(rope_output_hook))
            continue
        if name.endswith((".self_attn.qkv_proj", ".mlp.gate_up_proj")):
            branch = _linear_semantic_name(name)
            assert branch is not None
            handles.append(
                module.register_forward_pre_hook(canonical_input_hook(branch))
            )
        handles.append(module.register_forward_hook(output_hook(semantic)))

    modules = dict(model.named_modules())
    if final_norm := modules.get("model.norm"):
        handles.append(
            final_norm.register_forward_hook(output_hook("model.final_norm"))
        )
    if logits_processor := modules.get("logits_processor"):
        handles.append(
            logits_processor.register_forward_hook(output_hook("model.logits"))
        )
    return handles


def _install_kernel_tracing(
    model: Any,
    recorder: TensorRecorder,
    kernel_call_counts: Counter[str],
    quantizer_call_counts: Counter[str],
) -> Callable[[], None]:
    from vllm import _custom_ops
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.ops import smooth_quant

    names_by_id = {id(module): name for name, module in model.named_modules()}
    active_layers: list[str | None] = []

    original_cutlass_apply = CutlassInt8ScaledMMLinearKernel.apply_weights
    original_triton_apply = TritonInt8ScaledMMLinearKernel.apply_weights
    original_scaled_quant = _custom_ops.scaled_int8_quant
    original_smooth_quant = smooth_quant.smooth_scale_quant_int8

    def wrap_apply(
        kernel_name: str,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        def traced(self: Any, layer: Any, x: Any, bias: Any = None) -> Any:
            module_name = names_by_id.get(id(layer))
            active_layers.append(module_name)
            kernel_call_counts[kernel_name] += 1
            try:
                return original(self, layer, x, bias)
            finally:
                active_layers.pop()

        return traced

    def record_quantizer_output(kind: str, output: Any) -> None:
        quantizer_call_counts[kind] += 1
        module_name = active_layers[-1] if active_layers else None
        semantic = _linear_semantic_name(module_name)
        if semantic is None:
            return
        recorder.record(f"{semantic}.quant.x_q", output[0])
        recorder.record(f"{semantic}.quant.x_s", output[1])
        if len(output) > 2 and output[2] is not None:
            recorder.record(f"{semantic}.quant.x_zp", output[2])

    def traced_scaled_quant(*args: Any, **kwargs: Any) -> Any:
        output = original_scaled_quant(*args, **kwargs)
        record_quantizer_output("scaled_int8_quant", output)
        return output

    def traced_smooth_quant(*args: Any, **kwargs: Any) -> Any:
        output = original_smooth_quant(*args, **kwargs)
        record_quantizer_output("smooth_scale_quant_int8", output)
        return output

    CutlassInt8ScaledMMLinearKernel.apply_weights = wrap_apply(
        "CutlassInt8ScaledMMLinearKernel", original_cutlass_apply
    )
    TritonInt8ScaledMMLinearKernel.apply_weights = wrap_apply(
        "TritonInt8ScaledMMLinearKernel", original_triton_apply
    )
    _custom_ops.scaled_int8_quant = traced_scaled_quant
    smooth_quant.smooth_scale_quant_int8 = traced_smooth_quant

    def restore() -> None:
        CutlassInt8ScaledMMLinearKernel.apply_weights = original_cutlass_apply
        TritonInt8ScaledMMLinearKernel.apply_weights = original_triton_apply
        _custom_ops.scaled_int8_quant = original_scaled_quant
        smooth_quant.smooth_scale_quant_int8 = original_smooth_quant

    return restore


def run_worker(args: argparse.Namespace) -> None:
    # These variables must be set before vLLM is imported.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    import torch

    from vllm import LLM, SamplingParams

    if args.architecture is None or args.trace_output is None or args.role is None:
        raise ValueError(
            "Worker mode requires --role, --architecture, and --trace-output"
        )

    llm = LLM(
        model=args.model,
        hf_overrides={"architectures": [args.architecture]},
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    model = _load_worker_model(llm)
    recorder = TensorRecorder()
    _record_smooth_scales(model, recorder)
    selected_kernels = _selected_kernel_counts(model)
    attachment_errors = _validate_smooth_scale_attachments(model)
    smooth_linear_count = sum(
        getattr(module, "smooth_scale", None) is not None for module in model.modules()
    )

    handles = _register_module_hooks(model, recorder)
    kernel_call_counts: Counter[str] = Counter()
    quantizer_call_counts: Counter[str] = Counter()
    restore_tracing = _install_kernel_tracing(
        model,
        recorder,
        kernel_call_counts,
        quantizer_call_counts,
    )

    if args.token_ids_file is None:
        prompts: list[Any] = [args.prompt]
    else:
        with args.token_ids_file.open(encoding="utf-8") as file:
            token_ids = json.load(file)
        prompts = [{"prompt_token_ids": token_ids}]

    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                max_tokens=1,
                ignore_eos=True,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
    finally:
        restore_tracing()
        for handle in handles:
            handle.remove()

    request_output = outputs[0]
    prompt_token_ids = request_output.prompt_token_ids
    if prompt_token_ids is None:
        raise RuntimeError("vLLM did not return prompt_token_ids")

    metadata = {
        "role": args.role,
        "requested_architecture": args.architecture,
        "loaded_model_class": type(model).__name__,
        "prompt": args.prompt if args.token_ids_file is None else None,
        "prompt_token_ids": list(prompt_token_ids),
        "selected_token_ids": list(request_output.outputs[0].token_ids),
        "selected_kernel_counts": dict(
            selected_kernels or kernel_call_counts
        ),
        "kernel_call_counts": dict(kernel_call_counts),
        "quantizer_call_counts": dict(quantizer_call_counts),
        "smooth_linear_count": smooth_linear_count,
        "smooth_scale_attachment_errors": attachment_errors,
        "torch_version": str(torch.__version__),
        "vllm_version": str(__import__("vllm").__version__),
        "cuda_device": torch.cuda.get_device_name(),
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "tensors": recorder.tensors}, args.trace_output)
    print(f"Saved {args.role} trace to {args.trace_output}")

    # Each worker exits immediately after writing its trace. Explicit cleanup
    # avoids leaving the single-rank NCCL process group to interpreter teardown.
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _trace_sort_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"layer\.(\d+)\.(.+)", value)
    if match is None:
        return (sys.maxsize, 0, value)

    layer_index = int(match.group(1))
    suffix = match.group(2)
    stages = (
        "attention.smooth_scale",
        "input",
        "attention.canonical_input",
        "attention.quant.x_q",
        "attention.quant.x_s",
        "attention.qkv_output",
        "attention.q_norm",
        "attention.k_norm",
        "attention.rope_q",
        "attention.rope_k",
        "attention.attn_output",
        "attention.o_proj_output",
        "attention.post_norm",
        "residual_before_mlp",
        "mlp.smooth_scale",
        "mlp.canonical_input",
        "mlp.quant.x_q",
        "mlp.quant.x_s",
        "mlp.gate_up_output",
        "mlp.activation_output",
        "mlp.down_proj_output",
        "mlp.post_norm",
        "output",
    )
    stage = next(
        (i for i, prefix in enumerate(stages) if suffix.startswith(prefix)),
        99,
    )
    return (layer_index, stage, value)


def _finite_or_none(value: float) -> float | None:
    import math

    return value if math.isfinite(value) else None


def _compare_tensor(
    name: str,
    call_index: int,
    reference: Any,
    target: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "name": name,
        "call_index": call_index,
        "reference_shape": list(reference.shape),
        "target_shape": list(target.shape),
        "reference_dtype": str(reference.dtype),
        "target_dtype": str(target.dtype),
        "dtype_equal": reference.dtype == target.dtype,
    }
    if reference.shape != target.shape:
        result["status"] = "SHAPE_MISMATCH"
        return result

    exact = torch.equal(reference, target)
    result["exact"] = exact
    if not (reference.is_floating_point() or target.is_floating_point()):
        mismatch_count = int(torch.count_nonzero(reference != target).item())
        result.update(
            {
                "mismatch_count": mismatch_count,
                "mismatch_ratio": mismatch_count / max(reference.numel(), 1),
                "status": "EXACT" if exact else "FAIL",
            }
        )
        return result

    reference_float = reference.float()
    target_float = target.float()
    difference = reference_float - target_float
    abs_difference = difference.abs()
    reference_norm = torch.linalg.vector_norm(reference_float)
    difference_norm = torch.linalg.vector_norm(difference)
    target_norm = torch.linalg.vector_norm(target_float)
    denominator = float((reference_norm * target_norm).item())
    cosine = (
        float(torch.sum(reference_float * target_float).item()) / denominator
        if denominator > 0
        else float("nan")
    )

    is_scale = name.endswith(".quant.x_s")
    rtol = args.scale_rtol if is_scale else args.activation_rtol
    atol = args.scale_atol if is_scale else args.activation_atol
    close = torch.isclose(
        reference_float,
        target_float,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    mismatch_count = int(torch.count_nonzero(~close).item())
    result.update(
        {
            "max_abs": float(abs_difference.max().item()),
            "mean_abs": float(abs_difference.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
            "relative_l2": _finite_or_none(
                float((difference_norm / reference_norm.clamp_min(1e-12)).item())
            ),
            "cosine_similarity": _finite_or_none(cosine),
            "reference_nan_count": int(torch.isnan(reference_float).sum().item()),
            "target_nan_count": int(torch.isnan(target_float).sum().item()),
            "reference_inf_count": int(torch.isinf(reference_float).sum().item()),
            "target_inf_count": int(torch.isinf(target_float).sum().item()),
            "mismatch_count": mismatch_count,
            "mismatch_ratio": mismatch_count / max(reference.numel(), 1),
            "rtol": rtol,
            "atol": atol,
            "status": "EXACT" if exact else ("PASS" if mismatch_count == 0 else "FAIL"),
        }
    )
    return result


def compare_traces(
    reference_trace: dict[str, Any],
    target_trace: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference_tensors = reference_trace["tensors"]
    target_tensors = target_trace["tensors"]
    comparisons: list[dict[str, Any]] = []

    all_names = sorted(
        set(reference_tensors) | set(target_tensors), key=_trace_sort_key
    )
    for name in all_names:
        if name not in reference_tensors or name not in target_tensors:
            comparisons.append(
                {
                    "name": name,
                    "call_index": None,
                    "status": "MISSING_REFERENCE"
                    if name not in reference_tensors
                    else "MISSING_TARGET",
                }
            )
            continue

        reference_calls = reference_tensors[name]
        target_calls = target_tensors[name]
        if len(reference_calls) != len(target_calls):
            comparisons.append(
                {
                    "name": name,
                    "call_index": None,
                    "reference_call_count": len(reference_calls),
                    "target_call_count": len(target_calls),
                    "status": "CALL_COUNT_MISMATCH",
                }
            )
            continue
        comparisons.extend(
            _compare_tensor(name, index, reference, target, args)
            for index, (reference, target) in enumerate(
                zip(reference_calls, target_calls, strict=True)
            )
        )

    reference_metadata = reference_trace["metadata"]
    target_metadata = target_trace["metadata"]
    validation_errors: list[str] = []
    if (
        reference_metadata["loaded_model_class"]
        != reference_metadata["requested_architecture"]
    ):
        validation_errors.append(
            "Reference worker loaded an unexpected model class"
        )
    if (
        target_metadata["loaded_model_class"]
        != target_metadata["requested_architecture"]
    ):
        validation_errors.append("Target worker loaded an unexpected model class")
    if reference_metadata["prompt_token_ids"] != target_metadata["prompt_token_ids"]:
        validation_errors.append("Reference and target prompt token IDs differ")
    selected_tokens_equal = (
        reference_metadata["selected_token_ids"]
        == target_metadata["selected_token_ids"]
    )
    if target_metadata["smooth_scale_attachment_errors"]:
        validation_errors.extend(target_metadata["smooth_scale_attachment_errors"])

    smooth_linear_count = target_metadata["smooth_linear_count"]
    smooth_call_count = target_metadata["quantizer_call_counts"].get(
        "smooth_scale_quant_int8", 0
    )
    if smooth_linear_count and smooth_call_count != smooth_linear_count:
        validation_errors.append(
            "Target fused quantizer call count does not match smooth linear count: "
            f"{smooth_call_count} != {smooth_linear_count}"
        )

    failed = [item for item in comparisons if item["status"] == "FAIL"]
    structural_failures = [
        item
        for item in comparisons
        if item["status"]
        in {
            "MISSING_REFERENCE",
            "MISSING_TARGET",
            "SHAPE_MISMATCH",
            "CALL_COUNT_MISMATCH",
        }
    ]
    if validation_errors or structural_failures:
        verdict = "INVALID_TRACE"
    elif failed or not selected_tokens_equal:
        verdict = "FAIL"
    elif any(item["status"] == "PASS" for item in comparisons):
        verdict = "PASS_WITH_DRIFT"
    else:
        verdict = "PASS"

    first_divergence = next(
        (
            item
            for item in comparisons
            if item["status"]
            not in {
                "EXACT",
                "PASS",
            }
        ),
        None,
    )
    logits_summary = None
    if "model.logits" in reference_tensors and "model.logits" in target_tensors:
        reference_logits = reference_tensors["model.logits"][-1]
        target_logits = target_tensors["model.logits"][-1]
        if reference_logits.shape == target_logits.shape:
            vocab_size = reference_logits.shape[-1]
            topk = min(5, vocab_size)
            reference_topk = (
                reference_logits.reshape(-1, vocab_size)[-1]
                .topk(topk)
                .indices.tolist()
            )
            target_topk = (
                target_logits.reshape(-1, vocab_size)[-1]
                .topk(topk)
                .indices.tolist()
            )
            logits_summary = {
                "reference_topk": reference_topk,
                "target_topk": target_topk,
                "top1_equal": reference_topk[0] == target_topk[0],
                "topk_overlap": len(set(reference_topk) & set(target_topk)),
                "topk": topk,
            }

    return {
        "verdict": verdict,
        "validation_errors": validation_errors,
        "first_divergence": first_divergence,
        "reference_metadata": reference_metadata,
        "target_metadata": target_metadata,
        "selected_tokens_equal": selected_tokens_equal,
        "logits_summary": logits_summary,
        "comparisons": comparisons,
    }


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{value:.3e}"


def _format_ratio(value: Any) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def print_report(report: dict[str, Any], show_all: bool = False) -> None:
    print("\nEXAONE4 SmoothQuant numerical comparison")
    print(f"  Verdict: {report['verdict']}")
    print(
        "  Reference: "
        f"{report['reference_metadata']['loaded_model_class']}"
    )
    print(f"  Target:    {report['target_metadata']['loaded_model_class']}")
    print(
        "  Target kernels: "
        f"{report['target_metadata']['selected_kernel_counts']}"
    )
    print(
        "  Target quantizers: "
        f"{report['target_metadata']['quantizer_call_counts']}"
    )

    if report["validation_errors"]:
        print("\nTrace validation errors")
        for error in report["validation_errors"]:
            print(f"  - {error}")

    comparisons = report["comparisons"]
    if show_all:
        displayed = comparisons
    else:
        first_name = (report["first_divergence"] or {}).get("name", "")
        first_layer_match = re.match(r"(layer\.\d+)\.", first_name)
        first_layer = first_layer_match.group(1) if first_layer_match else None
        displayed = [
            comparison
            for comparison in comparisons
            if (
                first_layer is not None
                and comparison["name"].startswith(first_layer)
            )
            or re.fullmatch(r"layer\.\d+\.output", comparison["name"])
            or comparison["name"].startswith("model.")
            or comparison["status"].startswith("MISSING_")
            or comparison["status"].endswith("MISMATCH")
        ]

    print("\nBoundary comparison")
    print(
        f"{'boundary':64} {'max_abs':>10} {'rel_l2':>10} "
        f"{'mismatch':>10} {'status':>12}"
    )
    for comparison in displayed:
        call_index = comparison.get("call_index")
        label = comparison["name"]
        if call_index is not None:
            label += f"[{call_index}]"
        print(
            f"{label[:64]:64} "
            f"{_format_metric(comparison.get('max_abs')):>10} "
            f"{_format_metric(comparison.get('relative_l2')):>10} "
            f"{_format_ratio(comparison.get('mismatch_ratio')):>10} "
            f"{comparison['status']:>12}"
        )

    omitted = len(comparisons) - len(displayed)
    if omitted:
        print(f"  ... {omitted} boundaries omitted; use --show-all or --output.")

    first_divergence = report["first_divergence"]
    if first_divergence is None:
        print("\nFirst divergence: none")
    else:
        print(
            "\nFirst divergence: "
            f"{first_divergence['name']}[{first_divergence.get('call_index')}] "
            f"({first_divergence['status']})"
        )
    reference_tokens = report["reference_metadata"]["selected_token_ids"]
    target_tokens = report["target_metadata"]["selected_token_ids"]
    print(
        "Selected token IDs: "
        f"reference={reference_tokens}, target={target_tokens}, "
        f"equal={reference_tokens == target_tokens}"
    )
    if logits_summary := report["logits_summary"]:
        print(
            "Logits top-k: "
            f"reference={logits_summary['reference_topk']}, "
            f"target={logits_summary['target_topk']}, "
            f"overlap={logits_summary['topk_overlap']}/{logits_summary['topk']}"
        )


def _worker_command(
    args: argparse.Namespace,
    role: str,
    architecture: str,
    trace_output: Path,
    token_ids_file: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--role",
        role,
        "--architecture",
        architecture,
        "--trace-output",
        str(trace_output),
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if token_ids_file is not None:
        command.extend(("--token-ids-file", str(token_ids_file)))
    return command


def _run_coordinator_in_directory(args: argparse.Namespace, trace_dir: Path) -> None:
    import torch

    reference_path = trace_dir / "reference.pt"
    target_path = trace_dir / "target.pt"
    token_ids_path = trace_dir / "prompt_token_ids.json"

    print("Running reference worker...")
    subprocess.run(
        _worker_command(
            args,
            "reference",
            args.reference_architecture,
            reference_path,
        ),
        check=True,
    )
    reference_trace = torch.load(
        reference_path, map_location="cpu", weights_only=True
    )
    with token_ids_path.open("w", encoding="utf-8") as file:
        json.dump(reference_trace["metadata"]["prompt_token_ids"], file)

    print("Running target worker with reference token IDs...")
    subprocess.run(
        _worker_command(
            args,
            "target",
            args.target_architecture,
            target_path,
            token_ids_path,
        ),
        check=True,
    )
    target_trace = torch.load(target_path, map_location="cpu", weights_only=True)

    report = compare_traces(reference_trace, target_trace, args)
    print_report(report, show_all=args.show_all)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        print(f"\nSaved JSON report to {args.output}")


def run_coordinator(args: argparse.Namespace) -> None:
    if args.max_model_len < 2:
        raise ValueError("--max-model-len must be at least 2")
    if args.trace_dir is not None:
        args.keep_traces = True

    if args.keep_traces:
        if args.trace_dir is None:
            trace_dir = Path(tempfile.mkdtemp(prefix="exaone-smoothquant-"))
        else:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_dir = Path(
                tempfile.mkdtemp(prefix="run-", dir=args.trace_dir.resolve())
            )
        _run_coordinator_in_directory(args, trace_dir)
        print(f"Raw traces retained in {trace_dir}")
        return

    with tempfile.TemporaryDirectory(prefix="exaone-smoothquant-") as directory:
        _run_coordinator_in_directory(args, Path(directory))


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
