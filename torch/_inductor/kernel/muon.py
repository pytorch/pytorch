import math
import threading
from collections import OrderedDict
from typing import Any

import torch

from ..utils import clear_on_fresh_cache


def _match_normalization(
    placeholder: torch.fx.Node,
    numerator: Any,
    denominator: Any,
) -> tuple[float, bool] | None:
    """Match the decomposed BF16 X / max(||X||_F, eps) normalization."""

    def converted_input(node: Any, dtype: torch.dtype) -> torch.fx.Node | None:
        if (
            isinstance(node, torch.fx.Node)
            and node.target is torch.ops.prims.convert_element_type.default
            and len(node.args) == 2
            and node.args[1] is dtype
            and not node.kwargs
        ):
            input_node = node.args[0]
            return input_node if isinstance(input_node, torch.fx.Node) else None
        return None

    if not isinstance(numerator, torch.fx.Node):
        return None
    source = numerator
    transposed = False
    if source.target is torch.ops.aten.permute.default:
        if source.args[1:] != ([1, 0],):
            return None
        source = source.args[0]
        transposed = True
    cast_source = converted_input(source, torch.bfloat16)
    if cast_source is not None:
        source = cast_source
    else:
        source_val = placeholder.meta.get("val")
        if (
            not isinstance(source_val, torch.Tensor)
            or source_val.dtype is not torch.bfloat16
        ):
            return None
    if source is not placeholder:
        return None

    clamp = converted_input(denominator, torch.bfloat16)
    if (
        not isinstance(clamp, torch.fx.Node)
        or clamp.target is not torch.ops.aten.clamp_min.default
        or len(clamp.args) != 2
        or clamp.kwargs
        or not isinstance(clamp.args[1], (int, float))
    ):
        return None
    rounded_norm = converted_input(clamp.args[0], torch.float32)
    norm = converted_input(rounded_norm, torch.bfloat16)
    if (
        not isinstance(norm, torch.fx.Node)
        or norm.target is not torch.ops.aten.pow.Tensor_Scalar
        or len(norm.args) != 2
        or norm.args[1] != 0.5
        or norm.kwargs
    ):
        return None
    total = norm.args[0]
    if (
        not isinstance(total, torch.fx.Node)
        or total.target is not torch.ops.aten.sum.dim_IntList
        or len(total.args) != 2
        or total.args[1] is not None
        or total.kwargs
    ):
        return None
    squared = total.args[0]
    if (
        not isinstance(squared, torch.fx.Node)
        or squared.target is not torch.ops.aten.pow.Tensor_Scalar
        or len(squared.args) != 2
        or squared.args[1] != 2
        or squared.kwargs
        or converted_input(squared.args[0], torch.float32) is not numerator
    ):
        return None
    return float(clamp.args[1]), transposed


def match_muon_foreach(
    gm: torch.fx.GraphModule,
) -> tuple[tuple[float, float, float, int, float], list[int]] | None:
    """Match the functional Muon Newton-Schulz body captured by foreach_map."""
    if any(
        node.is_impure()
        for node in gm.graph.nodes
        if node.op not in ("placeholder", "output")
    ):
        return None
    all_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    positions = [
        index
        for index, node in enumerate(all_placeholders)
        if isinstance(node.meta.get("val"), torch.Tensor)
    ]
    placeholders = [all_placeholders[index] for index in positions]
    output = next((node for node in gm.graph.nodes if node.op == "output"), None)
    if output is None or len(output.args) != 1:
        return None
    outputs = output.args[0]
    if isinstance(outputs, torch.fx.Node) and len(placeholders) == 1:
        outputs = (outputs,)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != len(placeholders):
        return None
    options: tuple[float, float, float, int, float] | None = None
    for placeholder, result in zip(placeholders, outputs):
        if not isinstance(result, torch.fx.Node):
            return None
        output_transposed = False
        if result.target is torch.ops.aten.permute.default:
            if result.args[1] != [1, 0]:
                return None
            result = result.args[0]
            output_transposed = True
        steps = 0
        lane_options = None
        while (
            isinstance(result, torch.fx.Node)
            and result.target is torch.ops.aten.addmm.default
            and "alpha" not in result.kwargs
        ):
            x, gram_update, rhs = result.args
            if (
                x is not rhs
                or not isinstance(gram_update, torch.fx.Node)
                or gram_update.target is not torch.ops.aten.addmm.default
            ):
                return None
            gram = gram_update.args[0]
            if gram_update.args != (gram, gram, gram) or not isinstance(
                gram, torch.fx.Node
            ):
                return None
            if gram.target is torch.ops.inductor.quack_symmetric_mm.default:
                if gram.args != (x,):
                    return None
            else:
                if (
                    gram.target is not torch.ops.aten.mm.default
                    or gram.args[0] is not x
                ):
                    return None
                transpose = gram.args[1]
                if (
                    not isinstance(transpose, torch.fx.Node)
                    or transpose.target is not torch.ops.aten.permute.default
                    or transpose.args != (x, [1, 0])
                ):
                    return None
            current = (
                result.kwargs.get("beta", 1.0),
                gram_update.kwargs.get("beta", 1.0),
                gram_update.kwargs.get("alpha", 1.0),
            )
            if lane_options is not None and current != lane_options:
                return None
            lane_options = current
            result = x
            steps += 1
        if (
            steps == 0
            or not isinstance(result, torch.fx.Node)
            or result.target is not torch.ops.aten.div.Tensor
            or len(result.args) != 2
        ):
            return None
        normalization = _match_normalization(placeholder, *result.args)
        if normalization is None:
            return None
        eps, input_transposed = normalization
        if input_transposed != output_transposed or lane_options is None:
            return None
        a, b, c = lane_options
        if not (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and isinstance(c, (int, float))
        ):
            return None
        lane_options = (
            float(a),
            float(b),
            float(c),
            steps,
            float(eps),
        )
        if options is not None and lane_options != options:
            return None
        options = lane_options
    return (options, positions) if options is not None else None


def _reference(
    x: torch.Tensor,
    coefficients: tuple[float, float, float],
    steps: int,
    eps: float,
) -> torch.Tensor:
    transpose = x.shape[0] > x.shape[1]
    x = x.T if transpose else x
    x = x.bfloat16() / x.bfloat16().norm().clamp_min(eps)
    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.T
        update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, update, x, beta=a)
    return x.T if transpose else x


def _supports(
    device: torch.device,
    dtype: torch.dtype,
    shapes: list[tuple[int, int]],
    steps: int,
) -> bool:
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or dtype is not torch.bfloat16
        or steps <= 0
        or any(m <= 0 or k <= 0 or m % 8 or k % 8 for m, k in shapes)
    ):
        return False
    from torch._inductor.utils import ensure_cute_available

    m = shapes[0][0]
    count = len(shapes)
    max_k = max(shape[1] for shape in shapes)
    return (
        ensure_cute_available()
        and torch.cuda.get_device_capability(device)[0] in (10, 11)
        and (count >= 4 or m >= 5120 or (m >= 4096 and (count >= 2 or max_k >= 2 * m)))
    )


def _partition_inputs(
    metadata: list[tuple[torch.device, torch.dtype, tuple[int, int]]], steps: int
) -> list[tuple[list[int], bool]]:
    buckets: dict[
        tuple[torch.device, torch.dtype, int | tuple[int, int]], list[int]
    ] = {}
    small: dict[tuple[torch.device, torch.dtype], dict[tuple[int, int], list[int]]] = {}
    for index, (device, dtype, shape) in enumerate(metadata):
        m = shape[0]
        if m <= 1024:
            small.setdefault((device, dtype), {}).setdefault(shape, []).append(index)
        else:
            bucket = shape if m >= 4096 else m
            buckets.setdefault((device, dtype, bucket), []).append(index)
    for device_dtype, shape_buckets in small.items():
        remainder = []
        for shape, indices in shape_buckets.items():
            if len(indices) >= 6:
                buckets[(*device_dtype, shape)] = indices
            else:
                remainder.extend(indices)
        if remainder:
            buckets[(*device_dtype, 0)] = remainder

    chunks = []
    for (device, dtype, _), indices in buckets.items():
        indices.sort(key=lambda index: metadata[index][2])
        m = metadata[indices[0]][2][0]
        max_chunk = 4 if m >= 7168 else 8 if m >= 4096 else 32
        count = math.ceil(len(indices) / max_chunk)
        size, extra = divmod(len(indices), count)
        offset = 0
        for chunk_index in range(count):
            chunk = indices[offset : offset + size + (chunk_index < extra)]
            offset += len(chunk)
            shapes = [metadata[index][2] for index in chunk]
            chunks.append((chunk, _supports(device, dtype, shapes, steps)))
    return chunks


def can_use_grouped_muon(
    metadata: list[tuple[torch.device, torch.dtype, tuple[int, int]]], steps: int
) -> bool:
    return any(supported for _, supported in _partition_inputs(metadata, steps))


def _workspace_bytes(
    metadata: list[tuple[torch.device, torch.dtype, tuple[int, int]]], steps: int
) -> int:
    total = 0
    for chunk, supported in _partition_inputs(metadata, steps):
        if not supported:
            continue
        shapes = [metadata[index][2] for index in chunk]
        if len(shapes) <= 3:
            gram_elements = sum(m * m for m in {shape[0] for shape in shapes})
        else:
            gram_elements = sum(m * m for m, _ in shapes)
        total += (
            2
            * torch.bfloat16.itemsize
            * (sum(m * k for m, k in shapes) + gram_elements)
        )
    return total


class _MuonForeachPlan:
    def __init__(
        self,
        inputs: list[torch.Tensor],
        coefficients: tuple[float, float, float],
        steps: int,
    ) -> None:
        self._run_lock = threading.Lock()
        original_shapes: list[tuple[int, int]] = [
            (x.shape[0], x.shape[1]) for x in inputs
        ]
        metadata = [(x.device, x.dtype, (min(x.shape), max(x.shape))) for x in inputs]
        self.chunks: list[tuple[list[int], Any | None]] = []
        self.workspace_bytes = _workspace_bytes(metadata, steps)
        for chunk, supported in _partition_inputs(metadata, steps):
            device = metadata[chunk[0]][0]
            shapes = [original_shapes[index] for index in chunk]
            plan = None
            if supported:
                from ._grouped_symmetric_mm import (
                    GroupedMuonPlan,
                    PackedSymmetricMuonPlan,
                    SequentialSymmetricMuonPlan,
                )

                if len(shapes) <= 3:
                    plan = SequentialSymmetricMuonPlan(
                        shapes, device, coefficients, steps
                    )
                elif min(shapes[0]) >= 4096 and all(
                    (min(shape), max(shape)) == (min(shapes[0]), max(shapes[0]))
                    for shape in shapes
                ):
                    plan = PackedSymmetricMuonPlan(shapes, device, coefficients, steps)
                else:
                    plan = GroupedMuonPlan(shapes, device, coefficients, steps)
            self.chunks.append((chunk, plan))
        self.coefficients = coefficients
        self.steps = steps

    def __call__(self, inputs: list[torch.Tensor], eps: float) -> list[torch.Tensor]:
        outputs = [inputs[0]] * len(inputs)
        for indices, plan in self.chunks:
            selected = [inputs[index] for index in indices]
            if plan is None:
                results = [
                    _reference(x, self.coefficients, self.steps, eps) for x in selected
                ]
            else:
                results = plan(selected, eps)
            for index, result in zip(indices, results):
                outputs[index] = result.clone()
        return outputs


_PLAN_LOCK = threading.Lock()
_MAX_CACHED_PLAN_BYTES = 1024**3


class _PlanCache(OrderedDict[tuple[Any, ...], _MuonForeachPlan]):
    def __init__(self) -> None:
        super().__init__()
        self.total_bytes = 0

    def clear(self) -> None:
        super().clear()
        self.total_bytes = 0

    def cache_clear(self) -> None:
        with _PLAN_LOCK:
            self.clear()


_PLAN_CACHE = _PlanCache()
_PLAN_BUILD_EVENTS: dict[tuple[Any, ...], threading.Event] = {}
clear_on_fresh_cache(_PLAN_CACHE)


def grouped_muon_impl(
    inputs: list[torch.Tensor],
    a: float,
    b: float,
    c: float,
    steps: int,
    eps: float,
) -> list[torch.Tensor]:
    streams = tuple(
        torch.cuda.current_stream(x.device).cuda_stream
        if x.device.type == "cuda"
        else None
        for x in inputs
    )
    key = (
        tuple((x.device, x.dtype, tuple(x.shape)) for x in inputs),
        streams,
        a,
        b,
        c,
        steps,
    )
    metadata = [(x.device, x.dtype, (min(x.shape), max(x.shape))) for x in inputs]
    workspace_bytes = _workspace_bytes(metadata, steps)
    plan = None
    while plan is None:
        with _PLAN_LOCK:
            plan = _PLAN_CACHE.get(key)
            if plan is not None:
                _PLAN_CACHE.move_to_end(key)
                break
            build_event = _PLAN_BUILD_EVENTS.get(key)
            if build_event is None:
                build_event = threading.Event()
                _PLAN_BUILD_EVENTS[key] = build_event
                break
        build_event.wait()
    if plan is None:
        try:
            candidate = _MuonForeachPlan(inputs, (a, b, c), steps)
        except BaseException:
            with _PLAN_LOCK:
                _PLAN_BUILD_EVENTS.pop(key).set()
            raise
        with _PLAN_LOCK:
            plan = _PLAN_CACHE.get(key)
            if plan is None:
                plan = candidate
                if workspace_bytes <= _MAX_CACHED_PLAN_BYTES:
                    while (
                        _PLAN_CACHE
                        and _PLAN_CACHE.total_bytes + workspace_bytes
                        > _MAX_CACHED_PLAN_BYTES
                    ):
                        _, evicted = _PLAN_CACHE.popitem(last=False)
                        _PLAN_CACHE.total_bytes -= evicted.workspace_bytes
                        del evicted
                    _PLAN_CACHE[key] = plan
                    _PLAN_CACHE.total_bytes += plan.workspace_bytes
            else:
                _PLAN_CACHE.move_to_end(key)
            _PLAN_BUILD_EVENTS.pop(key).set()
    with plan._run_lock:
        return plan(inputs, eps)
