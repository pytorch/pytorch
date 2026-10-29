"""Direct FX lowering into experimental C++-owned, materialized LoopsIR.

Independent of GraphLowering. Structured construction, verification and printing
are C++; this lowerer intentionally supports only a static operator subset.

Two modes share this module:

- Diagnostic (default): ``lower_fx_to_cpp(gm, inputs)`` lowers the historic
  static subset (f32 pointwise add/mul/relu with unit-alpha folding, sum
  reductions with bodies, permute/expand views, mm opaque). Behavior is
  unchanged; the default-off ``print_cpp_loops_ir`` hook keeps working.

- Executable (opt-in): ``lower_fx_to_cpp(gm, inputs, executable=True)``
  lowers for the strict static single-CUDA-device inference backend. Only
  f32 add (alpha must be exactly 1)/mul/relu become generated Compute;
  every reduction (audited ``torch.Tag.reduction``) and every other pure
  tensor/custom op becomes an explicit structured eager extern call with the
  real symbolic target, ordered args/kwargs, and full result metadata. No
  Python callbacks enter the IR and no whole-graph Inductor delegation
  occurs. ``alpha != 1`` falls back to an exact ``aten.add`` extern (never a
  mul+add rewrite, which could change rounding/contraction).
"""

from __future__ import annotations

import functools
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.fx import GraphModule, Node


class LoopsIrUnsupportedError(RuntimeError):
    """The direct lowerer cannot faithfully represent an FX operation yet."""


@functools.lru_cache(None)
def load_cpp_ir() -> Any:
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parents[1] / "csrc/inductor/experimental/loops_ir"
    # Full pipeline sources. tiled_bindings.cpp is owned by the tiling worker;
    # until it lands the extension cannot link (bindings.cpp forward-declares
    # bind_tiled_ir). Do not trim this list to hide the dependency.
    names = (
        "ir.cpp",
        "printer.cpp",
        "passes.cpp",
        "kernel_ir.cpp",
        "tiled_ir.cpp",
        "tiled_bindings.cpp",
        "bindings.cpp",
    )
    return load(
        name="inductor_experimental_loops_ir",
        sources=[str(source / name) for name in names],
        extra_cflags=["-O0", "-g0", "-std=c++17"],
        verbose=False,
    )


@dataclass(frozen=True)
class _TensorRef:
    name: str
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    is_view: bool = False
    dtype: str = "f32"


def _static_ints(values: Any, description: str) -> tuple[int, ...]:
    if any(type(value) is not int for value in values):
        raise LoopsIrUnsupportedError(f"symbolic {description} are not supported yet")
    return tuple(values)


def _tensor_spec(value: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(value, torch.Tensor):
        raise LoopsIrUnsupportedError("expected a tensor input/result")
    if value.dtype != torch.float32 or value.layout != torch.strided:
        raise LoopsIrUnsupportedError("only strided f32 tensors are supported yet")
    sizes = _static_ints(value.shape, "sizes")
    strides = _static_ints(value.stride(), "strides")
    if any(size < 0 for size in sizes) or any(stride < 0 for stride in strides):
        raise LoopsIrUnsupportedError("negative sizes/strides are unsupported")
    return sizes, strides


def _storage_elements(sizes: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if 0 in sizes:
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(sizes, strides))


def _scalar(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise LoopsIrUnsupportedError("expected a finite numeric scalar")
    if abs(value) > 3.4028234663852886e38:
        raise LoopsIrUnsupportedError("scalar is outside the f32 range")
    return float(value)


class _Lowerer:
    def __init__(self, gm: GraphModule, example_inputs: list[Any] | tuple[Any, ...]):
        self.cpp = load_cpp_ir()
        self.program = self.cpp.Program("fx_program")
        self.gm = gm
        self.example_inputs = iter(example_inputs)
        self.refs: dict[Node, _TensorRef] = {}
        self.view_number = 0
        self.names = {node.name for node in gm.graph.nodes}

    def index(self, value: int) -> Any:
        return self.cpp.Index.constant(value)

    def offset(self, strides: tuple[int, ...], coordinates: list[Any]) -> Any:
        result = self.index(0)
        for stride, coordinate in zip(strides, coordinates):
            term = self.cpp.Index.binary("mul", coordinate, self.index(stride))
            result = self.cpp.Index.binary("add", result, term)
        return result

    def descriptor(self, name: str, ref: _TensorRef) -> Any:
        # Input buffers bind to tensor.data_ptr(), not the storage base. Thus
        # input storage_offset is already in the pointer; offsets here are relative.
        return self.cpp.Descriptor(
            name, ref.name, list(ref.sizes), list(ref.strides), 0
        )

    def ref(self, value: Any) -> _TensorRef:
        if not isinstance(value, Node) or value not in self.refs:
            raise LoopsIrUnsupportedError(
                "expected a previously lowered tensor operand"
            )
        return self.refs[value]

    def result_ref(self, node: Node) -> _TensorRef:
        sizes, strides = _tensor_spec(node.meta.get("val"))
        return _TensorRef(node.name, sizes, strides)

    def buffer(self, ref: _TensorRef) -> Any:
        return self.cpp.Buffer(
            ref.name, "f32", _storage_elements(ref.sizes, ref.strides)
        )

    def ranges(
        self, sizes: tuple[int, ...], prefix: str
    ) -> tuple[list[Any], list[Any]]:
        names = [f"{prefix}{axis}" for axis in range(len(sizes))]
        return (
            [self.cpp.Range(name, 0, size) for name, size in zip(names, sizes)],
            [self.cpp.Index.variable(name) for name in names],
        )

    def add_view(
        self,
        name: str,
        source: _TensorRef,
        sizes: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> _TensorRef:
        self.program.add_view(
            self.cpp.View(name, source.name, list(sizes), list(strides), 0)
        )
        return _TensorRef(name, sizes, strides, True)

    def broadcast(self, source: _TensorRef, sizes: tuple[int, ...]) -> _TensorRef:
        if source.sizes == sizes:
            return source
        if len(source.sizes) > len(sizes):
            raise LoopsIrUnsupportedError("broadcast rank mismatch")
        pad = len(sizes) - len(source.sizes)
        old_sizes = (1,) * pad + source.sizes
        old_strides = (0,) * pad + source.strides
        strides = []
        for old, new, stride in zip(old_sizes, sizes, old_strides):
            if old != new and old != 1:
                raise LoopsIrUnsupportedError("incompatible broadcast dimensions")
            strides.append(0 if old == 1 else stride)
        while f"broadcast_{self.view_number}" in self.names:
            self.view_number += 1
        name = f"broadcast_{self.view_number}"
        self.names.add(name)
        self.view_number += 1
        return self.add_view(name, source, sizes, tuple(strides))

    def read(self, identifier: int, ref: _TensorRef, coordinates: list[Any]) -> Any:
        if ref.is_view:
            return self.cpp.Instruction.view_read(identifier, ref.name, coordinates)
        return self.cpp.Instruction.load(
            identifier, ref.name, self.offset(ref.strides, coordinates)
        )

    def pointwise(self, node: Node, op: str) -> None:
        result = self.result_ref(node)
        ranges, coordinates = self.ranges(result.sizes, "p")
        operands = node.args[:1] if op == "relu" else node.args[:2]
        inputs: list[str] = []
        instructions = []
        for identifier, operand in enumerate(operands):
            if isinstance(operand, Node):
                ref = self.broadcast(self.ref(operand), result.sizes)
                if ref.name not in inputs:
                    inputs.append(ref.name)
                instructions.append(self.read(identifier, ref, coordinates))
            else:
                instructions.append(
                    self.cpp.Instruction.constant(identifier, _scalar(operand))
                )
        compute = self.cpp.Compute(
            node.name, inputs, [self.buffer(result)], ranges, [], []
        )
        for instruction in instructions:
            compute.add_post_instruction(instruction)
        identifier = len(operands)
        if op == "relu":
            compute.add_post_instruction(self.cpp.Instruction.relu(identifier, 0))
        else:
            rhs = 1
            alpha = _scalar(node.kwargs.get("alpha", 1)) if op == "add" else 1.0
            if alpha != 1.0:
                compute.add_post_instruction(
                    self.cpp.Instruction.constant(identifier, alpha)
                )
                compute.add_post_instruction(
                    self.cpp.Instruction.binary(identifier + 1, "mul", rhs, identifier)
                )
                rhs = identifier + 1
                identifier += 2
            compute.add_post_instruction(
                self.cpp.Instruction.binary(identifier, op, 0, rhs)
            )
        compute.add_post_instruction(
            self.cpp.Instruction.store(
                result.name, self.offset(result.strides, coordinates), identifier
            )
        )
        self.program.add_compute(compute)
        self.refs[node] = result

    def reduce_sum(self, node: Node) -> None:
        source = self.ref(node.args[0])
        rank = len(source.sizes)
        dimensions = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if dimensions is None or dimensions == [] or dimensions == ():
            axes = list(range(rank))
        else:
            dimensions = [dimensions] if type(dimensions) is int else list(dimensions)
            if any(type(axis) is not int for axis in dimensions):
                raise LoopsIrUnsupportedError("nonconstant reduction axes")
            if rank == 0:
                if any(axis not in (0, -1) for axis in dimensions):
                    raise LoopsIrUnsupportedError("scalar reduction axis out of range")
                axes = []
            else:
                if any(axis < -rank or axis >= rank for axis in dimensions):
                    raise LoopsIrUnsupportedError("reduction axis out of range")
                axes = sorted(axis % rank for axis in dimensions)
                if len(set(axes)) != len(axes):
                    raise LoopsIrUnsupportedError("duplicate reduction axes")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        if type(keepdim) is not bool:
            raise LoopsIrUnsupportedError("nonconstant keepdim")
        if node.kwargs.get("dtype") not in (None, torch.float32):
            raise LoopsIrUnsupportedError("non-f32 sum output")
        result = self.result_ref(node)
        remaining = [axis for axis in range(rank) if axis not in axes]
        pointwise_sizes = tuple(source.sizes[axis] for axis in remaining)
        reduction_sizes = tuple(source.sizes[axis] for axis in axes)
        p_ranges, p_coordinates = self.ranges(pointwise_sizes, "p")
        r_ranges, r_coordinates = self.ranges(reduction_sizes, "r")
        input_coordinates = [self.index(0) for _ in source.sizes]
        for axis, coordinate in zip(remaining, p_coordinates):
            input_coordinates[axis] = coordinate
        for axis, coordinate in zip(axes, r_coordinates):
            input_coordinates[axis] = coordinate
        output_coordinates = (
            [
                self.index(0) if axis in axes else input_coordinates[axis]
                for axis in range(rank)
            ]
            if keepdim
            else p_coordinates
        )
        expected_sizes = (
            tuple(1 if axis in axes else source.sizes[axis] for axis in range(rank))
            if keepdim
            else pointwise_sizes
        )
        if result.sizes != expected_sizes:
            raise LoopsIrUnsupportedError(
                "sum result metadata does not match reduction axes"
            )
        accumulators = [self.cpp.Accumulator("sum", "f32", 0.0, "add")] if axes else []
        compute = self.cpp.Compute(
            node.name,
            [source.name],
            [self.buffer(result)],
            p_ranges,
            r_ranges,
            accumulators,
        )
        if axes:
            compute.add_reduction_instruction(self.read(0, source, input_coordinates))
            compute.add_reduction_instruction(self.cpp.Instruction.accumulate([0]))
            compute.add_post_instruction(self.cpp.Instruction.read_accumulator(0, 0))
        else:
            compute.add_post_instruction(self.read(0, source, input_coordinates))
        compute.add_post_instruction(
            self.cpp.Instruction.store(
                result.name, self.offset(result.strides, output_coordinates), 0
            )
        )
        self.program.add_compute(compute)
        self.refs[node] = result

    def view(self, node: Node) -> None:
        source = self.ref(node.args[0])
        result = self.result_ref(node)
        if node.target == torch.ops.aten.permute.default:
            dimensions = list(node.args[1])
            rank = len(source.sizes)
            if any(
                type(axis) is not int or not -rank <= axis < rank for axis in dimensions
            ):
                raise LoopsIrUnsupportedError("invalid permutation")
            axes = [axis % rank for axis in dimensions]
            if sorted(axes) != list(range(rank)):
                raise LoopsIrUnsupportedError("invalid permutation")
            sizes = tuple(source.sizes[axis] for axis in axes)
            strides = tuple(source.strides[axis] for axis in axes)
            if result.sizes != sizes or result.strides != strides:
                raise LoopsIrUnsupportedError("permutation metadata mismatch")
            self.refs[node] = self.add_view(node.name, source, sizes, strides)
        else:
            expanded = self.broadcast(source, result.sizes)
            if expanded is source:
                expanded = self.add_view(
                    node.name, source, source.sizes, source.strides
                )
            if expanded.strides != result.strides:
                raise LoopsIrUnsupportedError("expand metadata mismatch")
            self.refs[node] = expanded

    def matrix_multiply(self, node: Node) -> None:
        lhs, rhs = self.ref(node.args[0]), self.ref(node.args[1])
        result = self.result_ref(node)
        if len(lhs.sizes) != 2 or len(rhs.sizes) != 2 or lhs.sizes[1] != rhs.sizes[0]:
            raise LoopsIrUnsupportedError("only rank-two matrix multiply is supported")
        if result.sizes != (lhs.sizes[0], rhs.sizes[1]):
            raise LoopsIrUnsupportedError("matrix multiply metadata mismatch")
        self.program.add_extern_call(
            self.cpp.ExternCall(
                node.name,
                "aten.mm.default",
                [lhs.name, rhs.name],
                [self.buffer(result)],
                [],
                "unknown",
                False,
                [
                    self.descriptor("lhs", lhs),
                    self.descriptor("rhs", rhs),
                    self.descriptor("result", result),
                ],
            )
        )
        self.refs[node] = result

    def lower(self) -> Any:
        aten = torch.ops.aten
        pointwise = {
            aten.add.Tensor: "add",
            aten.add.Scalar: "add",
            aten.mul.Tensor: "mul",
            aten.mul.Scalar: "mul",
            aten.relu.default: "relu",
        }
        for node in self.gm.graph.nodes:
            try:
                if node.op == "placeholder":
                    value = next(self.example_inputs)
                    sizes, strides = _tensor_spec(value)
                    ref = _TensorRef(node.name, sizes, strides)
                    self.program.add_input(self.buffer(ref))
                    self.program.add_input_descriptor(self.descriptor(node.name, ref))
                    self.refs[node] = ref
                elif node.op == "call_function" and node.target in pointwise:
                    self.pointwise(node, pointwise[node.target])
                elif node.op == "call_function" and node.target in (
                    aten.sum.default,
                    aten.sum.dim_IntList,
                ):
                    self.reduce_sum(node)
                elif node.op == "call_function" and node.target in (
                    aten.permute.default,
                    aten.expand.default,
                ):
                    self.view(node)
                elif node.op == "call_function" and node.target == aten.mm.default:
                    self.matrix_multiply(node)
                elif node.op == "output":
                    from torch.utils._pytree import tree_leaves

                    for number, value in enumerate(tree_leaves(node.args[0])):
                        self.program.add_output(
                            self.descriptor(f"output{number}", self.ref(value))
                        )
                else:
                    raise LoopsIrUnsupportedError(
                        f"unsupported operation {node.op}: {node.target}"
                    )
            except (LoopsIrUnsupportedError, StopIteration) as error:
                raise LoopsIrUnsupportedError(
                    f"FX node {node.name} ({node.target}): {error}"
                ) from error
        try:
            next(self.example_inputs)
        except StopIteration:
            pass
        else:
            raise LoopsIrUnsupportedError("more example inputs than FX placeholders")
        self.program.verify()
        return self.program


# ---------------------------------------------------------------------------
# Executable lowering (strict static single-CUDA-device inference).
# ---------------------------------------------------------------------------

_TORCH_DTYPE_TO_IR: dict[Any, str] = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "bool",
    torch.complex64: "c64",
    torch.complex128: "c128",
}

_TORCH_DTYPE_TO_CALLARG: dict[Any, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


_TORCH_LAYOUT_TO_CALLARG: dict[Any, str] = {
    torch.strided: "strided",
    torch.sparse_coo: "sparse_coo",
    torch.sparse_csr: "sparse_csr",
    torch.sparse_csc: "sparse_csc",
    torch.sparse_bsr: "sparse_bsr",
    torch.sparse_bsc: "sparse_bsc",
    torch._mkldnn: "mkldnn",  # type: ignore[attr-defined]
}

_TORCH_MEMORY_FORMAT_TO_CALLARG: dict[Any, str] = {
    torch.contiguous_format: "contiguous_format",
    torch.channels_last: "channels_last",
    torch.channels_last_3d: "channels_last_3d",
    torch.preserve_format: "preserve_format",
}


def _boundary_spec(value: Any) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Sizes/strides/ir-dtype for an opaque boundary tensor.

    Boundary buffers may use any common dtype in _TORCH_DTYPE_TO_IR; generated
    Compute data must stay f32 (checked separately). Layout must be strided,
    shapes static, signs non-negative.
    """
    if not isinstance(value, torch.Tensor):
        raise LoopsIrUnsupportedError("expected a tensor input/result")
    if value.layout != torch.strided:
        raise LoopsIrUnsupportedError("only strided tensors are supported")
    ir_dtype = _TORCH_DTYPE_TO_IR.get(value.dtype)
    if ir_dtype is None:
        raise LoopsIrUnsupportedError(f"unsupported boundary dtype {value.dtype}")
    sizes = _static_ints(value.shape, "sizes")
    strides = _static_ints(value.stride(), "strides")
    if any(size < 0 for size in sizes) or any(stride < 0 for stride in strides):
        raise LoopsIrUnsupportedError("negative sizes/strides are unsupported")
    return sizes, strides, ir_dtype


def _qualified_target_name(target: Any) -> str:
    """Symbolic dotted name (e.g. 'aten.add.Tensor'); no Python callbacks."""
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    raise LoopsIrUnsupportedError(f"non-overload call target {target!r}")


def _is_reduction_target(target: Any) -> bool:
    try:
        tags = target.tags
    except Exception:
        return False
    try:
        return torch.Tag.reduction in tags
    except Exception:
        return False


def _reject_unsafe_target(target: Any) -> None:
    """Reject mutation, randomness, and dynamic/data-dependent outputs.

    Pure-but-unsupported ops (including custom ops without these tags) are
    allowed as explicit unknown-effects extern barriers; anything mutating,
    nondeterministic, or shape-dynamic is an error, never an approximate
    substitution.
    """
    tags: Any = ()
    try:
        tags = target.tags
    except Exception:
        tags = ()
    try:
        if torch.Tag.nondeterministic_seeded in tags:
            raise LoopsIrUnsupportedError("nondeterministic (seeded) op rejected")
        if torch.Tag.nondeterministic_bitwise in tags:
            raise LoopsIrUnsupportedError("nondeterministic (bitwise) op rejected")
        if torch.Tag.data_dependent_output in tags:
            raise LoopsIrUnsupportedError("data-dependent output shape rejected")
        if torch.Tag.dynamic_output_shape in tags:
            raise LoopsIrUnsupportedError("dynamic output shape rejected")
        if torch.Tag.inplace in tags:
            raise LoopsIrUnsupportedError("in-place mutation rejected")
    except LoopsIrUnsupportedError:
        raise
    except Exception:
        pass
    schema = getattr(target, "_schema", None)
    if schema is not None:
        try:
            for arg in schema.arguments:
                alias = getattr(arg, "alias_info", None)
                if alias is not None and getattr(alias, "is_write", False):
                    raise LoopsIrUnsupportedError(
                        f"mutating schema argument '{arg.name}' rejected"
                    )
        except LoopsIrUnsupportedError:
            raise
        except Exception:
            pass


def _to_call_arg(cpp: Any, value: Any, tensor_names: dict[Node, str]) -> Any:
    """Convert an FX argument to a typed CallArg (tensors by buffer name)."""
    if value is None:
        return cpp.CallArg.none()
    if isinstance(value, Node):
        name = tensor_names.get(value)
        if name is None:
            raise LoopsIrUnsupportedError(
                "tensor argument is not a previously lowered tensor"
            )
        return cpp.CallArg.tensor(name)
    if type(value) is bool:
        return cpp.CallArg.boolean(value)
    if type(value) is int:
        if not (_INT64_MIN <= value <= _INT64_MAX):
            raise LoopsIrUnsupportedError("integer argument outside int64 range")
        return cpp.CallArg.integer(value)
    if type(value) is float:
        # Static non-finite mask constants: native accepts +-inf (exact f32
        # literals, no payload ambiguity) but rejects NaN (payloads cannot
        # round-trip). Mirror that boundary exactly; the factories raise
        # ValueError which we convert to the lane error.
        if math.isnan(value):
            raise LoopsIrUnsupportedError("non-finite float argument rejected")
        try:
            return cpp.CallArg.floating(float(value))
        except ValueError as error:
            raise LoopsIrUnsupportedError(str(error)) from error
    if type(value) is str:
        return cpp.CallArg.string(value)
    if isinstance(value, torch.dtype):
        name = _TORCH_DTYPE_TO_CALLARG.get(value)
        if name is None:
            raise LoopsIrUnsupportedError(f"unsupported dtype argument {value}")
        return cpp.CallArg.dtype(name)
    if isinstance(value, torch.layout):
        name = _TORCH_LAYOUT_TO_CALLARG.get(value)
        if name is None:
            raise LoopsIrUnsupportedError(f"unsupported layout argument {value}")
        try:
            return cpp.CallArg.layout(name)
        except ValueError as error:
            raise LoopsIrUnsupportedError(str(error)) from error
    if isinstance(value, torch.memory_format):
        name = _TORCH_MEMORY_FORMAT_TO_CALLARG.get(value)
        if name is None:
            raise LoopsIrUnsupportedError(f"unsupported memory_format argument {value}")
        try:
            return cpp.CallArg.memory_format(name)
        except ValueError as error:
            raise LoopsIrUnsupportedError(str(error)) from error
    if isinstance(value, torch.device):
        # Static single-CUDA-device lane: the device must be the compiled
        # CUDA device. Preserve it as an exact string payload (validated
        # breadth-first: no repr-based execution, fixed 'cuda[:index]'
        # spelling checked below).
        if value.type != "cuda":
            raise LoopsIrUnsupportedError(
                f"unsupported device argument {value} (lane is CUDA-only)"
            )
        text = "cuda" if value.index is None else f"cuda:{value.index}"
        return cpp.CallArg.string(text)
    if isinstance(value, list):
        return cpp.CallArg.list(
            [_to_call_arg(cpp, item, tensor_names) for item in value]
        )
    if isinstance(value, tuple):
        return cpp.CallArg.tuple(
            [_to_call_arg(cpp, item, tensor_names) for item in value]
        )
    raise LoopsIrUnsupportedError(f"unsupported call argument {type(value).__name__}")


class _ExecutableLowerer:
    """Strict inference lowering: f32 pointwise Compute, everything else extern.

    - Compute: aten.add (alpha exactly 1), aten.mul, aten.relu on f32 only,
      operand order and FP boundaries preserved exactly.
    - Views: permute/expand/t/transpose/reshape/view aliases when the fake
      metadata is wrapper-representable (sizes/strides carried verbatim, no
      hidden .contiguous()/reshape copies). Non-representable cases fall back
      to structured extern calls.
    - Extern: every Tag.reduction op plus every other pure tensor/custom op,
      with the real dotted target, ordered CallArgs, keyword CallArgs,
      multi-output mixed-dtype results, effects='unknown', may_alias=True.
    - getitem on a multi-output extern result is a ref selection (no IR node).
    """

    def _view_targets_set(self) -> frozenset[Any]:
        aten = torch.ops.aten
        return frozenset(
            {
                aten.permute.default,
                aten.expand.default,
                aten.t.default,
                aten.transpose.int,
                aten.reshape.default,
                aten.view.default,
            }
        )

    def __init__(self, gm: GraphModule, example_inputs: list[Any] | tuple[Any, ...]):
        from torch._inductor.fx_passes.cpp_ir_strict import cpp_ir_numerics

        self.cpp = load_cpp_ir()
        self.program = self.cpp.Program(
            "fx_program_executable", numerics=cpp_ir_numerics()
        )
        self.gm = gm
        self.example_inputs = iter(example_inputs)
        self.refs: dict[Node, _TensorRef] = {}
        # Multi-output extern nodes: FX node -> per-output tensor refs.
        self.tuple_refs: dict[Node, list[_TensorRef]] = {}
        self.storage_groups: dict[str, str] = {}
        self.fresh_storage_groups: set[str] = set()
        self.escaped_storage_groups: set[str] = set()
        self.view_number = 0
        self.names = {node.name for node in gm.graph.nodes}

    # -- construction helpers (same pointer/span contracts as diagnostic) --

    def index(self, value: int) -> Any:
        return self.cpp.Index.constant(value)

    def offset(self, strides: tuple[int, ...], coordinates: list[Any]) -> Any:
        result = self.index(0)
        for stride, coordinate in zip(strides, coordinates):
            term = self.cpp.Index.binary("mul", coordinate, self.index(stride))
            result = self.cpp.Index.binary("add", result, term)
        return result

    def descriptor(self, name: str, ref: _TensorRef) -> Any:
        return self.cpp.Descriptor(
            name, ref.name, list(ref.sizes), list(ref.strides), 0
        )

    def ref(self, value: Any) -> _TensorRef:
        if not isinstance(value, Node) or value not in self.refs:
            raise LoopsIrUnsupportedError(
                "expected a previously lowered tensor operand"
            )
        return self.refs[value]

    def tensor_names(self) -> dict[Node, str]:
        names: dict[Node, str] = {node: ref.name for node, ref in self.refs.items()}
        return names

    def buffer_for(self, ref: _TensorRef) -> Any:
        return self.cpp.Buffer(
            ref.name, ref.dtype, _storage_elements(ref.sizes, ref.strides)
        )

    def ranges(
        self, sizes: tuple[int, ...], prefix: str
    ) -> tuple[list[Any], list[Any]]:
        names = [f"{prefix}{axis}" for axis in range(len(sizes))]
        return (
            [self.cpp.Range(name, 0, size) for name, size in zip(names, sizes)],
            [self.cpp.Index.variable(name) for name in names],
        )

    def add_view(
        self,
        name: str,
        source: _TensorRef,
        sizes: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: str,
    ) -> _TensorRef:
        self.program.add_view(
            self.cpp.View(name, source.name, list(sizes), list(strides), 0)
        )
        self.storage_groups[name] = self.storage_groups[source.name]
        return _TensorRef(name, sizes, strides, True, dtype)

    def broadcast(self, source: _TensorRef, sizes: tuple[int, ...]) -> _TensorRef:
        if source.sizes == sizes:
            return source
        # Metadata-only boundary views/broadcasts (i64 indices, bool masks,
        # etc.) preserve strides/storage/aliases; only generated Compute
        # loads/results must stay f32 (checked in pointwise()).
        if len(source.sizes) > len(sizes):
            raise LoopsIrUnsupportedError("broadcast rank mismatch")
        pad = len(sizes) - len(source.sizes)
        old_sizes = (1,) * pad + source.sizes
        old_strides = (0,) * pad + source.strides
        strides = []
        for old, new, stride in zip(old_sizes, sizes, old_strides):
            if old != new and old != 1:
                raise LoopsIrUnsupportedError("incompatible broadcast dimensions")
            strides.append(0 if old == 1 else stride)
        while f"broadcast_{self.view_number}" in self.names:
            self.view_number += 1
        name = f"broadcast_{self.view_number}"
        self.names.add(name)
        self.view_number += 1
        return self.add_view(name, source, sizes, tuple(strides), source.dtype)

    def read(self, identifier: int, ref: _TensorRef, coordinates: list[Any]) -> Any:
        if ref.is_view:
            return self.cpp.Instruction.view_read(identifier, ref.name, coordinates)
        return self.cpp.Instruction.load(
            identifier, ref.name, self.offset(ref.strides, coordinates)
        )

    # -- result metadata (fake/result layout preserved verbatim) --

    def _fake_to_ref(self, name: str, fake: Any, is_view: bool = False) -> _TensorRef:
        sizes, strides, dtype = _boundary_spec(fake)
        return _TensorRef(name, sizes, strides, is_view, dtype)

    def _node_result_fakes(self, node: Node) -> Any:
        val = node.meta.get("val")
        if val is None:
            raise LoopsIrUnsupportedError("missing fake/result metadata")
        return val

    # -- generated pointwise (f32, alpha==1, order-preserving) --

    def pointwise(self, node: Node, op: str) -> bool:
        """Lower to Compute; return False when the node must go extern.

        Only exact-alpha-1 add, mul, and relu on f32 reach Compute. Anything
        else (alpha != 1, non-f32, unsupported scalar) returns False so the
        caller emits an exact eager extern instead of a mul+add rewrite.
        """
        try:
            fake = self._node_result_fakes(node)
            if isinstance(fake, (list, tuple)):
                return False
            sizes, strides, dtype = _boundary_spec(fake)
        except LoopsIrUnsupportedError:
            return False
        if dtype != "f32":
            return False
        if op == "add":
            alpha = node.kwargs.get("alpha", 1)
            if type(alpha) not in (int, float):
                return False
            if alpha != 1:
                # Exact-alpha fallback: never mul+add (FMA/rounding change).
                return False
        result = _TensorRef(node.name, sizes, strides, False, "f32")
        ranges, coordinates = self.ranges(result.sizes, "p")
        operands = node.args[:1] if op == "relu" else node.args[:2]
        inputs: list[str] = []
        instructions = []
        for identifier, operand in enumerate(operands):
            if isinstance(operand, Node):
                ref = self.ref(operand)
                if ref.dtype != "f32":
                    return False
                ref = self.broadcast(ref, result.sizes)
                if ref.name not in inputs:
                    inputs.append(ref.name)
                instructions.append(self.read(identifier, ref, coordinates))
            else:
                try:
                    instructions.append(
                        self.cpp.Instruction.constant(identifier, _scalar(operand))
                    )
                except LoopsIrUnsupportedError:
                    return False
        compute = self.cpp.Compute(
            node.name, inputs, [self.buffer_for(result)], ranges, [], []
        )
        for instruction in instructions:
            compute.add_post_instruction(instruction)
        identifier = len(operands)
        if op == "relu":
            compute.add_post_instruction(self.cpp.Instruction.relu(identifier, 0))
        else:
            compute.add_post_instruction(
                self.cpp.Instruction.binary(identifier, op, 0, 1)
            )
        compute.add_post_instruction(
            self.cpp.Instruction.store(
                result.name, self.offset(result.strides, coordinates), identifier
            )
        )
        self.program.add_compute(compute)
        self.refs[node] = result
        self.storage_groups[result.name] = result.name
        self.fresh_storage_groups.add(result.name)
        return True

    def inplace_pointwise(self, node: Node, op: str) -> None:
        """Import a private temporary update as a new materialized SSA version."""
        source = self.ref(node.args[0])
        group = self.storage_groups[source.name]
        if (
            source.is_view
            or group not in self.fresh_storage_groups
            or group in self.escaped_storage_groups
        ):
            raise LoopsIrUnsupportedError(
                "in-place update requires unescaped, private temporary storage"
            )
        sizes, strides, dtype = _boundary_spec(self._node_result_fakes(node))
        if (sizes, strides, dtype) != (source.sizes, source.strides, source.dtype):
            raise LoopsIrUnsupportedError("in-place update changes tensor metadata")
        if not self.pointwise(node, op):
            raise LoopsIrUnsupportedError(
                "in-place update is outside the strict FP32 pointwise subset"
            )
        result = self.refs[node]
        self.storage_groups[result.name] = group
        self.fresh_storage_groups.discard(result.name)
        versions = {source.name: result, result.name: result}

        def version(ref: _TensorRef) -> _TensorRef:
            if self.storage_groups[ref.name] != group:
                return ref
            if ref.name not in versions:
                while f"updated_view_{self.view_number}" in self.names:
                    self.view_number += 1
                name = f"updated_view_{self.view_number}"
                self.names.add(name)
                self.view_number += 1
                versions[ref.name] = self.add_view(
                    name, result, ref.sizes, ref.strides, ref.dtype
                )
            return versions[ref.name]

        # Existing FX names and view aliases refer to the updated logical tensor.
        # Earlier IR reads retain their old versions; no input buffer is written.
        self.refs = {key: version(ref) for key, ref in self.refs.items()}
        self.tuple_refs = {
            key: [version(ref) for ref in refs] for key, refs in self.tuple_refs.items()
        }

    # -- views (wrapper-representable aliases only) --

    def try_view(self, node: Node) -> bool:
        if node.target not in self._view_targets_set():
            return False
        if len(node.args) < 1 or not isinstance(node.args[0], Node):
            raise LoopsIrUnsupportedError("view source must be a lowered tensor")
        source = self.ref(node.args[0])
        fake = self._node_result_fakes(node)
        if isinstance(fake, (list, tuple)):
            raise LoopsIrUnsupportedError("view result must be a single tensor")
        sizes, strides, dtype = _boundary_spec(fake)
        if dtype != source.dtype:
            # Views must not change dtype; conversions stay eager extern.
            return False
        if node.target == torch.ops.aten.permute.default:
            try:
                dimensions = list(node.args[1])
            except Exception:
                raise LoopsIrUnsupportedError("invalid permutation") from None
            rank = len(source.sizes)
            if any(
                type(axis) is not int or not -rank <= axis < rank for axis in dimensions
            ):
                raise LoopsIrUnsupportedError("invalid permutation")
            axes = [axis % rank for axis in dimensions]
            if sorted(axes) != list(range(rank)):
                raise LoopsIrUnsupportedError("invalid permutation")
            expect_sizes = tuple(source.sizes[axis] for axis in axes)
            expect_strides = tuple(source.strides[axis] for axis in axes)
            if sizes != expect_sizes or strides != expect_strides:
                raise LoopsIrUnsupportedError("permutation metadata mismatch")
        elif node.target == torch.ops.aten.expand.default:
            expanded = self.broadcast(source, sizes)
            if expanded.strides != strides:
                raise LoopsIrUnsupportedError("expand metadata mismatch")
            self.refs[node] = expanded
            return True
        elif node.target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int):
            # t/transpose are pure permutations; fake metadata is authoritative.
            # No hidden copies: alias with the exact reported sizes/strides.
            pass
        elif node.target in (
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
        ):
            # reshape/view may copy in eager for incompatible layouts. Only
            # represent as an alias when the reported shape fits inside the
            # source buffer span (bounds-checked by the verifier); otherwise
            # fall back to an exact eager extern so a copy is never hidden.
            span = _storage_elements(source.sizes, source.strides)
            need = _storage_elements(sizes, strides)
            if need > span:
                return False
        else:
            return False
        self.refs[node] = self.add_view(node.name, source, sizes, strides, dtype)
        return True

    # -- structured eager extern --

    def extern_call(self, node: Node) -> None:
        target = node.target
        if not isinstance(target, torch._ops.OpOverload):
            raise LoopsIrUnsupportedError(f"unsupported operation {node.op}: {target}")
        _reject_unsafe_target(target)
        dotted = _qualified_target_name(target)
        names = self.tensor_names()
        arguments = [_to_call_arg(self.cpp, arg, names) for arg in node.args]
        keywords = [
            self.cpp.CallKeyword(key, _to_call_arg(self.cpp, val, names))
            for key, val in node.kwargs.items()
        ]
        fake = self._node_result_fakes(node)
        if isinstance(fake, (list, tuple)):
            fakes: list[Any] = list(fake)
        else:
            fakes = [fake]
        results: list[Any] = []
        descriptors: list[Any] = []
        out_refs: list[_TensorRef] = []
        for index, single in enumerate(fakes):
            if not isinstance(single, torch.Tensor):
                raise LoopsIrUnsupportedError(
                    "extern outputs must be tensors (no data-dependent/plain values)"
                )
            sizes, strides, dtype = _boundary_spec(single)
            # Dots cannot collide with FX node identifiers such as foo_1.
            buf_name = node.name if len(fakes) == 1 else f"{node.name}.result{index}"
            desc_name = "result" if len(fakes) == 1 else f"result{index}"
            buf = self.cpp.Buffer(buf_name, dtype, _storage_elements(sizes, strides))
            results.append(buf)
            ref = _TensorRef(buf_name, sizes, strides, False, dtype)
            out_refs.append(ref)
            self.storage_groups[ref.name] = ref.name
            schema = getattr(target, "_schema", None)
            if (
                target.namespace == "aten"
                and schema is not None
                and all(ret.alias_info is None for ret in schema.returns)
            ):
                self.fresh_storage_groups.add(ref.name)
            descriptors.append(
                self.cpp.Descriptor(desc_name, buf_name, list(sizes), list(strides), 0)
            )
        # Input descriptors: one per distinct tensor input, sizes/strides from
        # the lowered ref (data_ptr-relative offset 0).
        seen: set[str] = set()
        order_index = 0
        for arg in _walk_tensor_names(arguments, keywords):
            if arg in seen:
                continue
            seen.add(arg)
            ref = self._find_ref_by_name(arg)
            self.escaped_storage_groups.add(self.storage_groups[ref.name])
            descriptors.append(
                self.cpp.Descriptor(
                    f"arg{order_index}",
                    ref.name,
                    list(ref.sizes),
                    list(ref.strides),
                    0,
                )
            )
            order_index += 1
        call = self.cpp.ExternCall.structured(
            node.name,
            dotted,
            arguments,
            keywords,
            results,
            descriptors,
            "unknown",
            True,
        )
        self.program.add_extern_call(call)
        if isinstance(fake, (list, tuple)):
            self.tuple_refs[node] = out_refs
        else:
            self.refs[node] = out_refs[0]

    def _find_ref_by_name(self, name: str) -> _TensorRef:
        for ref in self.refs.values():
            if ref.name == name:
                return ref
        for refs in self.tuple_refs.values():
            for ref in refs:
                if ref.name == name:
                    return ref
        raise LoopsIrUnsupportedError(f"unknown tensor reference '{name}'")

    # -- getitem ref selection (no compute) --

    def handle_getitem(self, node: Node) -> None:
        if node.target is not operator.getitem:
            raise LoopsIrUnsupportedError("not a getitem node")
        if len(node.args) != 2 or not isinstance(node.args[0], Node):
            raise LoopsIrUnsupportedError("unsupported getitem form")
        parent = node.args[0]
        index = node.args[1]
        if type(index) is not int:
            raise LoopsIrUnsupportedError("non-constant getitem index")
        if parent not in self.tuple_refs:
            raise LoopsIrUnsupportedError("getitem parent is not a multi-output call")
        outs = self.tuple_refs[parent]
        if not (0 <= index < len(outs)):
            raise LoopsIrUnsupportedError("getitem index out of range")
        self.refs[node] = outs[index]

    # -- driver --

    def lower(self) -> Any:
        aten = torch.ops.aten
        pointwise = {
            aten.add.Tensor: "add",
            aten.add.Scalar: "add",
            aten.mul.Tensor: "mul",
            aten.mul.Scalar: "mul",
            aten.relu.default: "relu",
        }
        inplace_pointwise = {
            aten.add_.Tensor: "add",
            aten.add_.Scalar: "add",
            aten.mul_.Tensor: "mul",
            aten.mul_.Scalar: "mul",
            aten.relu_.default: "relu",
        }
        for node in self.gm.graph.nodes:
            try:
                if node.op == "placeholder":
                    value = next(self.example_inputs)
                    sizes, strides, dtype = _boundary_spec(value)
                    # Static single-device inference: CUDA only, no symbolic.
                    try:
                        device = value.device
                    except Exception:
                        device = None
                    if device is None or getattr(device, "type", None) != "cuda":
                        raise LoopsIrUnsupportedError(
                            "executable lowering requires static CUDA tensors"
                        )
                    ref = _TensorRef(node.name, sizes, strides, False, dtype)
                    self.program.add_input(self.buffer_for(ref))
                    self.program.add_input_descriptor(self.descriptor(node.name, ref))
                    self.refs[node] = ref
                    self.storage_groups[ref.name] = ref.name
                elif node.op == "call_function" and node.target is operator.getitem:
                    self.handle_getitem(node)
                elif node.op == "call_function" and node.target in inplace_pointwise:
                    self.inplace_pointwise(node, inplace_pointwise[node.target])
                elif node.op == "call_function" and node.target in pointwise:
                    lowered = self.pointwise(node, pointwise[node.target])
                    if not lowered:
                        # Exact eager fallback (alpha!=1 add, non-f32, etc.).
                        self.extern_call(node)
                elif (
                    node.op == "call_function"
                    and node.target in self._view_targets_set()
                ):
                    if not self.try_view(node):
                        self.extern_call(node)
                elif node.op == "call_function" and isinstance(
                    node.target, torch._ops.OpOverload
                ):
                    # Reductions (Tag.reduction) and every other pure op share
                    # the explicit structured extern path. Unsafe ops raise.
                    self.extern_call(node)
                elif node.op == "output":
                    from torch.utils._pytree import tree_leaves

                    for number, value in enumerate(tree_leaves(node.args[0])):
                        if not isinstance(value, Node):
                            raise LoopsIrUnsupportedError(
                                "non-tensor graph output rejected"
                            )
                        self.program.add_output(
                            self.descriptor(f"output{number}", self.ref(value))
                        )
                else:
                    raise LoopsIrUnsupportedError(
                        f"unsupported operation {node.op}: {node.target}"
                    )
            except (LoopsIrUnsupportedError, StopIteration) as error:
                raise LoopsIrUnsupportedError(
                    f"FX node {node.name} ({node.target}): {error}"
                ) from error
        try:
            next(self.example_inputs)
        except StopIteration:
            pass
        else:
            raise LoopsIrUnsupportedError("more example inputs than FX placeholders")
        self.program.verify()
        return self.program


def _walk_tensor_names(arguments: list[Any], keywords: list[Any]) -> list[str]:
    """Ordered tensor refs from structured args (repeats kept for order)."""
    out: list[str] = []

    def visit(arg: Any) -> None:
        try:
            kind = arg.kind_name
        except Exception:
            return
        if kind == "tensor":
            out.append(arg.tensor_name)
        elif kind in ("list", "tuple"):
            for item in arg.items:
                visit(item)

    for arg in arguments:
        visit(arg)
    for kw in keywords:
        try:
            visit(kw.value)
        except Exception:
            continue
    return out


def lower_fx_to_cpp(
    gm: GraphModule,
    example_inputs: list[Any] | tuple[Any, ...],
    *,
    executable: bool = False,
) -> Any:
    """Lower FX directly to a C++ Program, without existing Inductor loop IR.

    Default (diagnostic) preserves the historic subset and error behavior.
    ``executable=True`` uses the strict inference lowerer (f32 pointwise
    Compute; reductions + other pure ops as structured eager externs). It is
    an explicit argument, never a global config mutation.
    """
    if executable:
        return _ExecutableLowerer(gm, example_inputs).lower()
    return _Lowerer(gm, example_inputs).lower()


def print_cpp_loops_ir(
    gm: GraphModule, example_inputs: list[Any] | tuple[Any, ...]
) -> None:
    """Strict opt-in hook; verification finishes before anything is printed."""
    program = lower_fx_to_cpp(gm, example_inputs)
    print(program.dump())
