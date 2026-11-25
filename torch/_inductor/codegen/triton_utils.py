# mypy: allow-untyped-defs
import functools
import re
import warnings
from typing import Any

import sympy

import torch
from torch._guards import CompileContext, TracingContext
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    StrictMinMaxConstraint,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Mod
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._triton import has_triton_block_ptr

from .. import config, ir
from ..runtime.hints import AttrsDescriptorWrapper, DeviceProperties
from ..utils import (
    _type_of,
    device_supports_fp64,
    expr_fits_within_32bit,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    KernelArgType,
    SizeArg,
    TensorArg,
    TMADescriptorArg,
    WorkspaceArg,
)


@functools.cache
def _warn_block_ptr_unavailable() -> None:
    # Fires only in the no-op case (flag on, API gone), so users who never set
    # the flag -- and callers that patch it to False -- stay quiet.
    warnings.warn(
        "config.triton.use_block_ptr=True but the installed Triton no longer "
        "provides the block-pointer frontend API (removed in "
        "triton-lang/triton#10833); falling back to the default masked-indexing "
        "path. This flag is deprecated and will be removed (#191012).",
        FutureWarning,
        stacklevel=2,
    )


def use_block_ptr_enabled() -> bool:
    """Effective block-pointer codegen setting.

    Honors ``config.triton.use_block_ptr`` only where the installed Triton still
    provides the block-pointer frontend API. When the flag is set but the API is
    gone the request is a no-op: warn once, then fall back to masked indexing.
    """
    if not config.triton.use_block_ptr:
        return False
    if has_triton_block_ptr():
        return True
    _warn_block_ptr_unavailable()
    return False


def should_unwrap_unspec_arg(name: str) -> bool:
    if not V.graph.is_unspec_arg(name):
        return False

    wrapper_code = getattr(V.graph, "wrapper_code", None)
    if getattr(wrapper_code, "preserve_zero_dim_tensor_args", False):
        return False

    # Unwrap on all devices except CPU
    if V.graph.get_current_device_or_throw().type != "cpu":
        return True
    # Only unwrap on CPU if the input is not used as an output.
    return name not in V.graph.mutated_buffers


def use_uint8_triton_storage_for_cuda_float8_e4m3fn(
    dtype: torch.dtype,
    arg_name: str | None = None,
    *,
    device: torch.device | None = None,
) -> bool:
    # Triton rejects fp8e4nv pointer types before sm89, but eager CUDA can
    # still dequantize float8_e4m3fn values by treating storage as raw bytes.
    if dtype != torch.float8_e4m3fn or torch.version.hip is not None:
        return False
    if arg_name is not None and not arg_name.startswith("in_ptr"):
        return False

    if device is None:
        try:
            device = V.graph.get_current_device_or_throw()
        except AttributeError:
            return False

    if device.type != "cuda":
        return False

    return DeviceProperties.create(device).cc < 89


def triton_meta_device_props(device: torch.device) -> DeviceProperties:
    """DeviceProperties for a kernel's triton_meta, respecting compile-on-one-rank.

    Under CooR the rank-specific index is dropped so a kernel's triton_meta -- hence
    its cache key and its generated source -- is byte-identical across ranks; the
    launcher resolves the real device at load time.

    NB: torch.device("cuda").index is None, not 0, and that None is what reaches
    DeviceProperties. The None is load-bearing, not cosmetic: in triton_heuristics.py
    it is the sentinel _resolve_load_device keys on to pick the running rank's current
    device, and what sets kernel.device_agnostic -- which in turn forces the per-device
    static launcher instead of the fast launcher, whose single baked function pointer
    would be for the wrong GPU. Baking index=0 on every rank would make the source
    byte-identical yet still launch on rank 0's device, so a byte-identity check alone
    does not catch it.

    Every site that builds triton_meta must go through here. Applying the rule in
    TritonKernel alone left Triton templates (select_algorithm.py) and combo kernels
    (triton_combo_kernel.py) baking DeviceProperties(index=N), which makes their
    generated code differ across ranks.
    """
    from torch.fx.experimental.proxy_tensor import _coor_enabled

    if _coor_enabled():
        device = torch.device(device.type)
    return DeviceProperties.create(device)


def signature_of(
    arg: KernelArgType,
    *,
    size_dtype: str | None,
    use_fp64_for_python_float: bool = True,
) -> str:
    """Return the Triton signature type for an Inductor kernel argument."""
    if isinstance(arg, TensorArg):
        typ = _type_of(arg.dtype)
        if should_unwrap_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_typ = typ.lstrip("*")
            if new_typ in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_typ
        elif use_uint8_triton_storage_for_cuda_float8_e4m3fn(arg.dtype, arg.name):
            return "*u8"
        else:
            return typ
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            if triton_version_uses_attrs_dict():
                # In newer versions of Triton, the signature includes "None" args
                # and their type is marked as "constexpr"
                return "constexpr"
            else:
                # In older versions of Triton...
                # From triton/runtime/jit.py
                # `None` is nullptr.  Implicitly convert to *i8.
                return "*i8"
        elif _arg_equals_1(arg) and triton_version_uses_attrs_dict():
            # In new versions of Triton, if we have an equal-to-1 arg that's marked as a constant,
            # it should be marked as "constexpr" in the signature.
            return "constexpr"
        elif isinstance(arg.expr, (float, sympy.Float)):
            # Inductor-generated kernels use fp64 to preserve Python-float
            # precision. User-defined Triton kernels opt out so their compiled
            # signatures match Triton's eager specialization.
            if (
                use_fp64_for_python_float
                and config._use_fp64_for_unbacked_floats
                and device_supports_fp64(V.graph.current_device)
            ):
                return "fp64"
            return "fp32"
        elif isinstance(arg.expr, sympy.Symbol) and symbol_is_type(
            arg.expr, (SymT.UNBACKED_FLOAT)
        ):
            # Unbacked floats from .item() are runtime Python floats, so they
            # follow the same eager-vs-Inductor signature policy as literals.
            if (
                use_fp64_for_python_float
                and config._use_fp64_for_unbacked_floats
                and device_supports_fp64(V.graph.current_device)
            ):
                return "fp64"
            return "fp32"
        elif isinstance(arg.expr, bool):
            return "i1"

        # if this is an integer
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        elif size_dtype is None:
            # no hint: we'll see if we know that this is a 32-bit int, and guard if possible.
            int_max = torch.iinfo(torch.int32).max
            if expr_fits_within_32bit(arg.expr):
                V.graph.sizevars.check_leq(arg.expr, int_max)
                return "i32"
            else:
                return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return _type_of(arg.dtype)
    if isinstance(arg, TMADescriptorArg):
        if arg.api_type == "experimental":
            return "nvTmaDesc"
        else:
            # https://github.com/triton-lang/triton/blob/9695baed9b46cf957e08b157bb4133f4a4b331c5/python/triton/runtime/jit.py#L360-L363
            if arg.api_type != "stable":
                raise AssertionError(
                    f"expected api_type == 'stable', got {arg.api_type}"
                )
            if arg.block_shape is None:
                raise AssertionError("expected block_shape to not be None")
            if arg.dtype is None:
                raise AssertionError("expected dtype to not be None")
            inner = _type_of(arg.dtype)[1:]  # strip the `*`: *fp32 -> fp32
            return f"tensordesc<{inner}{list(arg.block_shape)}>"
    if isinstance(arg, ConstexprArg):
        return "constexpr"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def non_constexpr_signature(signature):
    new_signature = []
    for arg in signature:
        if not isinstance(arg, ConstexprArg):
            new_signature.append(arg)

    return new_signature


def select_tile_hint(size_hints, signature):
    """Pick TileHint.SQUARE vs TileHint.DEFAULT for 2D pointwise; None for 1D.

    SQUARE applies when the kernel has exactly 2 size hints and 4 non-constexpr
    signature args (input, output, and 2 numel args -> a square 2D tile).
    """
    from torch._inductor.runtime.hints import TileHint

    if len(size_hints) != 2:
        return None
    if len(non_constexpr_signature(signature)) == 4:
        return TileHint.SQUARE
    return TileHint.DEFAULT


def signature_to_meta(
    signature: list[KernelArgType],
    *,
    size_dtype: str | None,
    argdefs: list[ArgName],
    indices: list[int] | None = None,
    is_template: bool = False,
    use_fp64_for_python_float: bool = True,
) -> dict[str, str]:
    if indices is None:
        indices = list(range(len(signature)))

    def _decide_tl_dtype(arg):
        # Even if the ks0 symbol itself is within tl.int32 range, it's
        # risky to use tl.int32 dtype since we may have ks0*ks1 later
        # for kernels like torch.mean when dynamic shape is enabled.
        #
        # Block pointers do not support 64-bit indexing, so keep ks indices in
        # tl.int32 whenever the (deprecated) block-pointer path is actually
        # active. Templates like flex attention/decoding likewise use
        # hand-written block pointers, so they also stay on 32-bit ks indexing.
        #
        # assume_32bit_indexing already asserts (and guards) that every ks* symbol
        # fits in int32.
        if (
            not is_template
            and not use_block_ptr_enabled()
            and not config.assume_32bit_indexing
            and isinstance(arg, SizeArg)
            and arg.name.startswith("ks")
        ):
            return "tl.int64"
        return size_dtype

    return {
        argdefs[i].name: signature_of(
            arg,
            size_dtype=_decide_tl_dtype(arg),
            use_fp64_for_python_float=use_fp64_for_python_float,
        )
        for i, arg in zip(indices, signature)
    }


def _get_buffer_layout(buf_name: str) -> "torch._inductor.ir.Layout":
    """Get the layout for a buffer, handling both scheduler buffers and graph inputs."""
    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.try_get_buffer(buf_name)
        # output arg
        if not buffer:
            if buf_name != V.kernel.output_node.name:
                raise AssertionError(
                    f"expected buf_name == output_node.name, got {buf_name}"
                )
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()
    return layout


def is_unaligned_buffer_name(buf_name: str) -> bool:
    if buf_name in V.graph.unaligned_buffers:
        return True

    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        # For graph inputs that is not recorded in V.graph.unaligned_buffers,
        # we know for sure the tensor is aligned.
        return False

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    layout = _get_buffer_layout(buf_name)
    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def is_unaligned_buffer(arg: TensorArg):
    return is_unaligned_buffer_name(arg.buffer)


def _arg_equals_1(arg: KernelArgType) -> bool:
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )


def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
) -> tuple[int, ...]:
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1


def _is_tensor_within_2gb(arg: TensorArg) -> bool:
    """Check if a tensor argument's storage is provably within 2GB.

    Mirrors HIPBackend.is_within_2gb() using constant storage sizes or
    compile-time symbolic layouts. This enables canonicalize_pointers to
    decompose pointer arithmetic into (splat(base), offset) form for buffer ops.
    """
    MAX_BYTES = 2**31 - 1
    try:
        if arg.buffer in V.graph.constants:
            # A small view can share a backing allocation larger than 2GB.
            storage_bytes = V.graph.constants[arg.buffer].untyped_storage().nbytes()
        else:
            # Graph inputs aren't tracked by the scheduler; get their layout
            # from the graph_inputs dict to avoid KeyError in get_buffer_layout.
            if arg.buffer in V.graph.graph_inputs:
                inp = V.graph.graph_inputs[arg.buffer]
                if hasattr(inp, "get_layout"):
                    layout = inp.get_layout()
                else:
                    return False
            else:
                layout = _get_buffer_layout(arg.buffer)
            storage_bytes = layout.storage_size() * arg.dtype.itemsize
        return V.graph.sizevars.statically_known_true(storage_bytes <= MAX_BYTES)
    except Exception:
        return False


# Dynamo frame_id -> frame_compile_id of the compile that guarded on divisibility.
# Later compiles of the frame don't, so a size that breaks the guards costs at most
# one recompile rather than one per guarded expression. torch._dynamo.reset()
# clears it, along with the frame ids.
_divisibility_guarded_compile: dict[int, int] = {}

# Kernel args that are a range tree's numel, where divisibility lets loads and
# stores vectorize; strides and other sizes would only add recompiles.
_NUMEL_ARG = re.compile(r"(?:[xyz]|r\d+_)numel(?:_\d+)?")


def _strict_min_max_constrained_symbols(shape_env: Any) -> OrderedSet[sympy.Symbol]:
    result: OrderedSet[sympy.Symbol] = OrderedSet()
    for tracked in shape_env.tracked_fakes or ():
        if not isinstance(tracked.fake, torch.Tensor):
            continue
        context = tracked.symbolic_context
        for dims, constraints in (
            (tracked.fake.size(), getattr(context, "constraint_sizes", None)),
            (tracked.fake.stride(), getattr(context, "constraint_strides", None)),
        ):
            for dim, constraint in zip(dims, constraints or ()):
                if isinstance(constraint, StrictMinMaxConstraint) and isinstance(
                    dim, torch.SymInt
                ):
                    result.update(shape_env.replace(dim.node.expr).free_symbols)
    return result


def size_divisibility_guards_may_be_skipped() -> bool:
    """
    Whether nothing may check a guard this compile adds, or a failed one can't
    recompile.
    """
    from torch._dynamo import config as dynamo_config, eval_frame

    compile_context = CompileContext.try_get()
    return (
        (compile_context is not None and compile_context.guards_may_be_skipped)
        or eval_frame._stance.skip_guard_eval_unsafe
        or eval_frame._stance.stance in ("fail_on_recompile", "eager_on_recompile")
        or dynamo_config.error_on_recompile
    )


def _divisibility_guard_frame(shape_env: Any) -> tuple[int, int] | None:
    """
    (frame_id, frame_compile_id) of the Dynamo frame compile we're in, which turns
    a failed guard into a recompile, or None if there's no such compile.
    """
    compile_context = CompileContext.try_get()
    if compile_context is None:
        return None
    compile_id = compile_context.compile_id
    tracing_context = TracingContext.try_get()
    if (
        compile_id is None
        or compile_id.frame_id is None
        or compile_id.frame_compile_id is None
        or compile_id.compiled_autograd_id is not None
        # a ShapeEnv of its own
        or tracing_context is None
        or tracing_context.fake_mode is None
        or tracing_context.fake_mode.shape_env is not shape_env
        # Dynamo has already built its guards, e.g. for a backward compiled lazily
        # on first use, and would drop this one
        or shape_env.frozen
        or size_divisibility_guards_may_be_skipped()
    ):
        return None
    from torch._inductor.compile_fx import fx_compile_mode, FxCompileMode

    # a serialized compile guards on a copy of the ShapeEnv
    if fx_compile_mode != FxCompileMode.NORMAL:
        return None
    return compile_id.frame_id, compile_id.frame_compile_id


def may_add_size_divisibility_guards(shape_env: Any = None) -> bool:
    """
    Whether this compile (lowering with shape_env, by default the tracing context's)
    may add divisibility guards. Part of the FX graph cache key: a hit re-adds the
    cached graph's guards, which a compile that wouldn't have added them may not be
    able to keep or check. Coarse on purpose, so the key can be computed before
    lowering: a min/max range or hint_override on any symbol turns guards off for
    the whole graph, and the per-frame record makes the key depend on this
    process's compile history.
    """
    if shape_env is None:
        tracing_context = TracingContext.try_get()
        if tracing_context is None or tracing_context.fake_mode is None:
            return False
        shape_env = tracing_context.fake_mode.shape_env
    if shape_env is None or (frame := _divisibility_guard_frame(shape_env)) is None:
        return False
    frame_id, frame_compile_id = frame
    # compile ids only grow within a frame, so a larger one is from before a reset
    guarded = _divisibility_guarded_compile.get(frame_id)
    return (
        (guarded is None or guarded >= frame_compile_id)
        # a mark_dynamic hint_override is a hint, not the input's size, so a guard
        # built from it can fail on the very call being compiled
        and not shape_env.var_to_hint_override
        # a user-given range (mark_dynamic min/max) turns any guard on it into an
        # error
        and not _strict_min_max_constrained_symbols(shape_env)
    )


def note_size_divisibility_guards_loaded() -> None:
    """
    A cached graph brought its divisibility guards; count its compile as the one
    that guards, as _guard_multiple_of would have.
    """
    tracing_context = TracingContext.try_get()
    if tracing_context is None or tracing_context.fake_mode is None:
        return
    shape_env = tracing_context.fake_mode.shape_env
    if shape_env is None or (frame := _divisibility_guard_frame(shape_env)) is None:
        return
    frame_id, frame_compile_id = frame
    _divisibility_guarded_compile.setdefault(frame_id, frame_compile_id)


def _wrapper_bound_symbols(graph: Any) -> OrderedSet[sympy.Symbol]:
    """Symbols the wrapper reads straight off its inputs."""
    result: OrderedSet[sympy.Symbol] = OrderedSet()
    for value in graph.graph_inputs.values():
        if isinstance(value, sympy.Symbol):
            result.add(value)
        elif isinstance(value, ir.TensorBox):
            for dim in (*value.get_size(), *value.get_stride()):
                if isinstance(dim, sympy.Symbol):
                    result.add(dim)
    return result


def _guard_multiple_of(expr: sympy.Expr, alignment: int) -> bool:
    """
    With dynamic shapes a symbolic numel can't be proven divisible by 16, so Triton
    masks element by element and can't vectorize. If a backed numel is a multiple of
    alignment at its hint, guard on it staying one (a call that breaks the guard
    recompiles) and tell Triton. The wrapper checks it again on every call, in case
    something skipped the guard.
    """
    if (
        config.unsafe_skip_cache_dynamic_shape_guards
        or not isinstance(expr, sympy.Expr)
        or not expr.is_integer
        or not expr.free_symbols
        # AOTI: nothing checks guards when the compiled model runs
        or V.graph.aot_mode
    ):
        return False
    # the outermost graph's wrapper runs the check
    root = V.graph
    while (parent := getattr(root, "parent", None)) is not None:
        root = parent
    wrapper = getattr(root, "wrapper_code", None)
    if wrapper is None or not wrapper.supports_size_divisibility_checks:
        return False
    if (may_add := root.may_add_size_divisibility_guards) is None:
        may_add = root.may_add_size_divisibility_guards = (
            may_add_size_divisibility_guards(V.graph.sizevars.shape_env)
        )
    if not may_add:
        return False
    sizevars = V.graph.sizevars
    # see through precomputed sizes (ps0 = u0 * s1) to the symbols they use
    expr = sizevars.remove_precomputed_replacements(expr)
    if free_unbacked_symbols(expr):
        return False
    hint = sizevars.guarding_hint_or_throw(expr)
    if hint == 0 or hint % alignment != 0:
        return False
    # the wrapper's check can only read symbols it binds from its inputs
    if (bound := root.wrapper_bound_symbols) is None:
        bound = root.wrapper_bound_symbols = _wrapper_bound_symbols(root)
    if not expr.free_symbols <= bound:
        return False
    frame = _divisibility_guard_frame(sizevars.shape_env)
    if frame is None or not sizevars.guard_or_false(sympy.Eq(Mod(expr, alignment), 0)):
        return False
    frame_id, frame_compile_id = frame
    _divisibility_guarded_compile[frame_id] = frame_compile_id
    root.size_divisibility_guards.add((expr, alignment))
    return True


def config_of(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
    pointer_range_override: tuple[int, ...] | None = None,
    skip_cpp_wrapper_input_tensor_alignment: bool = False,
    guard_size_divisibility: bool = False,
) -> Any:
    """
    guard_size_divisibility: for Inductor's own kernels, also mark divisible the
    symbolic numels _guard_multiple_of can guard on.
    """
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/triton-lang/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize,
                    alignment,  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, (float, bool)):
                return False
            if V.graph.sizevars.statically_known_multiple_of(x.expr, alignment):  # type: ignore[arg-type]
                return True
            return (
                guard_size_divisibility
                and _NUMEL_ARG.fullmatch(x.name) is not None
                and _guard_multiple_of(x.expr, alignment)
            )
        if isinstance(x, WorkspaceArg):
            # We allocate the workspace ourselves, so it is always aligned
            return True
        if isinstance(x, (TMADescriptorArg, ConstexprArg)):
            return False
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    def include_tensor_alignment(arg: KernelArgType) -> bool:
        if (
            not skip_cpp_wrapper_input_tensor_alignment
            or not V.graph.cpp_wrapper
            or V.graph.aot_mode
            or not isinstance(arg, TensorArg)
            or arg.buffer not in V.graph.graph_inputs
        ):
            return True

        try:
            input_idx = V.graph.graph_input_names.index(arg.buffer)
        except ValueError:
            return True
        return input_idx not in (V.graph.inputs_to_check or ())

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(
                arg,
                alignment=16,
                include_tensor=include_tensor_alignment(arg),
            )
        )
    else:
        divisible_by_16 = ()

    equal_to_1 = equal_1_arg_indices(args, indices=indices)

    # On AMD/HIP, tag tensor args whose storage fits in 2GB so Triton
    # can use 32-bit pointer offsets and emit buffer load/store ops.
    if pointer_range_override is not None:
        pointer_range_32 = pointer_range_override
    elif torch.version.hip is not None and config.triton.emit_pointer_range_32:
        pointer_range_32 = tuple(
            i
            for i, arg in zip(indices, args)
            if isinstance(arg, TensorArg) and _is_tensor_within_2gb(arg)
        )
    else:
        pointer_range_32 = ()

    # pyrefly: ignore [bad-argument-count, bad-argument-type]
    return AttrsDescriptorWrapper(divisible_by_16, equal_to_1, pointer_range_32)
