import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import logging
import operator
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import Never

import sympy

import torch.fx as fx
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import redirect_to_mode
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import guard_scalar
from torch.types import IntLikeType
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


if TYPE_CHECKING:
    from triton._C.libtriton.ir import (
        module as TritonIRModule,
        operation as TritonIROperation,
    )

    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.constant import ConstantVariable
    from torch._dynamo.variables.functions import TritonKernelVariable
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
    from torch.fx.proxy import Proxy
    from torch.utils._triton import has_triton

    TritonMetaParamsType = dict[str, int]
    TritonGridTupleType = tuple[Union[int, sympy.Expr, SymInt], ...]
    TritonGridCallableType = Callable[[TritonMetaParamsType], tuple[int, ...]]
    TritonGridType = Union[TritonGridTupleType, TritonGridCallableType]

    if has_triton():
        from triton.runtime.autotuner import Autotuner, Config as TritonConfig
        from triton.runtime.jit import JITFunction
    else:

        class Autotuner:  # type: ignore[no-redef]
            pass

        class JITFunction:  # type: ignore[no-redef]
            pass

    TritonKernelType = Union[Autotuner, JITFunction]
    # mypy specifically complains that TritonAutotunerType is not a valid type if Autotuner is not inside of a Union.
    TritonAutotunerType = Union[Autotuner]

log = logging.getLogger("torch._dynamo")

# e.g. for a host-side Triton TMA API call ``create_2d_tma_descriptor(ptr, 50, 60, 32, 15, 4)``,
# the metadata will look like ``("experimental", ([50, 60], [32, 15], 4))``
TMAExperimentalMetadata = tuple[
    str,  # type of TMA (should be "experimental")
    tuple[
        list[IntLikeType],  # dims
        list[IntLikeType],  # block_dims
        IntLikeType,  # element_size
    ],
]

# e.g. for host-side Triton TMA API call ``TensorDescriptor.from_tensor(ptr, [32, 64])``
# the metadata will look like ``("stable", ([32, 64],))``
TMAStableMetadata = tuple[
    str,  # type of TMA ("experimental" or "stable")
    tuple[list[IntLikeType],],  # block_shape
]


def create_tma_experimental_metadata(
    dims: list[IntLikeType],
    block_dims: list[IntLikeType],
    element_size: IntLikeType,
) -> TMAExperimentalMetadata:
    return ("experimental", (dims, block_dims, element_size))


def maybe_unpack_tma_experimental_metadata(
    tma_meta: Union[TMAExperimentalMetadata, TMAStableMetadata],
) -> Optional[tuple[list[IntLikeType], list[IntLikeType], IntLikeType]]:
    if not tma_meta or len(tma_meta) != 2:
        return None
    if tma_meta[0] == "experimental":
        return tma_meta[1]  # type: ignore[return-value]
    return None


def create_tma_stable_metadata(
    block_shape: list[IntLikeType],
) -> TMAStableMetadata:
    return ("stable", (block_shape,))


def maybe_unpack_tma_stable_metadata(
    tma_meta: Union[TMAExperimentalMetadata, TMAStableMetadata],
) -> Optional[tuple[list[IntLikeType]]]:
    if not tma_meta or len(tma_meta) != 2:
        return None
    if tma_meta[0] == "stable":
        return tma_meta[1]  # type: ignore[return-value]
    return None


# TMADescriptorMetadata maps kernel parameter names to the metadata that allows
# reconstructing TMA descriptors from the underlying tensors (passed as kernel
# arguments in the fx graph, instead of the TMA descriptors).
#
# Since there are two TMA APIs (the old "experimental" API and the new "stable" API),
# each entry in the dict is a tuple that starts with a string, either "experimental"
# or "stable". The second entry in the tuple is another tuple, with data that depends
# on the API type (see TMAExperimentalMetadata and TMAStableMetadata above).
#
# These are stored as raw tuples (instead of classes) for ease of serialization.
TMADescriptorMetadata = dict[
    str,  # kernel parameter name
    Union[TMAExperimentalMetadata, TMAStableMetadata],
]


###############################################################################
# Kernel Side Table


# We cannot put Triton Kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable:
    id_to_kernel: dict[int, "TritonKernelType"] = {}
    kernel_to_id: dict["TritonKernelType", int] = {}
    constant_args: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()

    # Returns index on the table
    def add_kernel(self, kernel: "TritonKernelType") -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the triton kernel at the given index
    def get_kernel(self, idx: int) -> "TritonKernelType":
        # No need to lock here as fetching from dict is atomic
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    # Not every constant arg can be added to the graph. Use this side table
    # for constant args.
    def add_constant_args(self, args: dict[str, Any]) -> int:
        with self.lock:
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            return idx

    # Returns the constant args
    def get_constant_args(self, idx: int) -> dict[str, Any]:
        # No need to lock here as fetching from dict is atomic
        assert idx in self.constant_args
        return self.constant_args[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.constant_args = {}


kernel_side_table = KernelSideTable()


###############################################################################
# Mutation Tracker


@dataclasses.dataclass(frozen=True)
class Param:
    idx: int


@dataclasses.dataclass(frozen=True)
class Intermediate:
    idx: int

    def fake(self) -> bool:
        return self.idx < 0


@dataclasses.dataclass(frozen=True)
class Op:
    name: str
    fn_call_name: Optional[str]
    args: list[Union[Param, Intermediate]]
    ret: Intermediate = dataclasses.field(repr=False)
    # used for scf.yield: see [Note: scf.yield fix-up]
    sub_idx: Optional[int] = None
    # used for tt.elementwise_inline_asm
    # `is_pure = True` assumes the asm block has no side-effects
    is_pure: bool = False

    def __post_init__(self) -> None:
        if self.name == "tt.call":
            assert self.fn_call_name is not None
        else:
            assert self.fn_call_name is None


def generate_ttir(
    kernel: "TritonKernelType",
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> tuple["TritonIRModule", list[str]]:
    """
    Uses Triton's internal code generation to create TTIR
    """
    import sympy
    import triton
    import triton.runtime.jit
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    from torch._inductor.utils import (
        get_triton_attrs_descriptor_version,
        triton_version_uses_attrs_dict,
        TritonAttrsDescriptorVersion,
    )
    from torch.utils._triton import has_triton_tensor_descriptor_host_tma

    triton_version = get_triton_attrs_descriptor_version()

    import torch._inductor.ir
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            # If we are autotuning, then it doesn't matter which version gets
            # picked for tracing purposes, so lets pick the first one
            kwargs = {**kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    assert isinstance(kernel, JITFunction)

    # pyrefly: ignore  # missing-attribute
    context = triton._C.libtriton.ir.context()
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({})

    # ignore backend-specific kwargs same way as in the native Triton code
    # https://github.com/triton-lang/triton/blob/a6bb57d6285e723c58e87dd7cba263db6efff789/python/triton/runtime/jit.py#L594-L596
    # why this is important for user-defined Triton kernels on AMD: https://github.com/pytorch/pytorch/issues/140800
    for name in list(kwargs):
        if name not in kernel.arg_names and name in options.__dict__:
            kwargs.pop(name)

    if len(kwargs) != len(kernel.arg_names):
        raise ValueError(
            "Incorrect number of arguments passed to kernel: "
            f"passed {list(kwargs.keys())}, expected {kernel.arg_names}."
        )

    # Replace all SymExprs with a regular value for TTIR generation
    # Replace all FakeTensor/TensorBox with real tensors
    # These replacements are needed for triton's type, key and config functions
    ordered_args: dict[str, Any] = {}
    for name in kernel.arg_names:
        a = kwargs[name]
        if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
            ordered_args[name] = 2
        elif (
            stable_meta := maybe_unpack_tma_stable_metadata(
                # pyrefly: ignore [bad-argument-type]
                tma_descriptor_metadata.get(name, None)
            )
        ) is not None:
            from triton.tools.tensor_descriptor import TensorDescriptor

            block_shape = stable_meta[0]
            with torch._C._DisableTorchDispatch():
                # need 16-byte aligned strides
                elements_per_dim = max(1, 16 // a.dtype.itemsize)
                base_tensor = torch.empty(
                    [elements_per_dim] * len(block_shape), dtype=a.dtype
                )
            # pyrefly: ignore  # bad-argument-type
            ordered_args[name] = TensorDescriptor.from_tensor(base_tensor, block_shape)
        elif isinstance(a, (FakeTensor, torch._inductor.ir.TensorBox)):
            with torch._C._DisableTorchDispatch():
                ordered_args[name] = torch.empty(2, dtype=a.dtype)
        else:
            ordered_args[name] = a

    def is_stable_tensor_descriptor_arg(arg: Any) -> bool:
        if has_triton_tensor_descriptor_host_tma():
            from triton.tools.tensor_descriptor import TensorDescriptor

            if isinstance(arg, TensorDescriptor):
                return True
        return False

    def is_tensor_like_arg(arg: Any) -> bool:
        if isinstance(arg, Tensor) or is_stable_tensor_descriptor_arg(arg):
            return True
        return False

    # Note: one would expect that each input to the triton kernel maps to
    # one input parameter in the TTIR. This is _not_ true for TMA descriptors:
    # one TMA descriptor gets converted into:
    #   * one TMA descriptor input
    #   * N strides, for a rank-N tensor
    #   * N sizes, for a rank-N tensor
    # To account for this, we inject some fake arg names as placeholders for
    # the stride and size parameters.
    def get_tensor_names(name: str, arg: Any) -> list[str]:
        if isinstance(arg, Tensor):
            return [name]
        if is_stable_tensor_descriptor_arg(arg):
            stable_meta = maybe_unpack_tma_stable_metadata(
                tma_descriptor_metadata[name]
            )
            assert stable_meta is not None
            block_shape = stable_meta[0]
            tensor_rank = len(block_shape)
            names = [name]
            names.extend(name + f" STRIDE PLACEHOLDER {i}" for i in range(tensor_rank))
            names.extend(name + f" SIZE PLACEHOLDER {i}" for i in range(tensor_rank))
            return names
        return []

    ordered_tensor_names = list(
        itertools.chain.from_iterable(
            get_tensor_names(name, arg) for name, arg in ordered_args.items()
        )
    )

    def _get_specialization(args):  # type: ignore[no-untyped-def]
        # Support multiple triton versions.
        # This code basically copies JITFunction.run() logic to get the attrs to construct an ASTSource.
        if triton_version == TritonAttrsDescriptorVersion.V1_COMPILER:
            return kernel._get_config(*args)
        elif triton_version in {
            TritonAttrsDescriptorVersion.V2_BACKENDS,
            TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
        }:
            from triton.backends.compiler import AttrsDescriptor  # noqa: F401

            target = triton.runtime.driver.active.get_current_target()
            backend_ = triton.compiler.compiler.make_backend(target)
            # pyrefly: ignore  # missing-attribute
            return backend_.get_attrs_descriptor(args, kernel.params)
        else:
            assert (
                get_triton_attrs_descriptor_version()
                == TritonAttrsDescriptorVersion.V4_DICT
            )
            # specialize_impl switched to create_specialize_impl in https://github.com/triton-lang/triton/pull/6099
            if hasattr(triton.runtime.jit, "create_specialize_impl"):
                try:
                    # Latest versions of Triton take specialize_extra as an arg to create_specialize_impl
                    specialize_impl = triton.runtime.jit.create_specialize_impl(
                        specialize_extra=backend.get_arg_specialization
                    )
                except TypeError:  # Unknown arg `specialize_extra`
                    # Older versions of Triton take specialize_extra as an arg to specialize_impl
                    specialize_impl = functools.partial(
                        # pyrefly: ignore  # missing-argument
                        triton.runtime.jit.create_specialize_impl(),
                        specialize_extra=backend.get_arg_specialization,
                    )
            # create_specialize_impl is removed in https://github.com/triton-lang/triton/pull/7771
            # switch to native_specialize_impl instead
            elif hasattr(triton.runtime.jit, "native_specialize_impl"):
                from triton.backends import BaseBackend
                from triton.runtime.jit import native_specialize_impl

                def _native_specialize_impl(
                    arg: Any,
                    is_const: bool = False,
                    specialize_value: bool = True,
                    align: bool = True,
                ) -> Callable:
                    return native_specialize_impl(
                        BaseBackend, arg, is_const, specialize_value, align
                    )

                specialize_impl = _native_specialize_impl
            else:
                from triton.runtime.jit import specialize_impl as specialize_impl_orig

                specialize_impl = functools.partial(
                    specialize_impl_orig,
                    specialize_extra=backend.get_arg_specialization,
                )

            from triton._utils import find_paths_if, get_iterable_path

            # logic is copied from: binder = create_function_from_signature(self.signature, self.params, backend)
            attrvals = []
            for arg, kp in zip(args, kernel.params):
                if kp.is_constexpr:
                    attrvals.append(arg)
                else:
                    spec = specialize_impl(
                        arg,
                        is_const=kp.is_const,
                        specialize_value=not kp.do_not_specialize,
                        align=not kp.do_not_specialize_on_alignment,
                    )
                    # pyrefly: ignore [unsupported-operation]
                    attrvals.append(spec[1])

            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {
                k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs
            }
            return attrs

    specialization = _get_specialization(ordered_args.values())
    constants = {
        name: arg for name, arg in ordered_args.items() if not is_tensor_like_arg(arg)
    }

    if (mangle_type := getattr(triton.runtime.jit, "mangle_type", None)) is not None:

        def get_signature_value(idx: int, arg: Any) -> str:
            if kernel.params[idx].is_constexpr:
                return "constexpr"
            # pyrefly: ignore [not-callable]
            return mangle_type(arg)

    else:

        def get_signature_value(idx: int, arg: Any) -> str:
            return kernel._type_of(kernel.key_of(arg))

    if triton_version_uses_attrs_dict():
        # In newer versions of Triton, the signature includes constexpr args
        signature = {
            name: get_signature_value(i, arg)
            for i, (name, arg) in enumerate(ordered_args.items())
        }
    else:
        # In older versions of Triton, the signature does not include constexpr args
        constexprs = [p.num for p in kernel.params if p.is_constexpr]
        signature = {
            name: get_signature_value(i, arg)
            for i, (name, arg) in enumerate(ordered_args.items())
            if i not in constexprs
        }

    # pyrefly: ignore  # missing-attribute
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(kernel, signature, constants, specialization)

    # Triton changes ASTSource.make_ir to take 3/4 arguments. Handle
    # backward compatibility here.
    make_ir_sig_params = len(inspect.signature(src.make_ir).parameters)
    get_codegen_implementation_sig_params = len(
        # pyrefly: ignore  # missing-attribute
        inspect.signature(backend.get_codegen_implementation).parameters
    )
    if make_ir_sig_params == 2:
        # pyrefly: ignore  # missing-argument
        ttir_module = src.make_ir(options, context)
    elif make_ir_sig_params == 3:
        # pyrefly: ignore  # missing-attribute
        codegen_fns = backend.get_codegen_implementation()
        # pyrefly: ignore  # missing-argument
        ttir_module = src.make_ir(options, codegen_fns, context)
    elif make_ir_sig_params == 4:
        codegen_args = [options] if get_codegen_implementation_sig_params == 1 else []
        # pyrefly: ignore  # missing-attribute
        codegen_fns = backend.get_codegen_implementation(*codegen_args)
        module_map = backend.get_module_map()
        # pyrefly: ignore[missing-argument,bad-argument-type]
        ttir_module = src.make_ir(options, codegen_fns, module_map, context)
    else:
        codegen_args = [options] if get_codegen_implementation_sig_params == 1 else []
        # pyrefly: ignore  # missing-attribute
        codegen_fns = backend.get_codegen_implementation(*codegen_args)
        module_map = backend.get_module_map()
        # pyrefly: ignore  # bad-argument-count
        ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
    if not ttir_module.verify():
        raise RuntimeError("Verification for TTIR module has failed")

    return ttir_module, ordered_tensor_names


def ttir_to_functions(
    ttir_module: "TritonIRModule",
) -> dict[str, dict[Intermediate, list[Op]]]:
    """
    Walk the `ttir_module` bottom up to mine the `functions` from
    the structured MLIR entities representing the Triton kernel
    (mlir::Operation, mlir::Block, mlir::Region).
    """
    functions: dict[str, dict[Intermediate, list[Op]]] = {}

    # block id --> op result (Intermediate) --> one or more ops
    op_stack: dict[int, dict[Intermediate, list[Op]]] = defaultdict(
        lambda: defaultdict(list)
    )
    region_id_to_block_ids: dict[int, list[int]] = defaultdict(list)
    block_id_to_block_arg_ids: dict[int, list[int]] = {}
    replacements: dict[int, Union[Intermediate, Param]] = {}
    reindex_map: dict[int, int] = {}
    next_fake_intermediate = 0

    def reindex(idx: int) -> int:
        if idx not in reindex_map:
            reindex_map[idx] = len(reindex_map)
        return reindex_map[idx]

    def mlir_to_functions(op: "TritonIROperation") -> None:
        name: str = op.get_name()
        if name == "builtin.module":
            # this wraps all tt.func ops
            return

        operand_ids: list[int] = [
            reindex(op.get_operand(i).id()) for i in range(op.get_num_operands())
        ]
        result_ids: list[int] = [
            reindex(op.get_result(i).id()) for i in range(op.get_num_results())
        ]

        child_block_ids: list[int] = []
        for i in [op.get_region(i).id() for i in range(op.get_num_regions())]:
            # as the walk is bottom-up, the region_id_to_block_ids[i]
            # must be populated by the time we process the enclosing op
            child_block_ids.extend(region_id_to_block_ids[i])

        parent_block_id = -1
        parent_block = op.get_block()
        if parent_block is not None:
            parent_block_id = parent_block.id()
            if parent_block_id not in block_id_to_block_arg_ids:
                block_id_to_block_arg_ids[parent_block_id] = []
                for i in range(parent_block.get_num_arguments()):
                    block_id_to_block_arg_ids[parent_block_id].append(
                        reindex(parent_block.get_argument(i).id()),
                    )
                # the region info is collected via ops' parent blocks to be
                # used later when the region's encloding op is traversed
                parent_region = parent_block.get_parent()
                if parent_region is not None:
                    region_id_to_block_ids[parent_region.id()].append(parent_block_id)

        nonlocal next_fake_intermediate

        if name == "tt.func":
            # for function ops: gather and inline
            # the ops from all child blocks
            fn_ops = defaultdict(list)
            for child_block_id in child_block_ids:
                for result, block_fn_ops in op_stack.pop(child_block_id).items():
                    for block_fn_op in block_fn_ops:
                        fn_ops[result].append(block_fn_op)

            # replace the corresponding Intermediates in the
            # child op args with the function args (Params)
            for i, idx in enumerate(block_id_to_block_arg_ids[child_block_ids[0]]):
                replacements[idx] = Param(i)

            for fn_op_list in fn_ops.values():
                for fn_op in fn_op_list:
                    for i in range(len(fn_op.args)):
                        arg = fn_op.args[i]
                        seen = set()  # to break cycles
                        # there can be transitive replacements, but likely
                        # no cycles (we keep the `seen` set just in case)
                        while (
                            isinstance(arg, Intermediate)
                            and arg.idx in replacements
                            and arg.idx not in seen
                        ):
                            seen.add(arg.idx)
                            arg = fn_op.args[i] = replacements[arg.idx]

            # next function capture starts
            # with empty replacements
            replacements.clear()

            fn_name = op.get_str_attr("sym_name")
            functions[fn_name] = fn_ops
        elif child_block_ids:
            if name in {"scf.if", "scf.for", "scf.while", "tt.reduce", "tt.scan"}:
                # for blocked ops: inline the enclosed ops into
                # the parent block + rewire the last op in each
                # child block to return the block result
                return_ops = []
                for block_id in child_block_ids:
                    if name == "scf.for":
                        # example:
                        # %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> (i32) ...
                        # block args: 2 (%iv, %arg)
                        # op operands: 4 (%lb, %ub, %step, %init)
                        # `%arg` is mapping to `%init`
                        for i, idx in enumerate(block_id_to_block_arg_ids[block_id]):
                            if i == 0:
                                next_fake_intermediate -= 1
                                replacements[idx] = Intermediate(next_fake_intermediate)
                            else:
                                replacements[idx] = Intermediate(operand_ids[i + 2])
                    elif name == "scf.while":
                        # example:
                        # %3:3 = scf.while (%arg2 = %1, %arg3 = %2, %arg4 = %c0_i32_8) ...
                        # block args: 3 (%arg2, %arg3, %arg4)
                        # op operands: 3 (%1, %2, %c0_i32_8)
                        # `%arg2` is mapping to `%1`, `%arg3` is mapping to `%2`, ...
                        for i, idx in enumerate(block_id_to_block_arg_ids[block_id]):
                            replacements[idx] = Intermediate(operand_ids[i])
                    elif name == "scf.if":
                        # the scf block args are ignored by the pass. but, as they
                        # may be used as operands of the ops inside the block
                        # (and nested blocks inlined in the current block by now),
                        # they are replaced by new fake Intermediates to avoid "this
                        # operand is not returned by any other op in the fn" error
                        # in the downstream analysis
                        for idx in block_id_to_block_arg_ids[block_id]:
                            next_fake_intermediate -= 1
                            replacements[idx] = Intermediate(next_fake_intermediate)
                    else:
                        assert name in ("tt.reduce", "tt.scan")
                        # wire the block arguments to the op arguments
                        num_operands = len(operand_ids)
                        block_arg_ids = block_id_to_block_arg_ids[block_id]
                        assert len(block_arg_ids) == 2 * num_operands, (
                            f"{name} is expected to have twice as "
                            "many block arguments as op arguments: "
                            f"{operand_ids=}, {block_arg_ids=}."
                        )
                        for i, idx in enumerate(block_arg_ids):
                            # for a tt.reduce/tt.scan op with N arguments, the block
                            # arguments comprise N reduced values followed by
                            # N current values corresponding to the N op args
                            replacements[idx] = Intermediate(
                                operand_ids[i % num_operands]
                            )

                    if block_id in op_stack:
                        block_ops = op_stack.pop(block_id)
                        if not block_ops:
                            continue
                        last_ret, last_ops = block_ops.popitem()
                        if all(
                            op.name
                            in ("scf.yield", "tt.reduce.return", "tt.scan.return")
                            for op in last_ops
                        ):
                            # if last_ops are all return ops, treat them separately
                            return_ops.extend(last_ops)
                        else:
                            # otherwise, return last_ops to the block
                            block_ops[last_ret] = last_ops
                        for op_result, child_ops in block_ops.items():
                            op_stack[parent_block_id][op_result].extend(child_ops)

                scf_results = [Intermediate(idx) for idx in result_ids]

                if return_ops and all(
                    (op.name == "scf.yield" and len(result_ids) == len(op.args))
                    for op in return_ops
                ):
                    # [Note: scf.yield fix-up]
                    #
                    # TL;DR: if our scf.yield takes N args, then we'll create N scf.yield ops to handle each of the
                    # args.
                    #
                    #      **Context**:
                    # During mutation analysis, the analysis pass will identify mutating ops (e.g. tt.store)
                    # and then DFS upwards towards the parameters of the function. Specifically, the analysis pass
                    # looks at the mutated arg in tt.store; then looks for its source ops; and then recurses on the
                    # arguments to each of the source ops.
                    #
                    # In the case of scf.if/scf.for, we may have multiple return ops, each passed as an arg
                    # to scf.yield:
                    #
                    # %18:2 = scf.if %... -> (!tt.ptr<f32>, !tt.ptr<f32>) {
                    #   ...
                    #   scf.yield %1, %2
                    # } else {
                    #   scf.yield %3, %4
                    # }
                    #
                    # And for each of the returns of the scf.if, we'd naively assign the source op of each of the
                    # return values to be the scf.yields. But the scf.yields take _all_ the returns as arguments.
                    # Therefore, if _any_ of the return values of the scf.if are mutated, then the analysis pass
                    # would mark _all_ of the yield args as mutated.
                    #
                    #      **Solution**:
                    # For the purposes of this analysis pass, we create N yield ops - one for each
                    # return-val/yield-arg. In the example above, we'll have two scf.yield's for each branch of the
                    # scf.if.

                    for return_op in return_ops:
                        for i, (scf_result, yield_arg) in enumerate(
                            zip(scf_results, return_op.args)
                        ):
                            sub_yield_op = Op(
                                return_op.name,
                                return_op.fn_call_name,
                                [yield_arg],
                                return_op.ret,
                                sub_idx=i,
                            )
                            op_stack[parent_block_id][scf_result].append(sub_yield_op)

                else:
                    for scf_result in scf_results:
                        for return_op in return_ops:
                            op_stack[parent_block_id][scf_result].append(return_op)
            else:
                raise RuntimeError(
                    f"Unknown blocked function: {name}. Can't capture the TTIR."
                )
        else:
            callee = None
            if name == "tt.call":
                callee = op.get_flat_symbol_ref_attr("callee")
            args: list[Union[Param, Intermediate]] = [
                Intermediate(operand) for operand in operand_ids
            ]
            block_ops = op_stack[parent_block_id]

            is_pure = False
            # Handle the case for tt.elementwise_inline_asm to set `is_pure` for mutation analysis
            if name == "tt.elementwise_inline_asm":
                is_pure = op.get_bool_attr("pure")

            if result_ids:
                for result_id in result_ids:
                    res = Intermediate(result_id)
                    block_ops[res].append(Op(name, callee, args, res, is_pure=is_pure))
            else:
                next_fake_intermediate -= 1
                fake_res = Intermediate(next_fake_intermediate)
                block_ops[fake_res].append(
                    Op(name, callee, args, fake_res, is_pure=is_pure)
                )

    ttir_module.walk(mlir_to_functions)

    return functions


class MemoizeWithCycleCheck:
    fn: Callable[..., Any]
    cache: dict[tuple[Any], Any]

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.reset()

    def __call__(
        self,
        functions: dict[str, dict[Intermediate, list[Op]]],
        fn_name: str,
        *args: Any,
    ) -> list[bool]:
        key: tuple[Any, ...] = (fn_name, *args)
        if key not in self.cache:
            self.cache[key] = None
            self.cache[key] = self.fn(functions, fn_name, *args)
        if self.cache[key] is None:
            raise RuntimeError("Recursion is not supported")
        return self.cache[key]

    def reset(self) -> None:
        self.cache = {}


@MemoizeWithCycleCheck
def get_tma_stores(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str
) -> set[Union[Intermediate, Param]]:
    """
    Identifies all intermediates and parameters that are written to by a
    `tt.experimental_descriptor_store`. It tracks only the specific values
    written to via experimental_descriptor_store and the input values to
    `tt.reinterpret_tensor_descriptor` used to construct the direct inputs
    to tt.experimental_descriptor_store - not any recursive values
    used to construct those values.

    For example: for
      tt.reinterpret_tensor_descriptor(Intermediate(idx=0), ...)
      Intermediate(idx=1) = tt.experimental_descriptor_store(Intermediate(idx=0), ...)
    this function will return [Intermediate(idx=0), Intermediate(idx=1)],

    However
      Intermediate(idx=4) = arith.addptr(Intermediate(idx=2), Intermediate(idx=3))
      Intermediate(idx=5) = tt.experimental_descriptor_store(Intermediate(idx=4), ...)
      tt.experimental_descriptor_store(Intermediate(idx=5), ...)
    this function will mark only idx=4 and idx=5 (but not idx=2 or idx=3)

    If an intermediate/parameter is passed into a function and is written to
    via experimental_descriptor_store within that function, the argument to the
    function will also be marked.
    """

    result: set[Union[Intermediate, Param]] = set()

    ops = functions[fn_name]
    for op_list in ops.values():
        for op in op_list:
            if op.name == "tt.call":
                assert op.fn_call_name in functions
                # pyrefly: ignore [bad-argument-type]
                tma_stores = get_tma_stores(functions, op.fn_call_name)
                for i, inp in enumerate(op.args):
                    if Param(idx=i) in tma_stores:
                        result.add(inp)
            elif op.name == "tt.experimental_descriptor_store":
                assert len(op.args) >= 1
                result.add(op.args[0])
            elif op.name == "tt.descriptor_store":
                assert len(op.args) >= 1
                result.add(op.args[0])

    for val in list(result):
        if val in ops:
            if not isinstance(val, Intermediate):
                continue
            for op in ops[val]:
                if op.name == "tt.reinterpret_tensor_descriptor":
                    assert len(op.args) >= 1
                    result.add(op.args[0])

    return result


@MemoizeWithCycleCheck
def analyze_kernel_mutations(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str, num_args: int
) -> list[bool]:
    """
    Analyzes the graph to detect all sinks from a predefined list of sinks
    by using triton's MemWrite trait list. NOTE: What if triton exposed this?
    From each sink, it traverses the CFG backwards to identify all the input
    pointers that are mutated.
    """
    # Name of mutation op to mutated parameter indices
    # List from Triton Github include/triton/Dialect/Triton/IR/TritonOps.td
    # All the OPs that have MemWrite trait.
    # What if Triton exposed this?
    MUTATION_OPS = {
        "tt.store": [0],
        "tt.atomic_cas": [0],
        "tt.atomic_rmw": [0],
        "tt.experimental_descriptor_store": [0],
        "tt.experimental_tensormap_create": [0],
        "tt.descriptor_store": [0],
    }
    # Ops that we want to bail out on
    UNKNOWN_OPS = {"tt.elementwise_inline_asm"}

    stack: list[Union[Param, Intermediate]] = []
    visited = set()
    ops = functions[fn_name]
    tma_stores = get_tma_stores(functions, fn_name)

    for op_list in ops.values():
        for op in op_list:
            # If we encounter an operation with effects that cannot be reliably analyzed
            # (e.g. `tt.elementwise_inline_asm`), we assume it does not mutate any input parameters.
            if op.name in UNKNOWN_OPS:
                if op.name == "tt.elementwise_inline_asm" and op.is_pure:
                    continue
                raise RuntimeError(
                    f"ttir analysis hit an op we do not know how to analyze: {op.name}"
                )

            if op.name == "tt.experimental_tensormap_create":
                # Note: this is how we implement experimental_descriptor_store mutation analysis.
                # for on-device TMA.
                # experimental_tensormap_store(a, b, ...) stores b to the location specified
                # by descriptor in the memory of a.
                # To track this, we first find all the intermediates/params to which we store via
                # experimental_tensormap_store (get_tma_stores, called above). Then, during this
                # analysis we wait to find the corresponding experimental_tensormap_create (if it
                # exists), at which point we will mark the global_ptr as mutated (as done below).
                assert len(op.args) >= 2
                if op.args[0] in tma_stores:
                    stack.append(op.args[1])

            if op.name == "tt.call":
                assert op.fn_call_name in functions
                mutations = analyze_kernel_mutations(
                    functions,
                    # pyrefly: ignore [bad-argument-type]
                    op.fn_call_name,
                    len(op.args),
                )
                stack.extend(arg for arg, mutated in zip(op.args, mutations) if mutated)
            else:
                stack.extend(op.args[idx] for idx in MUTATION_OPS.get(op.name, []))

    # The following is an iterative DFS algorithm
    mutated = [False] * num_args
    while stack:
        arg = stack.pop()
        if arg in visited:
            continue

        visited.add(arg)

        if isinstance(arg, Param):
            if arg.idx >= num_args:
                # This is an argument defined in the kernel, not passed in
                continue
            mutated[arg.idx] = True
        elif isinstance(arg, Intermediate) and not arg.fake():
            for op in ops[arg]:
                # Skip arguments to load
                if op.name != "tt.load":
                    stack.extend(op.args)
    return mutated


def identify_mutated_tensors(
    kernel: "TritonKernelType",
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]:
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect all input tensor mutations
    """

    ttir_module = None
    functions = None
    try:
        ttir_module, ordered_tensor_names = generate_ttir(
            kernel, kwargs, tma_descriptor_metadata
        )

        # extract functions from TTIR using MLIR bindings exposed by Triton code
        functions = ttir_to_functions(ttir_module)

        assert functions is not None
        kernel_name = next(iter(functions.keys()))
        # Triton codegen modifies the name
        # pyrefly: ignore [missing-attribute]
        assert kernel.fn.__name__ in kernel_name
        # Reset the cache between top level invocations
        # The cache for analyze kernel mutations is mainly used for cycle
        # detection, so each top level invocation needs a clean cache
        analyze_kernel_mutations.reset()
        get_tma_stores.reset()
        mutations = analyze_kernel_mutations(
            functions, kernel_name, len(ordered_tensor_names)
        )

        return [
            ordered_tensor_names[i] for i, mutated in enumerate(mutations) if mutated
        ]
    except Exception:
        log.warning(
            "Encountered an exception in identify_mutated_tensors, assuming every input is mutated",
            exc_info=True,
        )
        if ttir_module is not None:
            log.debug("TTIR:\n%s", str(ttir_module))
        if functions is not None:
            log.debug("functions:")
            for name, fn in functions.items():
                log.debug("===\t%s\t===", name)
                for ret, ops in fn.items():
                    log.debug("%s\t=>\t%s", ret, ops)
        return [key for key, value in kwargs.items() if isinstance(value, Tensor)]


###############################################################################
# Triton Kernel Wrappers


# Used for wrapping a Triton Kernel
class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("triton_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
    ) -> Any:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kwargs=kwargs,
        )


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# Used for wrapping a Triton Kernel in a functional manner
class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("triton_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
        tensors_to_clone: list[str],
    ) -> dict[str, Any]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kwargs=kwargs,
            tensors_to_clone=tensors_to_clone,
        )


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(
            # pyrefly: ignore [missing-attribute]
            kernel.fn.__name__,
            # pyrefly: ignore [missing-attribute]
            kernel.configs,
            grid,
        )
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]

    if tma_descriptor_metadata:
        # as we need to launch the kernel here, we "unwrap" the
        # tma_descriptor_metadata, create the TMA descriptors
        # from it, and replace the tensors in the kwargs by the
        # corresponding TMA descriptors before launching
        kwargs = kwargs.copy()
        for k, v in tma_descriptor_metadata.items():
            tensor = kwargs[k]
            if (exp_meta := maybe_unpack_tma_experimental_metadata(v)) is not None:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                dims, block_dims, element_size = exp_meta
                create_tma_descriptor = (
                    create_1d_tma_descriptor
                    if len(dims) == 1
                    else create_2d_tma_descriptor
                )
                kwargs[k] = create_tma_descriptor(
                    tensor.data_ptr(),
                    *dims,
                    *block_dims,
                    element_size,
                )
            else:
                stable_meta = maybe_unpack_tma_stable_metadata(v)
                assert stable_meta is not None
                from triton.tools.tensor_descriptor import TensorDescriptor

                block_shape = stable_meta[0]
                # pyrefly: ignore  # bad-argument-type
                kwargs[k] = TensorDescriptor.from_tensor(tensor, block_shape)

    # move as many positional arguments from dicts to args as we
    # can to circumvent the bug with the kwargs and pre_/post_hook:
    # https://github.com/triton-lang/triton/issues/5082
    # TODO: remove this when the Triton issue above is fixed
    args = []
    # copy kwargs and constant_args here to
    # avoid mutating the original inputs
    kwargs = kwargs.copy()
    constant_args = constant_args.copy()
    # pyrefly: ignore [missing-attribute]
    for name in kernel.arg_names:
        if name in kwargs:
            args.append(kwargs.pop(name))
        elif name in constant_args:
            args.append(constant_args.pop(name))
        else:
            break

    # pyrefly: ignore [index-error]
    kernel[grid_fn](*args, **kwargs, **constant_args)


@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    with mode:
        return None


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    return None


def trace_triton_kernel_wrapper(
    proxy_mode: ProxyTorchDispatchMode,
    func_overload: Callable[..., Any],
    node_args: dict[str, Any],
) -> Optional[dict[str, Any]]:
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy,  # type: ignore[union-attr]
        node_args,
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )

    ret = track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
    return ret


@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
        },
    )

    return None


def get_mutated_tensors(
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]:
    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)
    return identify_mutated_tensors(
        kernel, {**kwargs, **constant_args}, tma_descriptor_metadata
    )


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    # TODO(oulgen): Preexisting bug, if two kernel inputs are views of each
    # other, and one gets mutated in kernel, and later another gets mutated,
    # they are no longer equal. Fix this by graph breaking on this condition
    # earlier in dynamo.
    tensors_to_clone = get_mutated_tensors(
        kernel_idx, constant_args_idx, unwrapped_kwargs, tma_descriptor_metadata
    )
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )

    assert set(unwrapped_outputs.keys()).issubset(set(kwargs.keys()))
    for key, output_arg in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        assert isinstance(input_arg, Tensor)

        ctx.replace(input_arg, output_arg)
        # indicate that above replace is hidden from autograd
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
    return None


@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]:
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    kwargs = {
        key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
        for key, val in kwargs.items()
    }
    triton_kernel_wrapper_mutation(
        kernel_idx=kernel_idx,
        constant_args_idx=constant_args_idx,
        grid=grid,
        tma_descriptor_metadata=tma_descriptor_metadata,
        kwargs=kwargs,
    )
    return {key: val for key, val in kwargs.items() if key in tensors_to_clone}


@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
def triton_kernel_wrapper_functional_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]:
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    with mode:
        return {
            key: clone_preserve_strides(val)
            for key, val in kwargs.items()
            if key in tensors_to_clone
        }


@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]:
    ret = trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_functional,
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
            "tensors_to_clone": tensors_to_clone,
        },
    )
    assert ret is not None
    return ret


@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )
        return ctx.wrap_tensors(outputs)  # type: ignore[return-value,arg-type]


triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCPU)

triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCPU)

# Adds SAC support for triton ops
redirect_to_mode(triton_kernel_wrapper_mutation, _CachingTorchDispatchMode)
redirect_to_mode(triton_kernel_wrapper_mutation, _CachedTorchDispatchMode)

###############################################################################
# The "TritonHOPifier": a class that transforms a call to a triton kernel into
# a call to the triton_kernel_wrapper_mutation HOP.


class TritonHOPifier:
    """Orchestrator for converting a user-defined triton kernel into a call
    to the triton_kernel_wrapper_mutation HOP.

    It has two main use cases.

    1. When Dynamo sees a triton kernel, it wraps it into a TritonKernelVariable
    and uses the TritonHOPifier to convert calls to the TritonKernelVariable
    into a call to the HOP.

    2. In order to capture a user-defined triton kernel while performing
    tracing (via make_fx or non-strict export), a user must annotate their
    triton kernel with the `wrap_triton` decorator. The decorator uses
    TritonHOPifier to convert calls to the triton kernel into a call
    to the HOP (which can then be traced).

    Because Dynamo has its own calling conventions for e.g. invoking a user-defined function
    TritonHOPifier is an abstract class that can be overridden by its subclasses.
    """

    def raise_unsupported(self, msg: str) -> Never:
        raise NotImplementedError("abstract method")

    def is_callable(self, maybe_callable: Any) -> bool:
        raise NotImplementedError("abstract method")

    def get_value(self, val: Any) -> Any:
        raise NotImplementedError("abstract method")

    def call_grid(  # type: ignore[no-untyped-def]
        self,
        grid,
        meta,
        tx,
    ) -> Union[tuple[Union[int, sympy.Expr, SymInt], ...], tuple["Proxy", ...]]:
        raise NotImplementedError("abstract method")

    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: Optional["InstructionTranslator"],
        variable: Optional[
            Union["TritonKernelVariable", "TraceableTritonKernelWrapper"]
        ],
        name: str,
    ) -> Any:
        raise NotImplementedError("abstract method")

    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list,
        kwargs: dict,
        tx: Optional["InstructionTranslator"],
        variable: Optional[
            Union["TritonKernelVariable", "TraceableTritonKernelWrapper"]
        ],
    ) -> Any:
        raise NotImplementedError("abstract method")

    def maybe_unpack_configs(
        self, configs: list["TritonConfig"], tx: Optional["InstructionTranslator"]
    ) -> list["TritonConfig"]:
        raise NotImplementedError("abstract method")

    def maybe_unpack_heuristic_result(self, result: Any) -> Any:
        raise NotImplementedError("abstract method")

    @staticmethod
    def do_prune_configs(  # type: ignore[no-untyped-def]
        autotuner: "TritonAutotunerType",
        early_config_prune: Optional[Callable],
        perf_model: Optional[Callable],
        top_k: float,
        configs: list,
        named_args: dict,
        kwargs: dict,
    ) -> list["TritonConfig"]:
        # Reimplement autotuner.prune_configs(...) here
        # see: https://github.com/triton-lang/triton/blob/e57b46897191b3b3061c78d0d60e58e94be565b6/python/triton/runtime/autotuner.py   # noqa: E501,B950
        # We do this to avoid calling prune_configs, which in turn calls early_config_prune and perf_model
        # These are both user-defined functions which can contain side effects, so we want to sandbox them in Dynamo

        if early_config_prune:
            configs = early_config_prune(configs, named_args, **kwargs)

        if perf_model:
            # we assert top_k is a float before calling this
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(configs) * top_k)
            elif not isinstance(top_k, int):
                """
                Slice index must be an integer, SupportsIndex or None
                """
                raise TypeError(
                    "Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int"
                )
            if len(configs) > top_k:
                est_timing = [
                    (
                        config,
                        float(
                            perf_model(**named_args, **kwargs, **config.all_kwargs())
                        ),
                    )
                    for config in configs
                ]
                configs = [
                    config[0]
                    for config in sorted(est_timing, key=operator.itemgetter(1))[:top_k]
                ]
        return configs

    def call_HOP(  # type: ignore[no-untyped-def]
        self,
        variable,
        grids,
        combined_args: dict[str, Any],
        tx,
    ) -> Optional["ConstantVariable"]:
        raise NotImplementedError("abstract method")

    def check_grid(  # type: ignore[no-untyped-def]
        self, grid
    ) -> Union[tuple[Union[int, sympy.Expr, SymInt], ...], tuple["Proxy", ...]]:
        raise NotImplementedError("abstract method")

    def init_variable(
        self,
        variable: Union["TraceableTritonKernelWrapper", "TritonKernelVariable"],
        kernel: "TritonKernelType",
        kernel_idx: Optional[int],
        grid: Optional["TritonGridType"],
    ) -> None:
        from triton.runtime.autotuner import Autotuner

        assert kernel is not None

        variable.kernel = kernel
        variable.kernel_idx = kernel_side_table.add_kernel(kernel)

        assert kernel_idx is None or variable.kernel_idx == kernel_idx

        # pyrefly: ignore [bad-assignment]
        variable.grid = grid

        if isinstance(kernel, Autotuner):
            import torch
            import torch._dynamo

            # We only support configs, keys, and restore_value arguments
            # of triton.autotune. Make sure other arguments are defaulted.
            defaults = inspect.signature(Autotuner.__init__).parameters
            # Newer version of triton change attribute name from warmup to num_warmup and rep to num_rep.
            # The call to get_first_attr is to maintain backward-compatibility.

            def defaults_ok(
                attr: str, alternates: tuple[str, ...], values: tuple[Any, ...]
            ) -> bool:
                if attr not in defaults:
                    return True
                value = torch._dynamo.utils.get_first_attr(kernel, attr, *alternates)
                if value == defaults[attr].default:
                    return True
                return value in values

            if (
                not torch._inductor.config.unsafe_ignore_unsupported_triton_autotune_args
                and (
                    not defaults_ok("num_warmups", ("warmup",), (25, None))
                    or not defaults_ok("num_reps", ("rep",), (100, None))
                    or not defaults_ok("use_cuda_graph", (), (False,))
                )
            ):
                self.raise_unsupported(
                    "Only configs, keys, restore_value, and reset_to_zero are supported for triton.autotune"
                )
            if (
                not torch._inductor.config.unsafe_ignore_unsupported_triton_autotune_args
                and (
                    # pre_hook requires running arbitrary code at runtime, which we cannot handle at this time
                    # https://github.com/pytorch/pytorch/issues/139059
                    # we can't support pre_hook or post_hook in user defined triton kernels at the moment,
                    # as they require the ability to execute code at runtime (AOTI can't support this)
                    (
                        hasattr(kernel, "user_defined_pre_hook")
                        and kernel.user_defined_pre_hook is not False
                    )
                    or (
                        hasattr(kernel, "user_defined_post_hook")
                        and kernel.user_defined_post_hook is not False
                    )
                    or (
                        # Check Config passed to autotuner in configs
                        any(cfg.pre_hook is not None for cfg in kernel.configs)
                    )
                )
            ):
                self.raise_unsupported(
                    "pre_hook and post_hook are not supported in triton.Autotune or triton.Config"
                )

    def call_getitem(
        self,
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"],
        args: Sequence[Any],
    ) -> Union["TritonKernelVariable", "TraceableTritonKernelWrapper"]:
        # __getitem__ should only be called if we don't already have a grid
        # Only grid needs to be passed
        if variable.grid is not None or len(args) != 1:
            self.raise_unsupported(
                "Triton kernels should be called with only a single grid"
            )

        return type(variable)(
            kernel=variable.kernel,
            kernel_idx=variable.kernel_idx,
            grid=args[0],
        )

    def call_run(
        self,
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: Optional["InstructionTranslator"],
    ) -> Optional["ConstantVariable"]:
        if "grid" not in kwargs:
            self.raise_unsupported("Triton kernel requires to be called with a grid")
        grid = kwargs.pop("grid")
        kwargs.pop("warmup", None)
        # rewrite kernel.run(*args, grid=grid) to kernel[grid](*args)
        return self.call_triton_kernel(
            type(variable)(
                kernel=variable.kernel, kernel_idx=variable.kernel_idx, grid=grid
            ),
            args,
            kwargs,
            tx,
        )

    def call_triton_kernel(
        self,
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: Optional["InstructionTranslator"],
    ) -> Optional["ConstantVariable"]:
        from triton import JITFunction
        from triton.runtime.autotuner import autotune, Autotuner, Config, Heuristics

        # Check if num_ctas is in kwargs
        if "num_ctas" in kwargs:
            self.raise_unsupported(
                "Passing num_ctas directly to the Triton kernel is not supported. "
                "Please use a Config in @triton.autotune instead."
            )

        # Make sure the kernel has a grid
        if variable.grid is None:
            self.raise_unsupported("Triton kernels should always be called with a grid")

        # raise an exception if there are multiple @triton.autotune decorators
        iter_kernel = variable.kernel
        autotuner_count = 0
        while not isinstance(iter_kernel, JITFunction):
            if isinstance(iter_kernel, Autotuner):
                autotuner_count += 1
            if autotuner_count > 1:
                self.raise_unsupported(
                    "Passing multiple @triton.autotune decorators is not supported. "
                    "Please use a single @triton.autotune decorator instead."
                )
            # pyrefly: ignore  # missing-attribute
            iter_kernel = iter_kernel.fn

        # Process the @triton.heuristics decorator:
        # - We know there is only 1 autotuner decorator here
        # - We can apply the heuristic to all triton.Configs in the order that the decorators appear
        #   This way, when the config is selected, the heuristics have already been applied.
        # - Decorators that appear *before* the autotuner are already processed correctly
        if isinstance(variable.kernel, Autotuner) and isinstance(
            variable.kernel.fn, Heuristics
        ):
            # unwrap the heuristics decorator, we don't need it anymore
            # variable.kernel ==> Autotuner
            # variable.kernel.fn ==> Heuristics
            # ...
            # There can be arbitrarily many heuristics wrappers here!
            # ...
            # variable.kernel.fn ==> JITFunction

            # Copy the configs, we are going to be modifying them
            new_configs = copy.deepcopy(variable.kernel.configs)

            named_args = dict(zip(variable.kernel.arg_names, args))

            # Iterate through all of the heuristics wrappers that come after the autotune wrapper
            iter_kernel = variable.kernel.fn
            while isinstance(iter_kernel, Heuristics):
                # For each config, apply the heuristic fn(s)
                for config_idx in range(len(new_configs)):
                    for kwarg_key, heuristic_fn in iter_kernel.values.items():
                        # Run heuristics on the combined configs + kwargs
                        heuristic_result = self.call_user_defined_fn(
                            heuristic_fn,
                            [
                                {
                                    **named_args,
                                    **kwargs,
                                    **new_configs[config_idx].__dict__["kwargs"],
                                },
                            ],
                            {},
                            tx,
                            variable,
                        )

                        # Update the kwargs in each config
                        # maybe_unpack_heuristic_result raises unsupported if the value is non-constant
                        new_configs[config_idx].__dict__["kwargs"][kwarg_key] = (
                            self.maybe_unpack_heuristic_result(heuristic_result)
                        )

                iter_kernel = iter_kernel.fn
            assert isinstance(iter_kernel, JITFunction)
            prune_configs_by = {
                "perf_model": variable.kernel.perf_model,
                "early_config_prune": variable.kernel.early_config_prune,
                "configs_top_k": variable.kernel.configs_top_k,
            }
            new_kernel = autotune(
                configs=new_configs, key=[], prune_configs_by=prune_configs_by
            )(iter_kernel)
            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = type(variable)(new_kernel, None, variable.grid)
            return self.call_triton_kernel(new_var, args, kwargs, tx)

        SPECIAL_CONFIG_NAMES = {
            "num_warps",
            "num_stages",
            "num_ctas",
            "num_consumer_groups",
            "num_buffers_warp_spec",
            "num_cpu_threads",
        }

        # move special config names to configs out of kwargs
        special_kwargs = {}
        for name in SPECIAL_CONFIG_NAMES:
            if name in kwargs:
                # remove special kwargs from `kwargs`
                val = kwargs.pop(name)
                special_kwargs[name] = self.get_value(val)

        if special_kwargs:
            if isinstance(variable.kernel, Autotuner):
                # if there is Autotuner already, set
                # special kwargs to each of its configs
                new_configs = copy.deepcopy(variable.kernel.configs)
                for config in new_configs:
                    config.__dict__.update(special_kwargs)
                prune_configs_by = {
                    "perf_model": variable.kernel.perf_model,
                    "early_config_prune": variable.kernel.early_config_prune,
                    "configs_top_k": variable.kernel.configs_top_k,
                }

                new_kernel = autotune(
                    configs=new_configs, key=[], prune_configs_by=prune_configs_by
                )(variable.kernel.fn)
            else:
                # if there is no Autotuner, wrap the kernel into a
                # new one with a single config with special kwargs
                new_config = Config(kwargs={}, **special_kwargs)

                new_kernel = autotune(configs=[new_config], key=[])(variable.kernel)

            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = type(variable)(new_kernel, None, variable.grid)
            return self.call_triton_kernel(new_var, args, kwargs, tx)

        if isinstance(variable.kernel, Autotuner):
            special_param_names = []
            for name in SPECIAL_CONFIG_NAMES:
                if name in variable.kernel.fn.arg_names:
                    special_param_names.append(name)

            if special_param_names:
                # If the Triton kernel has SPECIAL_CONFIG_NAMES in parameters, those should
                # be passed from the kernel configs: the behavior of Triton runtime is that
                # those values get folded into the kernel arguments iff there are parameters
                # with the same name. Normally the values of those parameters are defined
                # outside the `kwargs` part of the autotuning configs. Here we move them to
                # the `kwargs` part (if they're absent there) to facilitate passing them as
                # arguments to the kernel downstream.
                updated = False
                new_configs = copy.deepcopy(variable.kernel.configs)
                for config in new_configs:
                    for name in special_param_names:
                        if name not in config.__dict__["kwargs"]:
                            assert name in config.__dict__, (
                                f"{name} must be in autotuning configs to be used as a kernel parameter"
                            )
                            config.__dict__["kwargs"][name] = config.__dict__[name]
                            updated = True

                if updated:
                    prune_configs_by = {
                        "perf_model": variable.kernel.perf_model,
                        "early_config_prune": variable.kernel.early_config_prune,
                        "configs_top_k": variable.kernel.configs_top_k,
                    }

                    new_kernel = autotune(
                        configs=new_configs, prune_configs_by=prune_configs_by, key=[]
                    )(variable.kernel.fn)
                    new_var = type(variable)(new_kernel, None, variable.grid)
                    return self.call_triton_kernel(new_var, args, kwargs, tx)

        # These are the default values in upstream Triton
        # see: https://github.com/triton-lang/triton/blob/e57b46897191b3b3061c78d0d60e58e94be565b6/python/triton/runtime/autotuner.py # noqa: E501,B950
        default_perf_model = None
        default_early_config_prune = None

        # run prune_configs_by
        if isinstance(variable.kernel, Autotuner) and (
            variable.kernel.perf_model != default_perf_model
            or variable.kernel.early_config_prune != default_early_config_prune
        ):
            # Prune the configs
            named_args = dict(zip(variable.kernel.arg_names, args))

            # The source information is important here so the guards are installed correctly

            wrapped_early_configs_prune = self.wrap_user_defined_obj(
                variable.kernel.early_config_prune,
                tx,
                variable,
                "early_config_prune",
            )

            wrapped_perf_model = self.wrap_user_defined_obj(
                variable.kernel.perf_model, tx, variable, "perf_model"
            )

            wrapped_configs_top_k = self.wrap_user_defined_obj(
                variable.kernel.configs_top_k, tx, variable, "configs_top_k"
            )

            wrapped_configs = self.wrap_user_defined_obj(
                variable.kernel.configs, tx, variable, "configs"
            )

            pruned_configs = self.call_user_defined_fn(
                self.do_prune_configs,
                [
                    variable,
                    wrapped_early_configs_prune,
                    wrapped_perf_model,
                    wrapped_configs_top_k,
                    wrapped_configs,
                    named_args,
                    kwargs,
                ],
                {},
                tx,
                variable,
            )

            pruned_configs = self.maybe_unpack_configs(pruned_configs, tx)

            # after pruning the configs, create a new autotuner object with
            # these configs and recurse.
            new_kernel = autotune(configs=pruned_configs, key=[])(variable.kernel.fn)
            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = type(variable)(new_kernel, None, variable.grid)
            return self.call_triton_kernel(new_var, args, kwargs, tx)

        # Both for grid's meta as well as for the kernel, we need combined
        # args and kwargs combined and normalized
        # pyrefly: ignore  # missing-attribute
        combined_args_raw = {**dict(zip(variable.kernel.arg_names, args)), **kwargs}

        # precompute the grid for the kernel
        configs = (
            [config.kwargs for config in variable.kernel.configs]
            if isinstance(variable.kernel, Autotuner)
            else [{}]
        )
        grids = []
        for config_args in configs:
            # If the grid is a function, then lets execute it and convert it to
            # a list
            grid = variable.grid
            assert grid is not None
            if self.is_callable(grid):
                # Populate the special "meta" argument to call the grid function
                meta = {**combined_args_raw, **config_args}
                grid = self.call_grid(grid, meta, tx)  # type: ignore[arg-type]
            grids.append(self.check_grid(grid))

        for i in range(len(grids)):
            if not isinstance(grids[i], tuple):
                self.raise_unsupported("Only tuple grids are supported")
            # inductor expects all grids to be 3-tuple so lets make it
            if len(grids[i]) == 1:
                grids[i] = (grids[i][0], 1, 1)
            elif len(grids[i]) == 2:
                grids[i] = (grids[i][0], grids[i][1], 1)
            elif len(grids[i]) > 3:
                self.raise_unsupported("Grid can have at most rank 3")

        assert len(grids) != 0
        if isinstance(variable.kernel, JITFunction):
            constexprs = [p.num for p in variable.kernel.params if p.is_constexpr]
            arg_names = [p.name for p in variable.kernel.params]
        else:
            # If we are looking at an @triton.autotune decorator, the nested function should be a JITFunction
            # This is because we don't support @triton.heuristics or nested @triton.autotune decorators yet
            assert isinstance(variable.kernel, Autotuner)
            constexprs = [p.num for p in variable.kernel.fn.params if p.is_constexpr]
            arg_names = [p.name for p in variable.kernel.fn.params]

        for idx, arg_name in enumerate(arg_names):
            if idx in constexprs:
                if arg_name in combined_args_raw:
                    # [Note: Specialize tl.constexpr args in user-defined triton kernels]
                    # This arg is marked as tl.constexpr. That means that triton will recompile every time
                    # this value changes.
                    # https://github.com/pytorch/pytorch/issues/136504
                    # One option is to correctly pass the symints in so that the symbolic expressions are defined
                    # when the triton code is being executed.
                    # But since triton will have to recompile either way, we instead just specialize on the value.
                    #
                    # Depending on the type of `variable` we might expect different types for the symbolic args:
                    # either SymNodeVariables (for TritonKernelVariables) or SymInts (TracingTritonKernelWrapper)
                    combined_args_raw[arg_name] = variable.specialize_symbolic(
                        combined_args_raw[arg_name]
                    )
        return self.call_HOP(variable, grids, combined_args_raw, tx)


###############################################################################
# Helpers for wrap_triton API that makes a user-defined triton kernel traceable into
# a graph via make_fx or non-strict export (coming soon)


class TracingTritonHOPifier(TritonHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise RuntimeError(msg)

    def is_callable(self, maybe_callable: Any) -> bool:
        return callable(maybe_callable)

    def get_value(self, val: Any) -> Any:
        return val

    def call_grid(
        self,
        grid: "TritonGridCallableType",
        meta: "TritonMetaParamsType",
        tx: None,
    ) -> tuple[Union[int, sympy.Expr, SymInt], ...]:
        assert tx is None
        assert isinstance(meta, dict)
        assert callable(grid)
        return grid(meta)

    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: Optional["InstructionTranslator"],
        variable: Optional[
            Union["TritonKernelVariable", "TraceableTritonKernelWrapper"]
        ],
        name: str,
    ) -> Any:
        assert tx is None
        return user_obj

    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list,
        kwargs: dict,
        tx: Optional["InstructionTranslator"],
        variable: Optional[
            Union["TritonKernelVariable", "TraceableTritonKernelWrapper"]
        ],
    ) -> Any:
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert callable(user_fn)
        return user_fn(*args, **kwargs)

    def maybe_unpack_configs(
        self, configs: list["TritonConfig"], tx: Optional["InstructionTranslator"]
    ) -> list["TritonConfig"]:
        assert isinstance(configs, list)
        return configs

    def maybe_unpack_heuristic_result(self, result: Any) -> Any:
        return result

    def check_grid(
        self,
        grid: "TritonGridType",
    ) -> tuple[Union[int, sympy.Expr, SymInt], ...]:
        if not isinstance(grid, collections.abc.Sequence):
            raise RuntimeError(
                "wrap_triton can only handle grids that resolve to Sequence[int]."
            )
        # normalize to tuple
        return tuple(grid)

    def store_non_graphable_args(
        self,
        combined_args: dict[str, Any],
    ) -> tuple[dict, int]:
        """
        Some args cannot be stored in the FX graph.
        Put them in the side table.
        """

        def is_graphable(val: Any) -> bool:
            return isinstance(val, (fx.node.base_types, fx.Node))

        non_graphable_args = {
            k: v for k, v in combined_args.items() if not is_graphable(v)
        }
        graphable_args = {k: v for k, v in combined_args.items() if is_graphable(v)}

        constant_args_idx = kernel_side_table.add_constant_args(non_graphable_args)

        return graphable_args, constant_args_idx

    def call_HOP(
        self,
        variable: "TraceableTritonKernelWrapper",
        grids: list["TritonGridTupleType"],
        combined_args: dict[str, Any],
        tx: None,
    ) -> None:
        assert tx is None
        assert isinstance(variable, TraceableTritonKernelWrapper)

        graphable_args, constant_args_idx = self.store_non_graphable_args(combined_args)

        assert isinstance(variable.kernel_idx, int)
        return triton_kernel_wrapper_mutation(
            kernel_idx=variable.kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grids,  # type: ignore[arg-type]
            # TMA descriptor capturing not yet
            # supported in non-dynamo tracing
            tma_descriptor_metadata={},
            kwargs=graphable_args,
        )


tracing_triton_hopifier_singleton = TracingTritonHOPifier()


class TraceableTritonKernelWrapper:
    kernel: "TritonKernelType"
    kernel_idx: Optional[int]
    grid: Optional["TritonGridType"]

    def __init__(
        self,
        kernel: "TritonKernelType",
        kernel_idx: Optional[int],
        grid: Optional["TritonGridType"],
    ) -> None:
        # pyrefly: ignore  # bad-assignment
        self.kernel = None
        self.grid = None
        tracing_triton_hopifier_singleton.init_variable(self, kernel, kernel_idx, grid)
        assert self.kernel is not None

    def __getitem__(self, *args: Sequence[Any]) -> "TraceableTritonKernelWrapper":
        return tracing_triton_hopifier_singleton.call_getitem(self, args)  # type: ignore[return-value]

    def run(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_triton_hopifier_singleton.call_run(self, args, kwargs, None)
        else:
            assert self.kernel is not None
            # pyrefly: ignore [missing-attribute]
            return self.kernel.run(*args, **kwargs)

    def __call__(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_triton_hopifier_singleton.call_triton_kernel(
                self, args, kwargs, None
            )
        else:
            assert self.kernel is not None
            # pyrefly: ignore [index-error]
            return self.kernel[self.grid](*args, **kwargs)

    def specialize_symbolic(self, arg: Sequence[Any]) -> Any:
        import torch

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, (torch.SymInt, torch.SymBool, torch.SymFloat)):
            return guard_scalar(arg)
        return arg
