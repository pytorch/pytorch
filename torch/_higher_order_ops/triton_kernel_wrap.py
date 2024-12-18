import collections
import copy
import dataclasses
import inspect
import logging
import threading
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Never

import sympy

import torch.fx as fx
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import guard_scalar


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

    TritonMetaParamsType = Dict[str, int]
    TritonGridTupleType = Tuple[Union[int, sympy.Expr, SymInt], ...]
    TritonGridCallableType = Callable[[TritonMetaParamsType], Tuple[int, ...]]
    TritonGridType = Union[TritonGridTupleType, TritonGridCallableType]

    if has_triton():
        from triton import Config as TritonConfig
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    else:

        class Autotuner:  # type: ignore[no-redef]
            pass

        class JITFunction:  # type: ignore[no-redef]
            pass

        class TritonConfig:  # type: ignore[no-redef]
            pass

    TritonKernelType = Union[Autotuner, JITFunction]

    # types for prune_configs_by
    TritonAutotunerType = Union[Autotuner]
    TritonConfigType = Union[TritonConfig]

log = logging.getLogger("torch._dynamo")

# TMADescriptorMetadata maps kernel parameter names to the metadata that allows
# reconstructing TMA descriptors from the underlying tensors (passed as kernel
# arguments in the fx graph, instead of the TMA descriptors). Namely: a tuple
# conisting of list of dims, list of block dims, and element size. E.g., for this
# call in host-side Triton TMA API ``create_2d_tma_descriptor(ptr, 50, 60, 32, 15, 4)``,
# the metadata will look like ``([50, 60], [32, 15], 4)``. All ints can be SymInts.
TMADescriptorMetadata = Dict[
    str,  # kernel parameter name
    Tuple[
        List[Union[int, SymInt]],  # dims
        List[Union[int, SymInt]],  # block_dims
        Union[int, SymInt],  # element_size
    ],
]


###############################################################################
# Kernel Side Table


# We cannot put Triton Kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable:
    id_to_kernel: Dict[int, "TritonKernelType"] = {}
    kernel_to_id: Dict["TritonKernelType", int] = {}
    constant_args: Dict[int, Dict[str, Any]] = {}
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
    def add_constant_args(self, args: Dict[str, Any]) -> int:
        with self.lock:
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            return idx

    # Returns the constant args
    def get_constant_args(self, idx: int) -> Dict[str, Any]:
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
    args: List[Union[Param, Intermediate]]
    ret: Intermediate = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if self.name == "tt.call":
            assert self.fn_call_name is not None
        else:
            assert self.fn_call_name is None


def generate_ttir(
    kernel: "TritonKernelType", kwargs: Dict[str, Any]
) -> Tuple["TritonIRModule", List[str]]:
    """
    Uses Triton's internal code generation to create TTIR
    """
    import sympy
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    import torch._inductor.ir
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            # If we are autotuning, then it doesn't matter which version gets
            # picked for tracing purposes, so lets pick the first one
            kwargs = {**kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    assert isinstance(kernel, JITFunction)

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
    ordered_args: Dict[str, Any] = {}
    for name in kernel.arg_names:
        a = kwargs[name]
        if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
            ordered_args[name] = 2
        elif isinstance(a, (FakeTensor, torch._inductor.ir.TensorBox)):
            with torch._C._DisableTorchDispatch():
                ordered_args[name] = torch.empty(2, dtype=a.dtype)
        else:
            ordered_args[name] = a

    ordered_tensor_names = [
        name for name, arg in ordered_args.items() if isinstance(arg, Tensor)
    ]

    def _get_specialization(args):  # type: ignore[no-untyped-def]
        try:
            from triton.backends.compiler import AttrsDescriptor  # noqa: F401

            target = triton.runtime.driver.active.get_current_target()
            backend = triton.compiler.compiler.make_backend(target)
            return backend.get_attrs_descriptor(args, kernel.params)
        except ImportError:
            return kernel._get_config(*args)

    specialization = _get_specialization(ordered_args.values())
    constants = {
        name: arg for name, arg in ordered_args.items() if not isinstance(arg, Tensor)
    }

    # Build kernel signature -- doesn't include constexpr arguments.
    signature = {
        name: kernel._type_of(kernel._key_of(arg))
        for i, (name, arg) in enumerate(ordered_args.items())
        if i not in kernel.constexprs
    }

    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(kernel, signature, constants, specialization)

    # Triton changes ASTSource.make_ir to take 3/4 arguments. Handle
    # backward compatibility here.
    make_ir_sig_params = len(inspect.signature(src.make_ir).parameters)
    if make_ir_sig_params == 2:
        ttir_module = src.make_ir(options, context)
    elif make_ir_sig_params == 3:
        codegen_fns = backend.get_codegen_implementation()
        ttir_module = src.make_ir(options, codegen_fns, context)
    else:
        codegen_fns = backend.get_codegen_implementation()
        module_map = backend.get_module_map()
        ttir_module = src.make_ir(options, codegen_fns, module_map, context)
    if not ttir_module.verify():
        raise RuntimeError("Verification for TTIR module has failed")

    return ttir_module, ordered_tensor_names


def ttir_to_functions(
    ttir_module: "TritonIRModule",
) -> Dict[str, Dict[Intermediate, List[Op]]]:
    """
    Walk the `ttir_module` bottom up to mine the `functions` from
    the structured MLIR entities representing the Triton kernel
    (mlir::Operation, mlir::Block, mlir::Region).
    """
    functions: Dict[str, Dict[Intermediate, List[Op]]] = {}

    # block id --> op result (Intermediate) --> one or more ops
    op_stack: Dict[int, Dict[Intermediate, List[Op]]] = defaultdict(
        lambda: defaultdict(list)
    )
    region_id_to_block_ids: Dict[int, List[int]] = defaultdict(list)
    block_id_to_block_arg_ids: Dict[int, List[int]] = {}
    replacements: Dict[int, Union[Intermediate, Param]] = {}
    reindex_map: Dict[int, int] = {}
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

        operand_ids: List[int] = [
            reindex(op.get_operand(i).id()) for i in range(op.get_num_operands())
        ]
        result_ids: List[int] = [
            reindex(op.get_result(i).id()) for i in range(op.get_num_results())
        ]

        child_block_ids: List[int] = []
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
            args: List[Union[Param, Intermediate]] = [
                Intermediate(operand) for operand in operand_ids
            ]
            block_ops = op_stack[parent_block_id]
            if result_ids:
                for result_id in result_ids:
                    res = Intermediate(result_id)
                    block_ops[res].append(Op(name, callee, args, res))
            else:
                next_fake_intermediate -= 1
                fake_res = Intermediate(next_fake_intermediate)
                block_ops[fake_res].append(Op(name, callee, args, fake_res))

    ttir_module.walk(mlir_to_functions)

    return functions


class MemoizeWithCycleCheck:
    fn: Callable[..., Any]
    cache: Dict[Tuple[str, int], Any]

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.reset()

    def __call__(
        self,
        functions: Dict[str, Dict[Intermediate, List[Op]]],
        fn_name: str,
        num_args: int,
    ) -> List[bool]:
        key = (fn_name, num_args)
        if key not in self.cache:
            self.cache[key] = None
            self.cache[key] = self.fn(functions, fn_name, num_args)
        if self.cache[key] is None:
            raise RuntimeError("Recursion is not supported")
        return self.cache[key]

    def reset(self) -> None:
        self.cache = {}


@MemoizeWithCycleCheck
def analyze_kernel_mutations(
    functions: Dict[str, Dict[Intermediate, List[Op]]], fn_name: str, num_args: int
) -> List[bool]:
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
    }
    # Ops that we want to bail out on
    UNKNOWN_OPS = {"tt.elementwise_inline_asm"}

    stack: List[Union[Param, Intermediate]] = []
    visited = set()
    ops = functions[fn_name]
    for op_list in ops.values():
        for op in op_list:
            if op.name in UNKNOWN_OPS:
                raise RuntimeError(
                    f"ttir analysis hit an op we do not know how to analyze: {op.name}"
                )

            if op.name == "tt.call":
                assert op.fn_call_name in functions
                mutations = analyze_kernel_mutations(
                    functions, op.fn_call_name, len(op.args)
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
    kernel: "TritonKernelType", kwargs: Dict[str, Any]
) -> List[str]:
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect all input tensor mutations
    """

    ttir_module = None
    functions = None
    try:
        ttir_module, ordered_tensor_names = generate_ttir(kernel, kwargs)

        # extract functions from TTIR using MLIR bindings exposed by Triton code
        functions = ttir_to_functions(ttir_module)

        assert functions is not None
        kernel_name = next(iter(functions.keys()))
        # Triton codegen modifies the name
        assert kernel.fn.__name__ in kernel_name
        # Reset the cache between top level invocations
        # The cache for analyze kernel mutations is mainly used for cycle
        # detection, so each top level invocation needs a clean cache
        analyze_kernel_mutations.reset()
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
        grid: List["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: Dict[str, Any],
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
        grid: List["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: Dict[str, Any],
        tensors_to_clone: List[str],
    ) -> Dict[str, Any]:
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
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
) -> None:
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(
            kernel.fn.__name__, kernel.configs, grid
        )
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]

    if tma_descriptor_metadata:
        from triton.tools.experimental_descriptor import (  # noqa: F401
            create_1d_tma_descriptor,
            create_2d_tma_descriptor,
        )

        # as we need to launch the kernel here, we "unwrap" the
        # tma_descriptor_metadata, create the TMA descriptors
        # from it, and replace the tensors in the kwargs by the
        # correspoinding TMA descriptors before launching
        kwargs = kwargs.copy()
        for k, v in tma_descriptor_metadata.items():
            tensor = kwargs[k]
            dims, block_dims, element_size = v
            create_tma_descriptor = (
                create_1d_tma_descriptor if len(dims) == 1 else create_2d_tma_descriptor
            )
            kwargs[k] = create_tma_descriptor(
                tensor.data_ptr(),
                *dims,
                *block_dims,
                element_size,
            )
    # move as many positional arguments from dicts to args as we
    # can to circumvent the bug with the kwargs and pre_/post_hook:
    # https://github.com/triton-lang/triton/issues/5082
    # TODO: remove this when the Triton issue above is fixed
    args = []
    # copy kwargs and constant_args here to
    # avoid mutating the original inputs
    kwargs = kwargs.copy()
    constant_args = constant_args.copy()
    for name in kernel.arg_names:
        if name in kwargs:
            args.append(kwargs.pop(name))
        elif name in constant_args:
            args.append(constant_args.pop(name))
        else:
            break

    kernel[grid_fn](*args, **kwargs, **constant_args)


@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
) -> None:
    with mode:
        return None


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
) -> None:
    return None


def trace_triton_kernel_wrapper(
    proxy_mode: ProxyTorchDispatchMode,
    func_overload: Callable[..., Any],
    node_args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy, node_args  # type: ignore[union-attr]
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
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
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
    kernel_idx: int, constant_args_idx: int, kwargs: Dict[str, Any]
) -> List[str]:
    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)
    return identify_mutated_tensors(kernel, {**kwargs, **constant_args})


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
) -> None:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    # TODO(oulgen): Preexisting bug, if two kernel inputs are views of each
    # other, and one gets mutated in kernel, and later another gets mutated,
    # they are no longer equal. Fix this by graph breaking on this condition
    # earlier in dynamo.
    tensors_to_clone = get_mutated_tensors(
        kernel_idx, constant_args_idx, unwrapped_kwargs
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
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
    tensors_to_clone: List[str],
) -> Dict[str, Any]:
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
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
    tensors_to_clone: List[str],
) -> Dict[str, Any]:
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
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
    tensors_to_clone: List[str],
) -> Dict[str, Any]:
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
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
    tensors_to_clone: List[str],
) -> Dict[str, Any]:
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


def call_prune_configs(  # type: ignore[no-untyped-def]
    autotuner,
    early_config_prune: Callable,
    perf_model: Callable,
    top_k: float,
    configs: List,
    named_args: Dict,
    kwargs: Dict,
):
    # we need this to process the configs for the user-defined functions

    # Reimplement autotuner.prune_configs(...) here
    # see: https://github.com/triton-lang/triton/blob/e57b46897191b3b3061c78d0d60e58e94be565b6/python/triton/runtime/autotuner.py   # noqa: E501,B950
    # We do this to avoid calling prune_configs, which in turn calls early_config_prune and perf_model
    # These are both user-defined functions which can contain side effects, so we want to sandbox them in Dynamo

    if early_config_prune:
        configs = early_config_prune(configs, named_args, **kwargs)

    if perf_model:
        # we assert top_k is a float before calling this
        if top_k <= 1.0:
            top_k = int(len(configs) * top_k)
        if len(configs) > top_k:
            est_timing = {
                # call the user-defined function here as well
                config: perf_model(**named_args, **kwargs, **config.all_kwargs())
                for config in configs
            }
            # realize the results of calling perf_model
            est_timing = {k: v for k, v in est_timing.items()}
            configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
    return configs


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
    TritonHOPifier is an abstract class that can be overriden by its subclasses.
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
    ) -> Union[Tuple[Union[int, sympy.Expr, SymInt], ...], Tuple["Proxy", ...]]:
        raise NotImplementedError("abstract method")

    def call_user_defined_fn(
        self, user_fn: Callable[..., Any], args: List, kwargs: Dict
    ) -> Any:
        raise NotImplementedError("abstract method")

    def call_HOP(  # type: ignore[no-untyped-def]
        self,
        variable,
        grids,
        combined_args: Dict[str, Any],
        tx,
    ) -> Optional["ConstantVariable"]:
        raise NotImplementedError("abstract method")

    def check_grid(  # type: ignore[no-untyped-def]
        self, grid
    ) -> Union[Tuple[Union[int, sympy.Expr, SymInt], ...], Tuple["Proxy", ...]]:
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

        variable.grid = grid

        if isinstance(kernel, Autotuner):
            import torch
            import torch._dynamo

            # We only support configs, keys, and restore_value arguments
            # of triton.autotune. Make sure other arguments are defaulted.
            defaults = inspect.signature(Autotuner.__init__).parameters
            # Newer version of triton change attribute name from warmup to num_warmup and rep to num_rep.
            # The call to get_first_attr is to maintain backward-compatibility.
            if (
                not torch._inductor.config.unsafe_ignore_unsupported_triton_autotune_args
                and (
                    (
                        "warmup" in defaults
                        and defaults["warmup"].default
                        != torch._dynamo.utils.get_first_attr(
                            kernel, "num_warmups", "warmup"
                        )
                    )
                    or (
                        "rep" in defaults
                        and defaults["rep"].default
                        != torch._dynamo.utils.get_first_attr(kernel, "num_reps", "rep")
                    )
                    or (
                        "use_cuda_graph" in defaults
                        and defaults["use_cuda_graph"].default != kernel.use_cuda_graph
                    )
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
        kwargs: Dict[str, Any],
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
        kwargs: Dict[str, Any],
        tx: Optional["InstructionTranslator"],
    ) -> Optional["ConstantVariable"]:
        from triton import JITFunction
        from triton.runtime.autotuner import autotune, Autotuner, Config

        SPECIAL_CONFIG_NAMES = {"num_warps", "num_stages", "num_ctas"}

        if "num_ctas" in kwargs:
            self.raise_unsupported(
                "Passing num_ctas directly to the Triton kernel is not supported. "
                "Please use a Config in @triton.autotune instead."
            )

        # We support running a single Autotuner for each Triton kernel
        # Currently, if there are multiple autotuning decorators, the subsequent ones will be silently ignored
        # We raise an error here to avoid silent incorrectness
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
            iter_kernel = iter_kernel.fn
        # These are the default values in upstream Triton
        # see: https://github.com/triton-lang/triton/blob/e57b46897191b3b3061c78d0d60e58e94be565b6/python/triton/runtime/autotuner.py # noqa: E501,B950
        default_perf_model = None
        default_early_config_prune = None

        if isinstance(variable.kernel, Autotuner) and (
            variable.kernel.perf_model != default_perf_model
            or variable.kernel.early_config_prune != default_early_config_prune
        ):
            # Prune the configs
            named_args = dict(zip(variable.kernel.arg_names, args))

            assert tx is not None

            # These functions need to be wrapped since in the TritonKernelVariable they are just
            # "regular" functions and objects
            # The source information is important here so the guards are installed correctly

            wrapped_early_configs_prune = self.wrap_user_defined_obj(
                variable.kernel.early_config_prune,
                tx,
                variable.source,
                "early_config_prune",
            )
            wrapped_perf_model = self.wrap_user_defined_obj(
                variable.kernel.perf_model, tx, variable.source, "perf_model"
            )

            wrapped_configs_top_k = self.wrap_user_defined_obj(
                variable.kernel.configs_top_k, tx, variable.source, "configs_top_k"
            )

            wrapped_configs = self.wrap_user_defined_obj(
                variable.kernel.configs, tx, variable.source, "configs"
            )

            # OSS Triton requires top_k to be a float, dynamo can't handle running this check
            # So, we can do it here. if configs_top_k isn't a float, we just set it to 2.
            # This has the equivalent behavior.
            # We don't need to restore it because we are going to create a new autotuner object
            # that won't have this attribute anyways.
            if not isinstance(variable.kernel.configs_top_k, float):
                variable.kernel.configs_top_k = 2.0

            wrapped_configs_top_k = self.wrap_user_defined_obj(
                variable.kernel.configs_top_k, tx, variable.source, "configs_top_k"
            )

            pruned_configs = self.call_user_defined_fn(
                call_prune_configs,
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
                variable.source,
            )

            # The result of prune_configs must be a var sequence
            # (this is true in OSS as well)
            # Similar to guard_as_python_constant, this function inserts
            # guards for Dynamo
            pruned_configs = pruned_configs.unpack_var_sequence(tx)
            # guard_as_python_constant inserts some guards for Dynamo to check if the configs object changed.
            pruned_configs = [
                config.guard_as_python_constant() for config in pruned_configs
            ]
            # after pruning the configs, create a new autotuner object with
            # these configs and recurise.
            new_kernel = autotune(configs=pruned_configs, key=[])(variable.kernel.fn)
            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = type(variable)(new_kernel, None, variable.grid)
            return self.call_triton_kernel(new_var, args, kwargs, tx)

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

                new_kernel = autotune(configs=new_configs, key=[])(variable.kernel.fn)
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
                            assert (
                                name in config.__dict__
                            ), f"{name} must be in autotuning configs to be used as a kernel parameter"
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

        if variable.grid is None:
            self.raise_unsupported("Triton kernels should always be called with a grid")

        # Both for grid's meta as well as for the kernel, we need combined
        # args and kwargs combined and normalized
        combined_args_raw = {**dict(zip(variable.kernel.arg_names, args)), **kwargs}

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
            constexprs = variable.kernel.constexprs
        else:
            # If we are looking at an @triton.autotune decorator, the nested function should be a JITFunction
            # This is because we don't support @triton.heuristics or nested @triton.autotune decorators yet
            assert isinstance(variable.kernel, Autotuner)
            constexprs = variable.kernel.fn.constexprs

        for idx, arg_name in enumerate(variable.kernel.arg_names):
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
    ) -> Tuple[Union[int, sympy.Expr, SymInt], ...]:
        assert tx is None
        assert isinstance(meta, dict)
        assert callable(grid)
        return grid(meta)

    def call_user_defined_fn(
        self, user_fn: Callable[..., Any], args: List, kwargs: Dict
    ) -> Any:
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert callable(user_fn)
        return user_fn(*args, **kwargs)

    def check_grid(
        self,
        grid: "TritonGridType",
    ) -> Tuple[Union[int, sympy.Expr, SymInt], ...]:
        if not isinstance(grid, collections.abc.Sequence):
            raise RuntimeError(
                "wrap_triton can only handle grids that resolve to Sequence[int]."
            )
        # normalize to tuple
        return tuple(grid)

    def call_HOP(
        self,
        variable: "TraceableTritonKernelWrapper",
        grids: List["TritonGridTupleType"],
        combined_args: Dict[str, Any],
        tx: None,
    ) -> None:
        assert tx is None
        assert isinstance(variable, TraceableTritonKernelWrapper)

        def is_graphable(val: Any) -> bool:
            return isinstance(val, fx.node.base_types)

        non_graphable_args = {
            k: v for k, v in combined_args.items() if not is_graphable(v)
        }
        graphable_args = {k: v for k, v in combined_args.items() if is_graphable(v)}

        constant_args_idx = kernel_side_table.add_constant_args(non_graphable_args)
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
        self.kernel = None
        self.grid = None
        tracing_triton_hopifier_singleton.init_variable(self, kernel, kernel_idx, grid)
        assert self.kernel is not None

    def __getitem__(self, *args: Sequence[Any]) -> "TraceableTritonKernelWrapper":
        return tracing_triton_hopifier_singleton.call_getitem(self, args)  # type: ignore[return-value]

    def run(self, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_triton_hopifier_singleton.call_run(self, args, kwargs, None)
        else:
            assert self.kernel is not None
            return self.kernel.run(*args, **kwargs)

    def __call__(self, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_triton_hopifier_singleton.call_triton_kernel(
                self, args, kwargs, None
            )
        else:
            assert self.kernel is not None
            return self.kernel[self.grid](*args, **kwargs)

    def specialize_symbolic(self, arg: Sequence[Any]) -> Any:
        import torch

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, (torch.SymInt, torch.SymBool, torch.SymFloat)):
            return guard_scalar(arg)
        return arg
