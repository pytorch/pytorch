import dataclasses
import logging
import threading
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

log = logging.getLogger("torch._dynamo")


###############################################################################
# Kernel Side Table


# We cannot put Triton Kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable:
    id_to_kernel: Dict[int, Any] = dict()
    kernel_to_id: Dict[Any, int] = dict()
    lock = threading.Lock()

    # Returns index on the table
    def add_kernel(self, kernel) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the triton kernel at the given index
    def get_kernel(self, idx: int):
        # No need to lock here as fetching from dict is atomic
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = dict()
        self.kernel_to_id = dict()


kernel_side_table = KernelSideTable()


###############################################################################
# Mutation Tracker


@dataclasses.dataclass(frozen=True)
class Param:
    idx: int


@dataclasses.dataclass(frozen=True)
class Intermediate:
    idx: int

    def fake(self):
        return self.idx < 0


@dataclasses.dataclass(frozen=True)
class Op:
    name: str
    fn_call_name: Optional[str]
    args: List[Union[Param, Intermediate]]
    ret: Intermediate = dataclasses.field(repr=False)

    def __post_init__(self):
        if self.name == "tt.call":
            assert self.fn_call_name is not None
        else:
            assert self.fn_call_name is None


def generate_ttir(kernel, kwargs):
    """
    Uses Triton's internal code generation to create TTIR
    """
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    import torch
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            # If we are autotuning, then it doesn't matter which version gets
            # picked for tracing purposes, so lets pick the first one
            kwargs = {**kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    assert isinstance(kernel, JITFunction)

    if len(kwargs) != len(kernel.arg_names):
        raise Exception("Incorrect number of arguments passed to kernel")

    # Replace all SymExprs with a regular value for TTIR generation
    # Replace all FakeTensor with real tensors
    # These replacements are needed for triton's type, key and config functions
    ordered_args: Dict[str, Any] = {}
    for name in kernel.arg_names:
        a = kwargs[name]
        if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            ordered_args[name] = 2
        elif isinstance(a, FakeTensor):
            ordered_args[name] = torch.empty(2, dtype=a.dtype)
        else:
            ordered_args[name] = a

    ordered_tensor_names = [
        name for name, arg in ordered_args.items() if isinstance(arg, Tensor)
    ]
    specialization = kernel._get_config(*ordered_args.values())
    constants = {
        i: arg
        for i, arg in enumerate(ordered_args.values())
        if not isinstance(arg, Tensor)
    }

    # Build kernel signature -- doesn't include constexpr arguments.
    signature = {
        i: kernel._type_of(kernel._key_of(arg))
        for i, arg in enumerate(ordered_args.values())
        if i not in kernel.constexprs
    }

    def get_backend():
        from triton.compiler.backends.cuda import CUDABackend
        from triton.runtime.driver import driver

        target = driver.get_current_target()
        return CUDABackend(target)

    backend = get_backend()

    options = backend.parse_options(dict())
    # triton._C.libtriton.triton.ir.load_dialects(context)
    # backend.load_dialects(context)

    src = ASTSource(kernel, signature, constants, specialization)
    ttir_module = src.make_ir(options)
    if not ttir_module.verify():
        raise Exception("Verification for TTIR module has failed")

    return ttir_module, ordered_tensor_names


def ttir_to_functions(ttir_module) -> Dict[str, Dict[Intermediate, List[Op]]]:
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

    def reindex(idx):
        if idx not in reindex_map:
            reindex_map[idx] = len(reindex_map)
        return reindex_map[idx]

    def mlir_to_functions(op) -> None:
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
                        if isinstance(arg, Intermediate) and arg.idx in replacements:
                            fn_op.args[i] = replacements[arg.idx]

            # next function capture starts
            # with empty replacements
            replacements.clear()

            fn_name = op.get_str_attr("sym_name")
            functions[fn_name] = fn_ops
        elif child_block_ids:
            if name in ("scf.if", "scf.for", "scf.while"):
                # for blocked control flow ops: inline the enclosed
                # ops into the parent block + rewire the last op in
                # each child block (yield) to return the scf result
                yield_ops = []
                for block_id in child_block_ids:
                    # the block args used as operands of the ops in the block
                    # (and nested blocks inlined in the current block by now)
                    # are replaced by new fake Intermediates to avoid "this
                    # operand is not returned by anything other op in the fn"
                    # error in the downstream analysis
                    for idx in block_id_to_block_arg_ids[block_id]:
                        next_fake_intermediate -= 1
                        replacements[idx] = Intermediate(next_fake_intermediate)

                    if block_id in op_stack:
                        block_ops = op_stack.pop(block_id)
                        if not block_ops:
                            continue
                        last_ret, last_ops = block_ops.popitem()
                        if all(op.name == "scf.yield" for op in last_ops):
                            # if last_ops are scf.yield, treat them separately
                            yield_ops.extend(last_ops)
                        else:
                            # otherwise, return last_ops to the block
                            block_ops[last_ret] = last_ops
                        for op_result, child_ops in block_ops.items():
                            op_stack[parent_block_id][op_result].extend(child_ops)

                scf_results = [Intermediate(idx) for idx in result_ids]
                for scf_result in scf_results:
                    for yield_op in yield_ops:
                        op_stack[parent_block_id][scf_result].append(yield_op)
            else:
                # TODO(oulgen): add support for tt.reduce
                raise Exception(
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


def parse_ttir(ttir, kwargs):
    """
    Given a Triton emitted TTIR text, this function lexes and parses the
    code using a minimal grammar defined inside. During the lexing/parsing,
    we drop any constant value and type information as they are not
    necessary to us.
    Being able to choose what we need makes this not a general purpose TTIR
    parser which further makes parsing much simpler.
    """
    # TODO(oulgen):
    # - Support closures (e.g. "tt.reduce")

    try:
        import lark  # type: ignore[import-not-found]
        from lark import Lark, Transformer, v_args
    except ModuleNotFoundError:
        warnings.warn(
            "Using slow path for user-defined Triton kernels. `pip install lark` to fix this."
        )
        raise

    # Ops looks like one of the following forms:
    #
    # %14 = tt.addptr %13, %4 : tensor<4x!tt.ptr<f32, 1>>, tensor<4xi32>
    # tt.store %14, %12, %5 {cache = 1 : i32, evict = 1 : i32} : tensor<4xf32>
    # %15 = "tt.atomic_rmw"(%14, %12, %5) <{atomic_rmw_op = 5 : i32, scope = 1 : i32, sem = 4 : i32}> : (tensor<4x!tt.ptr<f32, 1>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>  # noqa: B950
    grammar = """
        start: (module_block | loc_line)+

        loc_line: "#loc" /.+/ NEWLINE

        module_block: "module" "{" func_block+ "}" LOC

        func_block: "tt.func" ("public"|"private") FN_NAME "(" /.+/ NEWLINE stmt* "}" LOC -> process_func

        ?stmt: op | if | for | while | condition_stmt | label_stmt | cf_stmt

        if: [assign_lhs "="] "scf.if" args rest stmt* "}" "else" "{" stmt* "}" LOC -> process_if
        for: [assign_lhs "="] "scf.for" args rest stmt* "}" divisibility_annot? LOC -> process_for
        while: [assign_lhs "="] "scf.while" args rest stmt* "}" "do" "{" stmt* "}" LOC -> process_while

        condition_stmt: "scf.condition" "(" arg ")" args rest
        label_stmt: LABEL ":" "// pred:" LABEL
                  | LABEL "(" /.+/ NEWLINE
        cf_stmt: "cf" "." NAME /.+/ NEWLINE

        op: OP_NAME LOC
          | [assign_lhs "="] OP_NAME [FN_NAME] args rest?  -> process_op

        ?rest: (":" | "{" | "\\"" | "->" | "<" | "=") /.+/ NEWLINE
        divisibility_annot: "{" "tt.divisibility_arg1" /[^}]+/ "}"

        args: | "(" ")" | "("? arg ("," arg)* ")"?

        ?arg: INTERMEDIATE
            | INTERMEDIATE_CONSTANT
            | CONSTANT
            | PARAM
            | "[" args "]"
            | arg_with_index

        ?arg_with_index: arg "#" DIGIT+

        ?assign_lhs: (INTERMEDIATE | INTERMEDIATE_CONSTANT) [":" DIGIT+]

        PARAM.5: "%arg" DIGIT+
        INTERMEDIATE.4: "%" DIGIT+
        INTERMEDIATE_CONSTANT.3: "%" NAME
        CONSTANT: FLOAT | DIGIT+ | NAME ("<" DIGIT+ ">")?
        LABEL: "^bb" DIGIT+

        NAME: (LETTER | DIGIT | "_")+
        NON_CF_NAME: /(?!(cf))/ NAME
        FN_NAME: "@" (NAME | ESCAPED_STRING)
        OP_NAME: "\\""? NON_CF_NAME ("." NAME)+ "\\""?

        LOC.5: "loc(#loc" DIGIT* ")"

        %import common.LETTER
        %import common.DIGIT
        %import common.WS
        %import common.NEWLINE
        %import common.ESCAPED_STRING
        %import common.FLOAT
        %ignore WS
    """

    next_fake_intermediate = 0

    def convert(token):
        if isinstance(token, lark.tree.Tree):
            if token.data == "args":
                res = []
                for a in token.children:
                    c = convert(a)
                    if isinstance(c, list):
                        res.extend(c)
                    else:
                        res.append(c)
                return res
            elif token.data in {"assign_lhs", "arg_with_index"}:
                # Drop length/index qualifier
                return convert(token.children[0])
            else:
                raise AssertionError(f"Tree node with {token.data}")

        if token is None or (
            isinstance(token, lark.lexer.Token)
            and token.type in ("CONSTANT", "INTERMEDIATE_CONSTANT")
        ):
            nonlocal next_fake_intermediate
            next_fake_intermediate -= 1
            return Intermediate(next_fake_intermediate)

        assert isinstance(token, lark.lexer.Token)

        if token.type == "INTERMEDIATE":
            return Intermediate(int(token.value[len("%") :]))
        if token.type == "PARAM":
            return Param(int(token.value[len("%arg") :]))

        raise AssertionError(f"{type(token.type)} => {token.value} invalid")

    # In alternative representation, function names are quoted.
    # It should be possible to move this into the grammar alltogether.
    def convert_name(token):
        if token is None:
            return None
        s = token.value
        if len(s) > 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    functions: Dict[str, Dict[Intermediate, List[Op]]] = {}

    def extend_dict_list(d1, d2):
        for key, values in d2.items():
            d1[key].extend(values)

    @v_args(inline=True)
    class TransformOps(Transformer):
        def process_op(self, ret, op_name, fn_name, args, *rest):
            return Op(
                convert_name(op_name),
                convert_name(fn_name),
                convert(args),
                convert(ret),
            )

        def process_func(self, name, _args, *stmts):
            ops: Dict[Intermediate, List[Op]] = defaultdict(list)
            for e in stmts:
                if isinstance(e, Op):
                    ops[e.ret].append(e)
                elif isinstance(e, dict):
                    extend_dict_list(ops, e)
            functions[name.value] = ops

        def _process_scf(self, ret, stmts):
            ret = convert(ret)
            ops: Dict[Intermediate, List[Op]] = defaultdict(list)
            for e in stmts:
                if isinstance(e, Op):
                    if e.name == "scf.yield":
                        ops[ret].append(Op(e.name, None, e.args, ret))
                    else:
                        ops[e.ret].append(e)
                elif isinstance(e, dict):
                    extend_dict_list(ops, e)
            return ops

        def process_if(self, ret, _args, _rest, *stmts):
            return self._process_scf(ret, stmts)

        def process_for(self, ret, _args, _rest, *stmts):
            return self._process_scf(ret, stmts)

        def process_while(self, ret, _args, _rest, *stmts):
            return self._process_scf(ret, stmts)

    parser = Lark(
        grammar, parser="lalr", maybe_placeholders=True, transformer=TransformOps()
    )
    parser.parse(ttir)
    return functions


class MemoizeWithCycleCheck:
    def __init__(self, fn):
        self.fn = fn
        self.reset()

    def __call__(self, functions, fn_name, num_args):
        key = (fn_name, num_args)
        if key not in self.cache:
            self.cache[key] = None
            self.cache[key] = self.fn(functions, fn_name, num_args)
        if self.cache[key] is None:
            raise Exception("Recursion is not supported")
        return self.cache[key]

    def reset(self):
        self.cache = {}


@MemoizeWithCycleCheck
def analyze_kernel_mutations(functions, fn_name, num_args):
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
    MUTATION_OPS = {"tt.store": [0], "tt.atomic_cas": [0], "tt.atomic_rmw": [0]}
    # Ops that we want to bail out on
    UNKNOWN_OPS = {"tt.elementwise_inline_asm"}

    stack: List[Union[Param, Intermediate]] = []
    visited = set()
    ops = functions[fn_name]
    for op_list in ops.values():
        for op in op_list:
            if op.name in UNKNOWN_OPS:
                raise Exception(
                    f"ttir analysis hit an op we do not know how to analyze: {op.name}"
                )

            if op.name == "tt.call":
                assert op.fn_call_name in functions
                mutations = analyze_kernel_mutations(
                    functions, op.fn_call_name, len(op.args)
                )
                stack.extend(arg for arg, mutated in zip(op.args, mutations) if mutated)
            else:
                for idx in MUTATION_OPS.get(op.name, []):
                    stack.append(op.args[idx])

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


def identify_mutated_tensors(kernel, kwargs):
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect all input tensor mutations
    """

    ttir_module = None
    functions = None
    try:
        from torch._dynamo import config

        if not config.optimize_user_defined_triton_kernels:
            raise Exception("optimize_user_defined_triton_kernels is False")

        ttir_module, ordered_tensor_names = generate_ttir(kernel, kwargs)

        # extract functions from TTIR
        if hasattr(ttir_module, "walk"):
            # use MLIR bindings exposed by Triton code
            functions = ttir_to_functions(ttir_module)
        else:
            # parse string representation of Triton IR
            functions = parse_ttir(str(ttir_module), kwargs)

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
    except Exception as e:
        import traceback

        warnings.warn(
            "Encountered an exception in identify_mutated_tensors, "
            "assuming every input is mutated:\n"
            "".join(
                traceback.TracebackException.from_exception(e).format()  # noqa: G001
            )
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
    def __init__(self):
        super().__init__("triton_kernel_wrapper_mutation")


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# Used for wrapping a Triton Kernel in a functional manner
class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_functional")


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(*, kernel_idx, grid, kwargs):
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(
            kernel.fn.__name__, kernel.configs, grid
        )
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]

    kernel[grid_fn](**kwargs)


@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(mode, *, kernel_idx, grid, kwargs):
    with mode:
        return None


def trace_triton_kernel_wrapper(proxy_mode, func_overload, node_args):
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode, *, kernel_idx, grid, kwargs
):
    if mode.enable_tracing:
        trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_mutation,
            {"kernel_idx": kernel_idx, "grid": grid, "kwargs": kwargs},
        )
    else:
        triton_kernel_wrapper_mutation(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)

    return None


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(ctx, kernel_idx, grid, kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    kernel = kernel_side_table.get_kernel(kernel_idx)
    # TODO(oulgen): Preexisting bug, if two kernel inputs are views of each
    # other, and one gets mutated in kernel, and later another gets mutated,
    # they are no longer equal. Fix this by graph breaking on this condition
    # earlier in dynamo.
    tensors_to_clone = identify_mutated_tensors(kernel, unwrapped_kwargs)
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
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
        # sync calls replace_ under the hood, so again indicate that
        # this indirect replace is hidden from autograd
        ctx.mark_mutation_hidden_from_autograd(input_arg)
    return None


@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(
    *, kernel_idx, grid, kwargs, tensors_to_clone
):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    kwargs = {
        key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
        for key, val in kwargs.items()
    }
    triton_kernel_wrapper_mutation(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)
    return {key: val for key, val in kwargs.items() if key in tensors_to_clone}


@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
def triton_kernel_wrapper_functional_fake_tensor_mode(
    mode, *, kernel_idx, grid, kwargs, tensors_to_clone
):
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
    mode, *, kernel_idx, grid, kwargs, tensors_to_clone
):
    if mode.enable_tracing:
        return trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_functional,
            {
                "kernel_idx": kernel_idx,
                "grid": grid,
                "kwargs": kwargs,
                "tensors_to_clone": tensors_to_clone,
            },
        )
    else:
        return triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=kwargs,
            tensors_to_clone=tensors_to_clone,
        )


@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(
    ctx, kernel_idx, grid, kwargs, tensors_to_clone
):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )
        return ctx.wrap_tensors(outputs)


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
