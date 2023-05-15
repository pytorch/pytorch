import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import typing
from collections import namedtuple
from itertools import chain

import sympy
from sympy.printing.printer import Printer

import torch

from .. import metrics
from ..utils import (
    DeferredLineBase,
    free_symbol_startswith,
    IndentedBuffer,
    sympy_dot,
    sympy_subs,
    sympy_symbol,
    unique,
)
from ..virtualized import ops, V

log = logging.getLogger(__name__)


def data_type_logger(msg):
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Data type propagation: %s", msg)


TensorArg = namedtuple("TensorArg", ["name", "buffer", "dtype"])
SizeArg = namedtuple("SizeArg", ["name", "expr"])


def index_prevent_reordering(index: typing.List[sympy.Expr], index_vars, sizes):
    from ..ir import FlexibleLayout

    # added contiguous index prevents reordering
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


def _data_type_propagation(sub_graph: torch.fx.Graph):
    def propagate_node(node: torch.fx.Node):
        _node: torch.fx.Node = node
        ops_to_bool = [
            "is_inf",
            "is_nan",
            "bitwise_xor",
            "logical_not",
            "signbit",
            "le",
            "lt",
            "ge",
            "gt",
            "eq",
            "ne",
        ]
        ops_with_dtype_arg = ["constant", "to_dtype", "rand", "randn"]
        reduction_to_dtype = {
            "any": torch.bool,
            "argmin": torch.int64,
            "argmax": torch.int64,
        }
        ops_without_dtype = ["ops", "get_index"]
        if _node.target in ops_without_dtype:
            return False
        if OptimizationContext.key in _node.meta:
            opt_ctx = _node.meta[OptimizationContext.key]
        else:
            opt_ctx = OptimizationContext()
        if opt_ctx.dtype is not None:
            return False
        if _node.target in ops_to_bool:
            opt_ctx.dtype = torch.bool
        elif _node.target in ops_with_dtype_arg:
            opt_ctx.dtype = _node.args[-1]
        elif _node.target == "reduction":
            reduction_type = _node.args[4]
            if reduction_type in reduction_to_dtype:
                opt_ctx.dtype = reduction_to_dtype[reduction_type]
        elif _node.target == "load":
            opt_ctx.dtype = V.graph.get_dtype(_node.args[1])
        if opt_ctx.dtype is not None:
            data_type_logger(
                f"for node.target = {_node.target}, dtype is propagated to {opt_ctx.dtype}"
            )
            _node.meta[OptimizationContext.key] = opt_ctx
            return True

        # node.target not belong to any ops which can directly get the dtype
        # need propogate dtype with it's input node
        dtype = None
        inputs = node.all_input_nodes
        input_nodes = [
            n
            for n in inputs
            if isinstance(n, torch.fx.node.Node) and n.target not in ops_without_dtype
        ]
        if len(input_nodes) == 0:
            return False
        all_input_nodes_propogated = all(
            OptimizationContext.key in n.meta
            and n.meta[OptimizationContext.key].dtype is not None
            for n in input_nodes
        )
        if not all_input_nodes_propogated:
            return False
        # all input nodes have propogated dtype, we will promot to dtype with highest precision
        dtype = functools.reduce(
            torch.promote_types,
            [n.meta[OptimizationContext.key].dtype for n in input_nodes],
        )
        opt_ctx.dtype = dtype
        msg = f"for node.target = {_node.target}, dtype is propagated to {opt_ctx.dtype}, "
        input_msg = "inputs dtypes: "
        for n in input_nodes:
            input_msg += (
                f"input {n.name}.dtype = {n.meta[OptimizationContext.key].dtype}"
            )
        data_type_logger(msg + input_msg)
        _node.meta[OptimizationContext.key] = opt_ctx
        return True

    new_node_propogated = False
    for node in sub_graph.nodes:
        new_node_propogated = propagate_node(node) or new_node_propogated

    if new_node_propogated:
        _data_type_propagation(sub_graph)


def data_type_propagation(node):
    from ..ir import LoopBody
    from ..scheduler import SchedulerNode

    assert isinstance(node, SchedulerNode)
    _node: SchedulerNode = node
    if isinstance(_node._body, LoopBody):
        body: LoopBody = node._body
        sub_blocks = [body.root_block] + list(body.subblocks.values())
        for sub_block in sub_blocks:
            _sub_graph: torch.fx.Graph = sub_block.graph
            _data_type_propagation(_sub_graph)


class ExprPrinter(Printer):
    @staticmethod
    def paren(string):
        def all_in_parens(string):
            if string[0] != "(" or len(string) < 2:
                return False
            count = 1
            for i, char in enumerate(string[1:]):
                if char == "(":
                    count += 1
                elif char == ")":
                    count -= 1
                if count == 0 and i != len(string) - 2:
                    return False
            assert count == 0
            return True

        if (
            isinstance(string, CSEVariable)
            or re.match(r"^[a-z0-9_.]+$", string, re.I)
            or re.match(r"^\([^)]*\)$", string, re.I)
            or string == ""
        ):
            return string
        # don't put extra parens for strings that are already wrapped in parens
        if all_in_parens(string):
            return string
        return f"({string})"

    def _print_Pow(self, expr):
        # Pow() confuses triton
        base, exp = expr.args
        base = self._print(base)
        # NB: Remember this is sizevar computation!  You don't typically
        # expect to have to do floating point computation including exponents
        # in sizevar compute.  Instead of adding support for floating
        # point pow, you should make upstream retranslate the Sympy expression
        # into Tensor expressions earlier and do that instead.
        if exp == 0.5:
            return f"math.sqrt({base})"
        assert exp == int(exp), exp
        exp = int(exp)
        if exp > 0:
            return "*".join([self.paren(base)] * exp)
        elif exp < 0:
            return "1/" + self.paren("*".join([self.paren(base)] * abs(exp)))
        else:  # exp == 0
            return "1"

    def _print_Mul(self, expr):
        return "*".join(map(self.paren, map(self._print, expr.args)))

    def _print_Add(self, expr):
        return " + ".join(map(self.paren, map(self._print, expr.args)))

    def _print_Mod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_CleanDiv(self, expr):
        return self._print_FloorDiv(expr)


class PythonPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"math.floor({self._print(expr.args[0])})"

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        return f"math.ceil({self._print(expr.args[0])})"


class OpOverrides:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def identity(value):
        # used to trigger cse
        return value

    @staticmethod
    def constant(value, dtype):
        return repr(value)

    @staticmethod
    def reciprocal(x):
        return ops.div("1", x)

    @staticmethod
    def square(x):
        return ops.mul(x, x)

    @staticmethod
    def bitwise_not(x):
        return f"~{ExprPrinter.paren(x)}"

    @staticmethod
    def logical_not(a):
        return f"{ExprPrinter.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x, y):
        return f"{ExprPrinter.paren(x)} & {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_or(x, y):
        return f"{ExprPrinter.paren(x)} | {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_xor(x, y):
        return f"{ExprPrinter.paren(x)} ^ {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_left_shift(x, y):
        return f"{ExprPrinter.paren(x)} << {ExprPrinter.paren(y)}"

    # TODO(fdrocha): this is currently not being used anywhere,
    # pending on moving triton pin past 972b761
    @staticmethod
    def bitwise_right_shift(x, y):
        return f"{ExprPrinter.paren(x)} >> {ExprPrinter.paren(y)}"

    @staticmethod
    def remainder(a, b):
        r = ops.mod(a, b)
        return ops.where(f"(({r} != 0) & (({r} < 0) != ({b} < 0)))", ops.add(r, b), r)


class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name, line):
        super().__init__(line)
        self.name = name

    def __call__(self):
        if (
            self.name not in V.graph.removed_buffers
            and self.name not in V.graph.inplaced_to_remove
        ):
            return self.line
        return None

    def _new_line(self, line):
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(typing.NamedTuple):
    inner_name: str
    other_names: typing.List[str]


class KernelArgs:
    @staticmethod
    def _lookup(prefix, odict, name):
        assert isinstance(name, (str, sympy.Symbol))
        if name not in odict:
            odict[name] = f"{prefix}{len(odict)}"
        return odict[name]

    def __init__(self, sizevars=None):
        self.input_buffers = dict()
        self.output_buffers = dict()
        self.inplace_buffers = dict()
        self.sizevars = sizevars or dict()

    def __repr__(self):
        return "KernelArgs({})".format(
            ", ".join(
                map(
                    repr,
                    [
                        self.input_buffers,
                        self.output_buffers,
                        self.inplace_buffers,
                        self.sizevars,
                    ],
                )
            )
        )

    def input(self, name):
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.output_buffers:
            return self.output_buffers[name]
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name):
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name, output_name):
        assert output_name not in self.inplace_buffers
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        else:
            buf = InplacedBuffer(
                f"in_out_ptr{len(unique(self.inplace_buffers.values()))}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    def size(self, name):
        if str(name) == "seed":
            self.sizevars["seed"] = "seed"
            return "seed"
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def wrap_ptr_arg(self, buf, dtype):
        return f"c_void_p({buf}.data_ptr())"

    def wrap_size_arg(self, size):
        return f"c_long({size})"

    def cpp_argdefs(self):
        from .cpp import DTYPE_TO_CPP, INDEX_TYPE

        # TODO(jansel): replace this with data from scheduler
        buffer_types = {x.get_name(): x.get_dtype() for x in V.graph.buffers}
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                if val.is_integer:
                    buffer_types[name] = torch.int64
                else:
                    buffer_types[name] = torch.float64
            else:
                buffer_types[name] = val.get_dtype()
        buffer_types.update(
            {name: val.dtype for name, val in V.graph.constants.items()}
        )

        call_args = []
        arg_defs = []
        arg_types = []
        for inplaced in unique(self.inplace_buffers.values()):
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = buffer_types[outer]
            cpp_dtype = DTYPE_TO_CPP[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = buffer_types[outer]
            cpp_dtype = DTYPE_TO_CPP[dtype]
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"const {cpp_dtype}*")
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or inner == "REMOVED":
                continue
            dtype = buffer_types[outer]
            cpp_dtype = DTYPE_TO_CPP[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.sizevars.items():
            arg_defs.append(f"const {INDEX_TYPE} {inner}")
            call_args.append(self.wrap_size_arg(outer))
            arg_types.append(f"const {INDEX_TYPE}")
        return arg_defs, call_args, arg_types

    def python_argdefs(self):
        arg_defs = []
        call_args = []
        precompile_args = []
        for inplaced in unique(self.inplace_buffers.values()):
            arg_defs.append(inplaced.inner_name)
            call_args.append(inplaced.other_names[-1])
            precompile_args.append(
                TensorArg(
                    inplaced.inner_name,
                    inplaced.other_names[-1],
                    V.graph.get_dtype(inplaced.other_names[-1]),
                )
            )
        for outer, inner in chain(
            self.input_buffers.items(), self.output_buffers.items()
        ):
            if outer in self.inplace_buffers or inner == "REMOVED":
                continue
            arg_defs.append(inner)
            call_args.append(outer)
            precompile_args.append(TensorArg(inner, outer, V.graph.get_dtype(outer)))
        for outer, inner in self.sizevars.items():
            arg_defs.append(inner)
            call_args.append(str(outer))
            precompile_args.append(SizeArg(inner, outer))

        return arg_defs, call_args, precompile_args

    def aliases(self):
        for inplaced in unique(self.inplace_buffers.values()):
            for other in inplaced.other_names:
                if other in V.graph.inplaced_to_remove:
                    continue
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                if other in self.output_buffers:
                    yield self.output_buffers[other], inplaced.inner_name

    def is_removed(self, name):
        def _is_removed(name, buffers):
            return name not in buffers or buffers[name] == "REMOVED"

        return _is_removed(name, self.output_buffers) and _is_removed(
            name, self.inplace_buffers
        )

    # Includes inplace buffers, excludes removed buffers.  Essentially,
    # after you do a call into this kernel, which buffers actually contain
    # updated data?  Modeled off of python_argdefs.
    def live_output_buffers(self):
        live_outs = set()
        for inplaced in unique(self.inplace_buffers.values()):
            live_outs.add(inplaced.other_names[-1])
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or inner == "REMOVED":
                continue
            live_outs.add(outer)
        return live_outs


class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    The backends can inherit from this class and overload the "create_cse_var" Kernel to do that.
    The "update_on_args" method gives you a hook for annotations, see example of TritonCSEVariable in triton.py.
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return type(other) == type(self) and other.name == self.name

    def update_on_args(self, name, args, kwargs):
        pass


class CppWrapperKernelArgs(KernelArgs):
    def wrap_ptr_arg(self, buf, dtype):
        from .cpp import DTYPE_TO_CPP

        return f"({DTYPE_TO_CPP[dtype]}*)({buf}.data_ptr())"

    def wrap_size_arg(self, size):
        return f"{size}"


class CSE:
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix="",
        suffix="",
        name_prefix="tmp",
        iter_buffers=None,
        store_cache=None,
        reduction_cache=None,
        varname_map=None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.cache = {}
        self.name_prefix = name_prefix
        self.store_cache = store_cache or {}
        self.reduction_cache = reduction_cache or {}
        self.iter_buffer_ids = iter_buffers or itertools.count()
        self.invalidated_stores = set()
        self.varname_map = varname_map or {}

    def invalidate(self, keep_vars: typing.Set[str]):
        for name, tmp in list(self.store_cache.items()):
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        self.cache = {k: v for k, v in self.cache.items() if v in keep_vars}

    def clone(self):
        # Note(fdrocha): reduction_cache is not being cloned, not sure if this is intentional
        return CSE(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
        )

    def generate(
        self,
        buffer: IndentedBuffer,
        expr: typing.Union[str, CSEVariable],
        write=True,
        assignment=True,
    ) -> CSEVariable:
        assert isinstance(expr, (str, CSEVariable)), type(expr)
        assert write or assignment
        if isinstance(expr, CSEVariable):
            return expr
        cache_key = expr
        if cache_key not in self.cache:
            var = self.newvar() if assignment else None
            self.cache[cache_key] = var
            if write:
                if V.kernel.current_node:
                    V.kernel.current_node.codegen_originating_info(
                        buffer, only_once=True
                    )
                if assignment:
                    line = f"{self.prefix}{var} = {expr}{self.suffix}"
                else:
                    line = f"{expr}{self.suffix}"
                buffer.writeline(line)

        return self.cache[cache_key]

    def newvar(self) -> CSEVariable:
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
        var = V.kernel.create_cse_var(var_name)
        self.varname_map[var_name] = var
        return var


class CodeGen:
    def __init__(self):
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class Kernel(CodeGen):
    newvar_prefix = ""
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, args=None):
        super().__init__()
        metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers = set()
        self.current_node = None
        self.store_buffer_names = set()

    @contextlib.contextmanager
    def set_current_node(self, node):
        prior = self.current_node
        self.current_node = node
        try:
            yield
        finally:
            self.current_node = prior

    @contextlib.contextmanager
    def swap_buffers(self, lb, cb=None, sb=None):
        if cb is None:
            cb = lb
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.clone()
        try:
            yield
        finally:
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse

    def load(self, name: str, index: sympy.Expr):
        raise NotImplementedError()

    def indirect_load(self, name: str, index: sympy.Expr):
        """A load the depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store(self, name, index, value, mode=None):
        raise NotImplementedError()

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        raise NotImplementedError()

    def __enter__(self):
        class CSEProxy:
            self.name = "CSEProxy"

            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    csevar = self.cse.generate(
                        self.compute, getattr(parent_handler, name)(*args, **kwargs)
                    )
                    csevar.update_on_args(name, args, kwargs)
                    return csevar

                return inner

            @staticmethod
            def indirect_indexing(index_var, size):
                return sympy_symbol(str(index_var))

            @staticmethod
            def load(name: str, index: sympy.Expr):
                if name in self.cse.invalidated_stores:
                    # A load from an invalidated store requires us to
                    # keep the actual buffer around
                    V.kernel.must_keep_buffers.add(name)
                if free_symbol_startswith(index, "tmp"):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if name in store_cache:
                    return store_cache[name]
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                self.store_buffer_names.add(name)
                if mode is None:
                    self.cse.store_cache[name] = value
                    if self.current_node:
                        for other_name in self.current_node.get_mutations():
                            self.cse.store_cache[other_name] = value
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(name, dtype, src_dtype, reduction_type, index, value):
                self.store_buffer_names.add(name)
                return self.reduction(
                    name, dtype, src_dtype, reduction_type, index, value
                )

        super().__enter__()
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if V.graph.scheduler:
            V.graph.scheduler.remove_kernel_local_buffers()
        super().__exit__(exc_type, exc_val, exc_tb)

    def rename_indexing(self, index) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.args.size(x)
            for x in sorted_symbols
            if x.name.startswith("s") or x.name.startswith("ps")
        }
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args, **kwargs):
        return CSEVariable(*args, **kwargs)


@dataclasses.dataclass
class OptimizationContext:
    key: typing.ClassVar[str] = "opt_ctx"

    # Load value as mask
    is_load_as_mask: bool = False
    # Load bfloat16 value as float32
    is_load_bf16_as_fp32: bool = False
    # Store float32 value as bfloat16
    is_store_fp32_as_bf16: bool = False
    # do not  need type cast for
    # for mem copy only node bf16 load -> bf16 store,
    is_bf16_mem_copy: bool = False

    dtype: torch.dtype = None
    ops_name: str = ""
    is_most_inner_loop_irrevelant: bool = False

    # Load uint8 value as float32
    is_load_uint8_as_float: bool = False
    # Store float32 value as uint8
    is_store_float_as_uint8: bool = False
