import copy
import functools
import inspect
import itertools
from typing import Callable, List, Union, Tuple, Optional
import operator

import torch
from torch import fx
from torch._C import _te  # type: ignore[attr-defined]
from functorch._C import CompileCache, CompileResult

FOLD_ALIASES = True
_SHAPE_TYPES = {"one", "other"}
_STRIDE_TYPES = {"zero", "one", "contiguous", "transposed_contiguous", "as_arg"}
_identity = lambda x: x
_TORCH_TO_EXPR_MAP = {
    "sin": _te.sin,
    "cos": _te.cos,
    "tan": _te.tan,
    "asin": _te.asin,
    "acos": _te.acos,
    "atan": _te.atan,
    "sinh": _te.sinh,
    "cosh": _te.cosh,
    "tanh": _te.tanh,
    "sigmoid": _te.sigmoid,
    "exp": _te.exp,
    "expm1": _te.expm1,
    "abs": _te.abs,
    "log": _te.log,
    "log2": _te.log2,
    "log10": _te.log10,
    "log1p": _te.log1p,
    "erf": _te.erf,
    "erfc": _te.erfc,
    "sqrt": _te.sqrt,
    "rsqrt": _te.rsqrt,
    "ceil": _te.ceil,
    "floor": _te.floor,
    "round": _te.round,
    "trunc": _te.trunc,
    "frac": _te.frac,
    "lgamma": _te.lgamma,
    "isnan": _te.isnan,
    "add": operator.add,
    "sub": operator.sub,
    "subtract": operator.sub,
    "mul": operator.mul,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "div": operator.truediv,
    "remainder": _te.remainder,
    "fmod": _te.fmod,
    "pow": _te.pow,
    "atan2": _te.atan2,
    "detach": _identity,
    "neg": lambda x: _create_constant(0.0, torch.float32) - x,
}

_int = _te.ExprHandle.int


def _argmax(x):
    return int(torch.argmax(torch.LongTensor(x, device="cpu")))


def _zero():
    return _int(0)


def _one():
    return _int(1)


def _num_args(fn: Callable):
    return len(inspect.signature(fn).parameters)


def _combine_dtype(a: torch.dtype, b: torch.dtype):
    if a == b:
        return a
    # TODO(jansel): find a cleaner way to implement this
    return (
        torch.zeros(1, dtype=a, device="cpu") + torch.zeros(1, dtype=b, device="cpu")
    ).dtype


def _fx_to_expr(fn: Callable, dtype: torch.dtype):
    """Convert the fx graph to equivalent Tensor Expr"""

    def apply(arg):
        if isinstance(arg, (int, float)):
            return gm.graph.create_node("call_function", _create_constant, (arg, dtype))
        return arg

    gm: fx.GraphModule = fx.symbolic_trace(fn)
    for node in list(gm.graph.nodes):
        with gm.graph.inserting_before(node):
            node.args = tuple(apply(a) for a in node.args)
            if node.op == "call_function":
                if node.target.__name__ not in _TORCH_TO_EXPR_MAP:
                    raise NotImplementedError(
                        "Missing mapping from op ",
                        node.target.__name__,
                        " to Tensor Expr",
                    )

                    # Get the parser function to parse the torch op to tensor expr handle

                def _parser(*args, op_name):
                    return _TORCH_TO_EXPR_MAP[op_name](*args)

                new_node = gm.graph.create_node(
                    "call_function",
                    _parser,
                    node.args,
                    {"op_name": node.target.__name__},
                )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
    gm.recompile()
    return gm


def _create_constant(value: Union[int, float], dtype: torch.dtype):
    return _te.Cast.make(
        dtype,
        {int: _te.ExprHandle.int, float: _te.ExprHandle.double}[type(value)](value),
    )


class PointwiseCompiler(object):
    def __init__(
        self,
        name: str,
        module_name: str,
        pointwise_fn: Callable,
        spec: List,
        result: CompileResult,
    ):
        self.name = name
        self.module_name = module_name
        self.pointwise_fn = pointwise_fn
        self.spec = spec
        self.result = result
        self.ndim = max(x.ndim for x in spec)
        self.shapes = [["one"] * (self.ndim - x.ndim) + x.shape for x in spec]
        self.strides = [["zero"] * (self.ndim - x.ndim) + x.stride for x in spec]
        self.shape_flags = copy.deepcopy(self.shapes)
        self.stride_flags = copy.deepcopy(self.strides)
        self.shape_args = [_te.VarHandle(torch.int32) for _ in range(self.ndim)]
        self.shape_vars = list(self.shape_args)
        self.iter_vars = [_te.VarHandle(torch.int32) for _ in range(self.ndim)]
        self.stride_args: List[_te.VarHandle] = []
        self.strides_from: List[Tuple[int, int]] = []
        self.broadcasts: List[Tuple[int, int]] = []
        self.output_order: List[int] = []

        (self.device,) = list(set(x.device.type for x in spec))
        # TODO(jansel): support meta tensors
        self.compile_mode = {"cpu": "llvm", "cuda": "cuda"}[self.device]

        if spec[-1].out:
            self.dtype = spec[-1].dtype
        else:
            self.dtype = functools.reduce(_combine_dtype, [x.dtype for x in spec])

        self.run()

    def add_stride_arg(self, a, d):
        var = _te.VarHandle(torch.int32)
        self.stride_args.append(var)
        self.strides_from.append((a, d))
        return var

    def replace_shape(self, a, d, expected, replacement):
        if self.shapes[a][d] == expected:
            self.shapes[a][d] = replacement()

    def replace_stride(self, a, d, expected, replacement):
        if self.strides[a][d] == expected:
            self.strides[a][d] = replacement()

    def error_checks(self):
        spec = self.spec
        (layout,) = list(set(x.layout for x in spec))
        assert layout == torch.strided, "TODO: support other layouts"
        assert [x.out for x in spec[:-1]] == [False] * (len(spec) - 1)
        assert all(
            shape_type in _SHAPE_TYPES for shape_type in itertools.chain(*self.shapes)
        )
        assert all(
            stride_type in _STRIDE_TYPES
            for stride_type in itertools.chain(*self.strides)
        )

    def make_backwards(self, index: int):
        """
        Compute the derivative of self.pointwise_fn with respect to input number index
        """
        # TODO(jansel): implement this without sympy
        from sympy import symbols, diff  # type: ignore[import]

        vars = symbols([f"v{i}" for i in range(1 + _num_args(self.pointwise_fn))])
        backwards_expr = (
            diff(self.pointwise_fn(*vars[:-1]), vars[index]) * vars[-1]
        )  # chain rule
        return _source_to_pointwise_operator(
            f"lambda {','.join(map(str, vars))}: {backwards_expr}",
            name=f"{self.name}.backwards{index}",
            module_name=self.module_name,
        )

    def handle_autograd(self):
        cnt = sum(int(x.requires_grad) for x in self.spec)
        if cnt == 0:
            return
        assert all(
            x.alias_group == 0 for x in self.spec
        ), "TODO: support aliased backwards"

        for i, spec in enumerate(self.spec):
            if spec.requires_grad:
                assert spec.alias_group == 0, "TODO: support aliased backwards"
                assert spec.out == 0, "TODO: support autograd on out= ?"
                for d in range(self.ndim):
                    shape_types = {shape[d] for shape in self.shapes}
                    assert (
                        len(shape_types) == 1
                    ), "TODO: support backwards for broadcasting"
                self.result.set_backwards(i, self.make_backwards(i))

    def compute_broadcasts_and_size_checks(self):
        ndim = self.ndim
        spec = self.spec
        nargs = len(spec)
        longest = _argmax([x.ndim for x in spec])
        shapes = self.shapes
        shape_from = [(longest, d) for d in range(ndim)]
        for d in range(ndim):
            first = None
            for a in range(nargs):
                if shapes[a][d] == "one":
                    self.broadcasts.append((a, d))
                elif shapes[a][d] == "other":
                    if first is None:
                        shape_from[d] = first = (a, d - (ndim - spec[a].ndim))
                    else:
                        self.result.add_shape_check(
                            (first[0], first[1], a, d - (ndim - spec[a].ndim))
                        )

            if all(shapes[a][d] == "one" for a in range(nargs)):
                self.shape_vars[d] = _one()

        self.result.set_shape_from(shape_from)

    def compute_output_order(self):
        """
        Decide on an iteration order (permutation) for the dimensions of the output
        """
        ndim = self.ndim
        strides = self.strides
        output_order = []
        output_order_remaining = [[i] for i in range(ndim)]
        # packed dims first
        for d in reversed(range(ndim)):
            if strides[0][d] == "one":
                output_order.extend(output_order_remaining[d])
                output_order_remaining[d].clear()
        # swap the order for transposed
        for d in reversed(range(ndim)):
            if strides[0][d] == "transposed_contiguous":
                output_order_remaining[d - 1].extend(output_order_remaining[d])
                output_order_remaining[d].clear()
        # rest contiguous
        for d in reversed(range(ndim)):
            output_order.extend(output_order_remaining[d])
            output_order_remaining[d].clear()

        assert not self.output_order
        self.output_order = output_order
        assert sorted(output_order) == list(range(ndim))

    def compute_symbolic_shapes_and_strides(self):
        nargs = len(self.spec)
        ndim = self.ndim
        shapes = self.shapes
        strides = self.strides
        for a in range(nargs):
            # first fill in the terminal ones
            for d in range(ndim):
                self.replace_shape(a, d, "one", _one)
                self.replace_shape(a, d, "other", lambda: self.shape_args[d])
                self.replace_stride(a, d, "zero", _zero)
                self.replace_stride(a, d, "one", _one)
                if strides[a][d] == "as_arg":
                    strides[a][d] = self.add_stride_arg(a, d)

            # next the dependent ones
            while any(isinstance(x, str) for x in strides[a]):
                for d in reversed(range(ndim)):
                    self.replace_stride(
                        a, d, "contiguous", lambda: strides[a][d + 1] * shapes[a][d + 1]
                    )
                    if isinstance(strides[a][d], str):
                        break
                for d in range(ndim):
                    self.replace_stride(
                        a,
                        d,
                        "transposed_contiguous",
                        lambda: strides[a][d - 1] * shapes[a][d - 1],
                    )
                    if isinstance(strides[a][d], str):
                        break

        for a, d in self.broadcasts:
            strides[a][d] = _zero()

        self.result.set_stride_args_from(self.strides_from)

    def indexing(self, stride):
        result = _zero()
        for c, s in zip(self.iter_vars, stride):
            result = result + c * s
        return result

    def compute_code(self):
        bufs = [_te.BufHandle(s.dtype) for s in self.spec]
        if not self.spec[-1].out:
            options_from = [
                i for i in range(len(self.spec)) if self.spec[i].dtype == self.dtype
            ][0]
            self.result.add_allocated_output(options_from, self.output_order)
            bufs.append(_te.BufHandle(self.dtype))

            self.shapes.append(list(self.shape_vars))
            output_strides = [None] * self.ndim
            next_stride = _one()
            for i in self.output_order:
                output_strides[i] = next_stride
                next_stride *= self.shape_vars[i]
            assert all((x is not None) for x in output_strides)
            self.strides.append(output_strides)

        bufs_args = list(bufs)

        aliases = {}
        for i, s in enumerate(self.spec):
            assert s.alias_group >= 0, "TODO: support complex aliasing"
            if s.alias_group > 0 and s.alias_group not in aliases:
                aliases[s.alias_group] = i
            elif s.alias_group > 0 and FOLD_ALIASES:
                # BufHandle in buf_args is now ignored
                bufs[i] = bufs[aliases[s.alias_group]]

        input_bufs = bufs[:-1]
        input_strides = self.strides[:-1]
        output_bufs = bufs[-1:]
        output_strides = self.strides[-1:]

        inputs = [
            _te.Cast.make(self.dtype, buf.load(self.indexing(stride)))
            for buf, stride in zip(input_bufs, input_strides)
        ]
        val = _fx_to_expr(self.pointwise_fn, self.dtype)(*inputs)
        out = _te.Block(
            [
                buf.store(self.indexing(stride), val)
                for buf, stride in zip(output_bufs, output_strides)
            ]
        )

        loops: List[_te.For] = []
        for i in self.output_order:
            var = self.iter_vars[i]
            size = self.shape_vars[i]
            out = _te.For.make(var, _zero(), size, out)
            loops.insert(0, out)

        loopnest = _te.LoopNest(_te.Block([out]), output_bufs)

        if self.device == "cuda" and loops:
            flattened = loopnest.flatten(loops)
            assert flattened
            inner = _te.LoopNest.split_with_mask(flattened, 512)
            assert inner
            flattened.set_gpu_block_index(0)
            inner.set_gpu_thread_index(0)
        elif self.dtype == "llvm" and loops:
            pass  # TODO(jansel): need a parallel CPU schedule

        loopnest.prepare_for_codegen()
        cg = _te.construct_codegen(
            self.compile_mode,
            loopnest.simplify(),
            bufs_args + self.stride_args + self.shape_args,
        )
        self.result.set_code(cg)

    def run(self):
        self.error_checks()
        self.handle_autograd()
        self.compute_broadcasts_and_size_checks()
        self.compute_output_order()
        self.compute_symbolic_shapes_and_strides()
        self.compute_code()


class _CompileCache(CompileCache):
    pass


@functools.lru_cache(None)
def _source_to_pointwise_operator(
    fn_str: str, name: Optional[str] = None, module_name: Optional[str] = None
):
    """ Used when creating backwards() methods """
    return pointwise_operator(eval(fn_str), name=name, module_name=module_name)


def pointwise_operator(
    fn: Callable, name: Optional[str] = None, module_name: Optional[str] = None
):
    """
    Decorator to create a new pointwise operator.  The operator will be
    JIT compiled for different dtypes/devices/layouts/etc -- but supports dynamic shapes.

        @pointwise_operator
        def add(a, b):
            return a + b
    """
    name = name or fn.__name__
    module_name = module_name or fn.__module__
    args = [f"Tensor {name}" for name in inspect.signature(fn).parameters.keys()]
    signature = f"{name}({', '.join(args)}, *, Tensor? out=None)"

    def compile_fn(spec, result):
        return PointwiseCompiler(str(name), str(module_name), fn, spec, result)

    # This items are needed to support FX tracing
    rv = _CompileCache(name, module_name, [signature], compile_fn, _num_args(fn))
    rv.__name__ = name
    rv.__qualname__ = name
    rv.__module__ = module_name
    return rv
