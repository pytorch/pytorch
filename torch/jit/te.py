import torch
import itertools
from torch._C import _te
from pprint import pprint

_SHAPE_TYPES = {"one", "other"}
_STRIDE_TYPES = {"zero", "one", "contiguous", "transposed_contiguous", "as_arg"}


def _argmax(x):
    return int(torch.argmax(torch.LongTensor(x, device='cpu')))


def _zero():
    return _te.ExprHandle.int(0)


def _one():
    return _te.ExprHandle.int(1)


class PointwiseCompiler(object):
    def __init__(self, pointwise_fn, spec, result):
        self.pointwise_fn = pointwise_fn
        self.spec = spec
        self.result = result
        self.ndim = max(x.ndim for x in spec)
        self.shapes = [["one"] * (self.ndim - x.ndim) + x.shape for x in spec]
        self.strides = [["zero"] * (self.ndim - x.ndim) + x.stride for x in spec]
        self.shape_args = [_te.VarHandle(torch.int32) for _ in range(self.ndim)]
        self.shape_vars = list(self.shape_args)
        self.stride_args = []
        self.output_order = None
        self.run()

    def add_stride_arg(self):
        var = _te.VarHandle(torch.int32)
        self.stride_args.append(var)
        return var

    def replace_shape(self, a, d, expected, replacement):
        if self.shapes[a][d] == expected:
            self.shapes[a][d] = replacement()

    def replace_stride(self, a, d, expected, replacement):
        if self.strides[a][d] == expected:
            self.strides[a][d] = replacement()

    def error_checks(self):
        spec = self.spec
        devices = list(set(x.device for x in spec))
        dtypes = list(set(x.dtype for x in spec))
        layouts = list(set(x.layout for x in spec))
        assert len(devices) == 1
        assert len(dtypes) == 1
        assert len(layouts) == 1
        assert all((not x.requires_grad) for x in spec)
        assert [x.out for x in spec] == [False] * (len(spec) - 1) + [True]
        assert all(shape_type in _SHAPE_TYPES for shape_type in itertools.chain(*self.shapes))
        assert all(stride_type in _STRIDE_TYPES for stride_type in itertools.chain(*self.strides))

    def compute_broadcasts_and_size_checks(self):
        ndim = self.ndim
        spec = self.spec
        nargs = len(spec)
        longest = _argmax([x.ndim for x in spec])
        shapes = self.shapes
        strides = self.strides
        shape_from = [(longest, d) for d in range(ndim)]
        for d in range(ndim):
            first = None
            for a in range(nargs):
                if shapes[a][d] == "one":
                    strides[a][d] = "zero"  # broadcast
                if shapes[a][d] == "other":
                    if first is None:
                        shape_from[d] = first = (a, d - (ndim - spec[a].ndim))
                    else:
                        self.result.add_shape_check((
                            first[0], first[1],
                            a, d - (ndim - spec[a].ndim)))

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
                self.replace_stride(a, d, "as_arg", self.add_stride_arg)
            # next the dependent ones
            while any(isinstance(x, str) for x in strides[a]):
                for d in reversed(range(ndim)):
                    self.replace_stride(a, d, "contiguous", lambda: strides[a][d + 1] * shapes[a][d + 1])
                    if isinstance(strides[a][d], str):
                        break
                for d in range(ndim):
                    self.replace_stride(a, d, "transposed_contiguous", lambda: strides[a][d - 1] * shapes[a][d - 1])
                    if isinstance(strides[a][d], str):
                        break

    def indexing(self, coord, stride):
        result = _zero()
        for c, s in zip(coord, stride):
            result = result + c * s
        return result

    def run(self):
        # pprint(self.spec)
        self.error_checks()
        self.compute_broadcasts_and_size_checks()
        self.compute_output_order()
        self.compute_symbolic_shapes_and_strides()

        bufs = [_te.BufHandle(s.dtype) for s in self.spec[:-1]]
        dtype = self.spec[-1].dtype
        assert self.spec[-1].out

        def compute(*coord):
            inputs = [_te.Cast.make(dtype,
                                    buf.load(self.indexing(coord, stride)))
                      for buf, stride in zip(bufs, self.strides)]
            return self.pointwise_fn(*inputs)

        out = _te.Compute('out', self.shape_vars, compute)
        loopnest = _te.LoopNest([out])
        loopnest.prepare_for_codegen()
        self.result.set_code(_te.construct_codegen(
            'llvm',
            loopnest.simplify(),
            bufs + [out] + self.stride_args + self.shape_args))


def pointwise_operator(fn):
    """
    Decorator to create a new pointwise operator.  The operator will be
    JIT compiled for different dtypes/devices/layouts/etc -- but supports dynamic shapes.

        @pointwise_operator
        def add(a, b):
            return a + b
    """
    return _te.CompileCache(lambda spec, result: PointwiseCompiler(fn, spec, result))
