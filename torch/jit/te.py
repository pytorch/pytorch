import torch
import itertools
import functools
from torch._C import _te

FOLD_ALIASES = True
_SHAPE_TYPES = {"one", "other"}
_STRIDE_TYPES = {"zero", "one", "contiguous", "transposed_contiguous", "as_arg"}
_int = _te.ExprHandle.int


def _argmax(x):
    return int(torch.argmax(torch.LongTensor(x, device='cpu')))


def _zero():
    return _int(0)


def _one():
    return _int(1)


def _combine_dtype(a, b):
    if a == b:
        return a
    # TODO(jansel): find a cleaner way to implement this
    return (torch.zeros(1, dtype=a, device="cpu") +
            torch.zeros(1, dtype=b, device="cpu")).dtype


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
        self.iter_vars = [_te.VarHandle(torch.int32) for _ in range(self.ndim)]
        self.stride_args = []
        self.broadcasts = []
        self.output_order = None

        device, = list(set(x.device.type for x in spec))
        self.compile_mode = {"cpu": "llvm", "cuda": "cuda"}[device]

        if spec[-1].out:
            self.dtype = spec[-1].dtype
        else:
            self.dtype = functools.reduce(
                _combine_dtype, [x.dtype for x in spec])

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
        layout, = list(set(x.layout for x in spec))
        assert layout == torch.strided, "TODO: support other layouts"
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

        for a, d in self.broadcasts:
            strides[a][d] = _zero()

    def indexing(self, stride):
        result = _zero()
        for c, s in zip(self.iter_vars, stride):
            result = result + c * s
        return result

    def compute_code(self):
        bufs = [_te.BufHandle(s.dtype) for s in self.spec]
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

        assert self.spec[-1].out

        inputs = [_te.Cast.make(self.dtype,
                                buf.load(self.indexing(stride)))
                  for buf, stride in zip(input_bufs, input_strides)]
        val = self.pointwise_fn(*inputs)
        out = _te.Block([buf.store(self.indexing(stride), val)
                         for buf, stride in zip(output_bufs, output_strides)])

        for var, size in reversed(list(zip(self.iter_vars, self.shape_vars))):
            out = _te.For.make(var, _zero(), size, out)

        loopnest = _te.LoopNest(out, output_bufs)
        loopnest.prepare_for_codegen()

        # TODO(jansel): use some sort of schedule here?

        cg = _te.construct_codegen(
            self.compile_mode,
            loopnest.simplify(),
            bufs_args + self.stride_args + self.shape_args)
        self.result.set_code(cg)

    def run(self):
        # pprint(self.spec)
        self.error_checks()
        self.compute_broadcasts_and_size_checks()
        self.compute_output_order()
        self.compute_symbolic_shapes_and_strides()
        self.compute_code()


def pointwise_operator(fn):
    """
    Decorator to create a new pointwise operator.  The operator will be
    JIT compiled for different dtypes/devices/layouts/etc -- but supports dynamic shapes.

        @pointwise_operator
        def add(a, b):
            return a + b
    """
    return _te.CompileCache(lambda spec, result: PointwiseCompiler(fn, spec, result))
