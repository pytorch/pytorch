import torch
import torch._C._te as te
import numpy as np
import timeit
import itertools
import pprint

scope = te.KernelScope()
pprint = pprint.pprint
SHAPE_TYPES = {"one", "other"}
STRIDE_TYPES = {"zero", "one", "contiguous", "transposed_contiguous", "as_arg"}


class Buffer(object):
    def __init__(self, spec):
        self.handle = te.BufHandle(spec)
        self.strides = []


class PointwiseCompiler(object):
    def __init__(self, pointwise_fn, spec, result):
        self.pointwise_fn = pointwise_fn
        self.spec = spec
        self.result = result
        self.ndim = max(x.ndim for x in spec)
        self.shapes = [["one"] * (self.ndim - x.ndim) + x.shape for x in spec]
        self.strides = [["zero"] * (self.ndim - x.ndim) + x.stride for x in spec]
        self.shape_args = [te.VarHandle(torch.int32) for _ in range(self.ndim)]
        self.shape_vars = list(self.shape_args)
        self.stride_args = []
        self.run()

    @staticmethod
    def zero():
        return te.ExprHandle.int(0)

    @staticmethod
    def one():
        return te.ExprHandle.int(1)

    def add_stride_arg(self):
        var = te.VarHandle(torch.int32)
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
        assert all(shape_type in SHAPE_TYPES for shape_type in itertools.chain(*self.shapes))
        assert all(stride_type in STRIDE_TYPES for stride_type in itertools.chain(*self.strides))

    def compute_broadcasts_and_size_checks(self):
        ndim = self.ndim
        spec = self.spec
        nargs = len(spec)
        longest = int(np.argmax([x.ndim for x in spec]))
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
                self.shape_vars[d] = self.one()

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
                self.replace_shape(a, d, "one", self.one)
                self.replace_shape(a, d, "other", lambda: self.shape_args[d])
                self.replace_stride(a, d, "zero", self.zero)
                self.replace_stride(a, d, "one", self.one)
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
        result = self.zero()
        for c, s in zip(coord, stride):
            result = result + c * s
        return result

    def run(self):
        pprint(self.spec)
        self.error_checks()
        self.compute_broadcasts_and_size_checks()
        self.compute_output_order()
        self.compute_symbolic_shapes_and_strides()

        bufs = [te.BufHandle(s.dtype) for s in self.spec[:-1]]
        dtype = self.spec[-1].dtype
        assert self.spec[-1].out

        def compute(*coord):
            inputs = [te.Cast.make(dtype,
                                   buf.load(self.indexing(coord, stride)))
                      for buf, stride in zip(bufs, self.strides)]
            return self.pointwise_fn(*inputs)

        out = te.Compute('out', self.shape_vars, compute)
        loopnest = te.LoopNest([out])
        loopnest.prepare_for_codegen()
        self.result.set_code(te.construct_codegen(
            'llvm',
            loopnest.simplify(),
            bufs + [out] + self.stride_args + self.shape_args))


def pointwise_operator(fn):
    return te.CompileCache(lambda spec, result: PointwiseCompiler(fn, spec, result))


@pointwise_operator
def nnc_add(a, b):
    return a + b


torch_add = torch.add


def main(n=16):
    tA = torch.randn(n, n)
    tB = torch.randn(n, n)
    result1 = torch.randn(n, n)
    result2 = torch.randn(n, n)

    def nnc_fn():
        nnc_add(tA, tB, out=result1)

    def aten_fn():
        torch_add(tA, tB, out=result2)

    if False:
        nnc_fn()
        aten_fn()
        torch.testing.assert_allclose(result1, result2)
        print("ok")
        return

    nnc = np.median(timeit.repeat(nnc_fn, number=10000, repeat=20))
    aten = np.median(timeit.repeat(aten_fn, number=10000, repeat=20))

    torch.testing.assert_allclose(result1, result2)
    print(f"n={n:4} nnc={nnc:.5f} aten={aten:.5f} aten/nnc={aten / nnc:.2f}x")


if __name__ == '__main__':
    main(1)
    main(64)
    # main(4096)
