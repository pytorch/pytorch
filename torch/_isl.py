import torch

try:
    import tvm
except ImportError:
    raise ImportError("torch.isl requires the tvm module")


class TvmBuilder(object):
    """
    This class is a thin wrapper around the tvm API which keeps
    tracks of the inputs, outputs, ops and variables you declare.
    This helps us satisfy the API requirements of c2isl, which
    require you to explicitly declare all of these things.

    TODO: For some reason, TVM's own APIs don't seem to need to
    track all this information, so it's worth checking if there's
    a way to avoid having to faff around with all of this, and just
    use tvm directly.
    """

    def __init__(self):
        self.vars = []
        self.inputs = []
        self.ops = []
        self.outputs = []

    # Core operations
    def var(self, *args, **kwargs):
        """
        Create a new TVM variable with some name.

        >>> M = builder.var('M')

        NB: Due to limitations in c2isl, the name is mandatory and must be fresh.
        See https://github.com/nicolasvasilache/c2isl/issues/92
        """
        v = tvm.var(*args, **kwargs)
        self.vars.append(v)
        return v

    def placeholder(self, *args, **kwargs):
        """
        Create a new placeholder tensor representing an input to the operator.
        This takes a tuple of variables specifying the shape of the tensor,
        a keyword argument specifying the type of the tensor (in TVM type
        form), and the name of the tensor.

        >>> A = builder.placeholder((M, N), dtype='float64', name='A')

        NB: Due to limitations in c2isl, the name is mandatory and must be fresh.
        See https://github.com/nicolasvasilache/c2isl/issues/92
        """
        p = tvm.placeholder(*args, **kwargs)
        self.inputs.append(p)
        return p

    def compute(self, *args, **kwargs):
        """
        Perform a tensor comprehension producing some output.  This takes the
        output tensor shape (a tuple) and a lambda, taking an index argument per
        dimension in the output tensor and returning the TVM expression which
        computes the value of the output tensor at that index.

        An example will illuminate:

        >>> C = builder.compute((M, N), lambda i, j: A[i, j] + B[i, j], name='C')

        This computes an output tensor C with shape M x N, where C[i, j]
        is computed by adding A[i, j] and B[i, j] (i.e., pointwise sum.)
        TVM also supports more advanced uses such as reductions;
        for more examples, you should grep the TVM codebase for 'compute'.

        NB: Due to limitations in c2isl, the name is mandatory and must be fresh.
        See https://github.com/nicolasvasilache/c2isl/issues/92
        """
        o = tvm.compute(*args, **kwargs)
        self.ops.append(o)
        return o

    def output(self, out):
        """
        Register a tensor (usually produced by 'compute') as the output of this
        operator.

        >>> builder.output(C)
        """
        self.outputs.append(out)

    # These functions are a straight port from tvm.cc in c2isl, and not very
    # Pythonic.
    #
    # TODO: Delete these from the builder once it's easier to stitch operators
    # together.
    def makeMatMult(self, I, W, outputName, transI=False, transW=False):
        outputShape = (I.shape[1 if transI else 0], W.shape[0 if transW else 1])
        reductionExpr = I.shape[0 if transI else 1]
        return self.tvmOpMakeMatMult(outputShape, [reductionExpr], I, W, outputName, transI, transW)

    def access(self, input, i, j, transpose=False):
        if transpose:
            return input[j, i]
        else:
            return input[i, j]

    def tvmOpMakeMatMult(self, outputShape, reductionExpr, input, weight, name, transposeInput, transposeWeight):
        assert len(reductionExpr) == 1
        r = reductionExpr[0]
        k = tvm.reduce_axis((0, r), "k")

        def l(i, j):
            return tvm.sum(self.access(input, i, k, transposeInput) *
                           self.access(weight, k, j, transposeWeight),
                           axis=k)
        # Register the op!
        return self.compute(outputShape, l, name=name)


class IslFunction(object):
    """
    Superclass for operators which are defined directly using TVM
    IR and then compiled into runnable CUDA code using c2isl.
    See IslMatMul for an example of how to instantiate this class.

    Limitations:
      * CUDA only
      * As TVM IR requires you to specify the types of placeholder
        variables upfront, this means that operators created
        with this are necessarily monomorphic.
      * If you reuse this operator, you must call it again with
        tensors that are exactly the same size (this is because
        c2isl generates size-specialized kernels).
    """

    # This is an irritating little function.  I used the pre-existing
    # plumbing for exposing C++ Functions as Python Functions.
    # But these are not "true" classes: they're just functions that
    # return a Python wrapper to the underlying C++ object. This
    # means we can't subclass them object.  So instead we overload
    # __new__ to do all of the necessary setup, and then dispatch over.
    # Ouch!  Perhaps worth a rewrite if we decide to do the C++ end
    # properly.
    def __new__(cls, *args, **kwargs):
        builder = TvmBuilder()
        cls.tvm(builder, *args, **kwargs)

        def swizzle(ls):
            return tuple(l.handle.value for l in ls)
        return torch._C._functions.IslFunction(
            cls.name(),
            *map(swizzle, (builder.outputs, builder.inputs, builder.vars, builder.ops)))

    @staticmethod
    def tvm(builder):
        raise NotImplementedError

    @staticmethod
    def name():
        raise NotImplementedError


# This is an example use-case of this functionality.
class IslMatMul(IslFunction):
    @staticmethod
    def name():
        return "matmul"

    @staticmethod
    def tvm(builder, dtype='float64'):
        M = builder.var('M')
        N = builder.var('N')
        O = builder.var('O')

        # dtype monomorphic!
        A = builder.placeholder((M, N), dtype=dtype, name='A')
        B = builder.placeholder((N, O), dtype=dtype, name='B')
        C = builder.makeMatMult(A, B, "C")

        builder.output(C)
