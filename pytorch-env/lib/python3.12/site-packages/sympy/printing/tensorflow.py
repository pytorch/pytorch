from sympy.external.importtools import version_tuple
from collections.abc import Iterable

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.codegen.cfunctions import Sqrt
from sympy.external import import_module
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pycode import AbstractPythonCodePrinter, ArrayPrinter
import sympy

tensorflow = import_module('tensorflow')

class TensorflowPrinter(ArrayPrinter, AbstractPythonCodePrinter):
    """
    Tensorflow printer which handles vectorized piecewise functions,
    logical operators, max/min, and relational operators.
    """
    printmethod = "_tensorflowcode"

    mapping = {
        sympy.Abs: "tensorflow.math.abs",
        sympy.sign: "tensorflow.math.sign",

        # XXX May raise error for ints.
        sympy.ceiling: "tensorflow.math.ceil",
        sympy.floor: "tensorflow.math.floor",
        sympy.log: "tensorflow.math.log",
        sympy.exp: "tensorflow.math.exp",
        Sqrt: "tensorflow.math.sqrt",
        sympy.cos: "tensorflow.math.cos",
        sympy.acos: "tensorflow.math.acos",
        sympy.sin: "tensorflow.math.sin",
        sympy.asin: "tensorflow.math.asin",
        sympy.tan: "tensorflow.math.tan",
        sympy.atan: "tensorflow.math.atan",
        sympy.atan2: "tensorflow.math.atan2",
        # XXX Also may give NaN for complex results.
        sympy.cosh: "tensorflow.math.cosh",
        sympy.acosh: "tensorflow.math.acosh",
        sympy.sinh: "tensorflow.math.sinh",
        sympy.asinh: "tensorflow.math.asinh",
        sympy.tanh: "tensorflow.math.tanh",
        sympy.atanh: "tensorflow.math.atanh",

        sympy.re: "tensorflow.math.real",
        sympy.im: "tensorflow.math.imag",
        sympy.arg: "tensorflow.math.angle",

        # XXX May raise error for ints and complexes
        sympy.erf: "tensorflow.math.erf",
        sympy.loggamma: "tensorflow.math.lgamma",

        sympy.Eq: "tensorflow.math.equal",
        sympy.Ne: "tensorflow.math.not_equal",
        sympy.StrictGreaterThan: "tensorflow.math.greater",
        sympy.StrictLessThan: "tensorflow.math.less",
        sympy.LessThan: "tensorflow.math.less_equal",
        sympy.GreaterThan: "tensorflow.math.greater_equal",

        sympy.And: "tensorflow.math.logical_and",
        sympy.Or: "tensorflow.math.logical_or",
        sympy.Not: "tensorflow.math.logical_not",
        sympy.Max: "tensorflow.math.maximum",
        sympy.Min: "tensorflow.math.minimum",

        # Matrices
        sympy.MatAdd: "tensorflow.math.add",
        sympy.HadamardProduct: "tensorflow.math.multiply",
        sympy.Trace: "tensorflow.linalg.trace",

        # XXX May raise error for integer matrices.
        sympy.Determinant : "tensorflow.linalg.det",
    }

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        tensorflow_version=None
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        version = self._settings['tensorflow_version']
        if version is None and tensorflow:
            version = tensorflow.__version__
        self.tensorflow_version = version

    def _print_Function(self, expr):
        op = self.mapping.get(type(expr), None)
        if op is None:
            return super()._print_Basic(expr)
        children = [self._print(arg) for arg in expr.args]
        if len(children) == 1:
            return "%s(%s)" % (
                self._module_format(op),
                children[0]
            )
        else:
            return self._expand_fold_binary_op(op, children)

    _print_Expr = _print_Function
    _print_Application = _print_Function
    _print_MatrixExpr = _print_Function
    # TODO: a better class structure would avoid this mess:
    _print_Relational = _print_Function
    _print_Not = _print_Function
    _print_And = _print_Function
    _print_Or = _print_Function
    _print_HadamardProduct = _print_Function
    _print_Trace = _print_Function
    _print_Determinant = _print_Function

    def _print_Inverse(self, expr):
        op = self._module_format('tensorflow.linalg.inv')
        return "{}({})".format(op, self._print(expr.arg))

    def _print_Transpose(self, expr):
        version = self.tensorflow_version
        if version and version_tuple(version) < version_tuple('1.14'):
            op = self._module_format('tensorflow.matrix_transpose')
        else:
            op = self._module_format('tensorflow.linalg.matrix_transpose')
        return "{}({})".format(op, self._print(expr.arg))

    def _print_Derivative(self, expr):
        variables = expr.variables
        if any(isinstance(i, Iterable) for i in variables):
            raise NotImplementedError("derivation by multiple variables is not supported")
        def unfold(expr, args):
            if not args:
                return self._print(expr)
            return "%s(%s, %s)[0]" % (
                    self._module_format("tensorflow.gradients"),
                    unfold(expr, args[:-1]),
                    self._print(args[-1]),
                )
        return unfold(expr.expr, variables)

    def _print_Piecewise(self, expr):
        version = self.tensorflow_version
        if version and version_tuple(version) < version_tuple('1.0'):
            tensorflow_piecewise = "tensorflow.select"
        else:
            tensorflow_piecewise = "tensorflow.where"

        from sympy.functions.elementary.piecewise import Piecewise
        e, cond = expr.args[0].args
        if len(expr.args) == 1:
            return '{}({}, {}, {})'.format(
                self._module_format(tensorflow_piecewise),
                self._print(cond),
                self._print(e),
                0)

        return '{}({}, {}, {})'.format(
            self._module_format(tensorflow_piecewise),
            self._print(cond),
            self._print(e),
            self._print(Piecewise(*expr.args[1:])))

    def _print_Pow(self, expr):
        # XXX May raise error for
        # int**float or int**complex or float**complex
        base, exp = expr.args
        if expr.exp == S.Half:
            return "{}({})".format(
                self._module_format("tensorflow.math.sqrt"), self._print(base))
        return "{}({}, {})".format(
            self._module_format("tensorflow.math.pow"),
            self._print(base), self._print(exp))

    def _print_MatrixBase(self, expr):
        tensorflow_f = "tensorflow.Variable" if expr.free_symbols else "tensorflow.constant"
        data = "["+", ".join(["["+", ".join([self._print(j) for j in i])+"]" for i in expr.tolist()])+"]"
        return "%s(%s)" % (
            self._module_format(tensorflow_f),
            data,
        )

    def _print_MatMul(self, expr):
        from sympy.matrices.expressions import MatrixExpr
        mat_args = [arg for arg in expr.args if isinstance(arg, MatrixExpr)]
        args = [arg for arg in expr.args if arg not in mat_args]
        if args:
            return "%s*%s" % (
                self.parenthesize(Mul.fromiter(args), PRECEDENCE["Mul"]),
                self._expand_fold_binary_op(
                    "tensorflow.linalg.matmul", mat_args)
            )
        else:
            return self._expand_fold_binary_op(
                "tensorflow.linalg.matmul", mat_args)

    def _print_MatPow(self, expr):
        return self._expand_fold_binary_op(
            "tensorflow.linalg.matmul", [expr.base]*expr.exp)

    def _print_CodeBlock(self, expr):
        # TODO: is this necessary?
        ret = []
        for subexpr in expr.args:
            ret.append(self._print(subexpr))
        return "\n".join(ret)

    _module = "tensorflow"
    _einsum = "linalg.einsum"
    _add = "math.add"
    _transpose = "transpose"
    _ones = "ones"
    _zeros = "zeros"


def tensorflow_code(expr, **settings):
    printer = TensorflowPrinter(settings)
    return printer.doprint(expr)
