
from sympy.printing.pycode import AbstractPythonCodePrinter, ArrayPrinter
from sympy.matrices.expressions import MatrixExpr
from sympy.core.mul import Mul
from sympy.printing.precedence import PRECEDENCE
from sympy.external import import_module
from sympy.codegen.cfunctions import Sqrt
from sympy import S
from sympy import Integer

import sympy

torch = import_module('torch')


class TorchPrinter(ArrayPrinter, AbstractPythonCodePrinter):

    printmethod = "_torchcode"

    mapping = {
        sympy.Abs: "torch.abs",
        sympy.sign: "torch.sign",

        # XXX May raise error for ints.
        sympy.ceiling: "torch.ceil",
        sympy.floor: "torch.floor",
        sympy.log: "torch.log",
        sympy.exp: "torch.exp",
        Sqrt: "torch.sqrt",
        sympy.cos: "torch.cos",
        sympy.acos: "torch.acos",
        sympy.sin: "torch.sin",
        sympy.asin: "torch.asin",
        sympy.tan: "torch.tan",
        sympy.atan: "torch.atan",
        sympy.atan2: "torch.atan2",
        # XXX Also may give NaN for complex results.
        sympy.cosh: "torch.cosh",
        sympy.acosh: "torch.acosh",
        sympy.sinh: "torch.sinh",
        sympy.asinh: "torch.asinh",
        sympy.tanh: "torch.tanh",
        sympy.atanh: "torch.atanh",
        sympy.Pow: "torch.pow",

        sympy.re: "torch.real",
        sympy.im: "torch.imag",
        sympy.arg: "torch.angle",

        # XXX May raise error for ints and complexes
        sympy.erf: "torch.erf",
        sympy.loggamma: "torch.lgamma",

        sympy.Eq: "torch.eq",
        sympy.Ne: "torch.ne",
        sympy.StrictGreaterThan: "torch.gt",
        sympy.StrictLessThan: "torch.lt",
        sympy.LessThan: "torch.le",
        sympy.GreaterThan: "torch.ge",

        sympy.And: "torch.logical_and",
        sympy.Or: "torch.logical_or",
        sympy.Not: "torch.logical_not",
        sympy.Max: "torch.max",
        sympy.Min: "torch.min",

        # Matrices
        sympy.MatAdd: "torch.add",
        sympy.HadamardProduct: "torch.mul",
        sympy.Trace: "torch.trace",

        # XXX May raise error for integer matrices.
        sympy.Determinant: "torch.det",
    }

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        torch_version=None,
        requires_grad=False,
        dtype="torch.float64",
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        version = self._settings['torch_version']
        self.requires_grad = self._settings['requires_grad']
        self.dtype = self._settings['dtype']
        if version is None and torch:
            version = torch.__version__
        self.torch_version = version

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

    # mirrors the tensorflow version
    _print_Expr = _print_Function
    _print_Application = _print_Function
    _print_MatrixExpr = _print_Function
    _print_Relational = _print_Function
    _print_Not = _print_Function
    _print_And = _print_Function
    _print_Or = _print_Function
    _print_HadamardProduct = _print_Function
    _print_Trace = _print_Function
    _print_Determinant = _print_Function

    def _print_Inverse(self, expr):
        return '{}({})'.format(self._module_format("torch.linalg.inv"),
                               self._print(expr.args[0]))

    def _print_Transpose(self, expr):
        if expr.arg.is_Matrix and expr.arg.shape[0] == expr.arg.shape[1]:
            # For square matrices, we can use the .t() method
            return "{}({}).t()".format("torch.transpose", self._print(expr.arg))
        else:
            # For non-square matrices or more general cases
            # transpose first and second dimensions (typical matrix transpose)
            return "{}.permute({})".format(
                self._print(expr.arg),
                ", ".join([str(i) for i in range(len(expr.arg.shape))])[::-1]
            )

    def _print_PermuteDims(self, expr):
        return "%s.permute(%s)" % (
            self._print(expr.expr),
            ", ".join(str(i) for i in expr.permutation.array_form)
        )

    def _print_Derivative(self, expr):
        # this version handles multi-variable and mixed partial derivatives. The tensorflow version does not.
        variables = expr.variables
        expr_arg = expr.expr

        # Handle multi-variable or repeated derivatives
        if len(variables) > 1 or (
            len(variables) == 1 and not isinstance(variables[0], tuple) and variables.count(variables[0]) > 1):
            result = self._print(expr_arg)
            var_groups = {}

            # Group variables by base symbol
            for var in variables:
                if isinstance(var, tuple):
                    base_var, order = var
                    var_groups[base_var] = var_groups.get(base_var, 0) + order
                else:
                    var_groups[var] = var_groups.get(var, 0) + 1

            # Apply gradients in sequence
            for var, order in var_groups.items():
                for _ in range(order):
                    result = "torch.autograd.grad({}, {}, create_graph=True)[0]".format(result, self._print(var))
            return result

        # Handle single variable case
        if len(variables) == 1:
            variable = variables[0]
            if isinstance(variable, tuple) and len(variable) == 2:
                base_var, order = variable
                if not isinstance(order, Integer): raise NotImplementedError("Only integer orders are supported")
                result = self._print(expr_arg)
                for _ in range(order):
                    result = "torch.autograd.grad({}, {}, create_graph=True)[0]".format(result, self._print(base_var))
                return result
            return "torch.autograd.grad({}, {})[0]".format(self._print(expr_arg), self._print(variable))

        return self._print(expr_arg)  # Empty variables case

    def _print_Piecewise(self, expr):
        from sympy import Piecewise
        e, cond = expr.args[0].args
        if len(expr.args) == 1:
            return '{}({}, {}, {})'.format(
                self._module_format("torch.where"),
                self._print(cond),
                self._print(e),
                0)

        return '{}({}, {}, {})'.format(
            self._module_format("torch.where"),
            self._print(cond),
            self._print(e),
            self._print(Piecewise(*expr.args[1:])))

    def _print_Pow(self, expr):
        # XXX May raise error for
        # int**float or int**complex or float**complex
        base, exp = expr.args
        if expr.exp == S.Half:
            return "{}({})".format(
                self._module_format("torch.sqrt"), self._print(base))
        return "{}({}, {})".format(
            self._module_format("torch.pow"),
            self._print(base), self._print(exp))

    def _print_MatMul(self, expr):
        # Separate matrix and scalar arguments
        mat_args = [arg for arg in expr.args if isinstance(arg, MatrixExpr)]
        args = [arg for arg in expr.args if arg not in mat_args]
        # Handle scalar multipliers if present
        if args:
            return "%s*%s" % (
                self.parenthesize(Mul.fromiter(args), PRECEDENCE["Mul"]),
                self._expand_fold_binary_op("torch.matmul", mat_args)
            )
        else:
            return self._expand_fold_binary_op("torch.matmul", mat_args)

    def _print_MatPow(self, expr):
        return self._expand_fold_binary_op("torch.mm", [expr.base]*expr.exp)

    def _print_MatrixBase(self, expr):
        data = "[" + ", ".join(["[" + ", ".join([self._print(j) for j in i]) + "]" for i in expr.tolist()]) + "]"
        params = [str(data)]
        params.append(f"dtype={self.dtype}")
        if self.requires_grad:
            params.append("requires_grad=True")

        return "{}({})".format(
            self._module_format("torch.tensor"),
            ", ".join(params)
        )

    def _print_isnan(self, expr):
        return f'torch.isnan({self._print(expr.args[0])})'

    def _print_isinf(self, expr):
        return f'torch.isinf({self._print(expr.args[0])})'

    def _print_Identity(self, expr):
        if all(dim.is_Integer for dim in expr.shape):
            return "{}({})".format(
                self._module_format("torch.eye"),
                self._print(expr.shape[0])
            )
        else:
            # For symbolic dimensions, fall back to a more general approach
            return "{}({}, {})".format(
                self._module_format("torch.eye"),
                self._print(expr.shape[0]),
                self._print(expr.shape[1])
            )

    def _print_ZeroMatrix(self, expr):
        return "{}({})".format(
            self._module_format("torch.zeros"),
            self._print(expr.shape)
        )

    def _print_OneMatrix(self, expr):
        return "{}({})".format(
            self._module_format("torch.ones"),
            self._print(expr.shape)
        )

    def _print_conjugate(self, expr):
        return f"{self._module_format('torch.conj')}({self._print(expr.args[0])})"

    def _print_ImaginaryUnit(self, expr):
        return "1j"  # uses the Python built-in 1j notation for the imaginary unit

    def _print_Heaviside(self, expr):
        args = [self._print(expr.args[0]), "0.5"]
        if len(expr.args) > 1:
            args[1] = self._print(expr.args[1])
        return f"{self._module_format('torch.heaviside')}({args[0]}, {args[1]})"

    def _print_gamma(self, expr):
        return f"{self._module_format('torch.special.gamma')}({self._print(expr.args[0])})"

    def _print_polygamma(self, expr):
        if expr.args[0] == S.Zero:
            return f"{self._module_format('torch.special.digamma')}({self._print(expr.args[1])})"
        else:
            raise NotImplementedError("PyTorch only supports digamma (0th order polygamma)")

    _module = "torch"
    _einsum = "einsum"
    _add = "add"
    _transpose = "t"
    _ones = "ones"
    _zeros = "zeros"

def torch_code(expr, requires_grad=False, dtype="torch.float64", **settings):
    printer = TorchPrinter(settings={'requires_grad': requires_grad, 'dtype': dtype})
    return printer.doprint(expr, **settings)
