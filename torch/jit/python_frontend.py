import torch
import sys
import ast
import inspect
from functools import partial
from collections import namedtuple
from torch._C._jit_tree_views import *

# TODO: improve error reporting (show source)
#       once this is done, convert all asserts into checks with nicer error messages

PY2 = sys.version_info[0] == 2
_reserved_prefix = '__jit'


def get_jit_ast(fn):
    source = dedent(inspect.getsource(fn))
    py_ast = ast.parse(source)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError("expected a single top-level function")
    return build_def(SourceRangeFactory(source), py_ast.body[0])


def dedent(source):
    lines = source.split('\n')
    indent_depth = 0
    while lines[0][indent_depth] == ' ':
        indent_depth += 1
    return '\n'.join(l[indent_depth:] for l in lines)


class Builder(object):
    def __call__(self, ctx, node):
        try:
            method = getattr(self, 'build_' + node.__class__.__name__)
        except AttributeError:
            raise RuntimeError(node.__class__.__name__ + " isn't a supported Python feature")
        return method(ctx, node)


class CountReturns(ast.NodeVisitor):
    def __init__(self):
        self.num_returns = 0

    def visit_Return(self, ret):
        self.num_returns += 1

    @staticmethod
    def get_count(py_def):
        counter = CountReturns()
        counter.visit(py_def)
        return counter.num_returns


_ret_err_msg = ("JIT-ed functions can only have a single return, "
                "and it has to be the last statement in the body")


def build_def(ctx, py_def):
    assert len(py_def.decorator_list) == 0
    returns = []
    ret_body = []
    body = py_def.body
    num_returns = CountReturns.get_count(py_def)
    if num_returns == 1:
        ret_stmt, body = body[-1], body[:-1]
        if not isinstance(ret_stmt, ast.Return):
            raise ValueError(_ret_err_msg)
        ret_expr = ret_stmt.value
        ret_vals = ret_expr.elts if isinstance(ret_expr, ast.Tuple) else [ret_expr]
        for i, val in enumerate(ret_vals):
            val_expr = build_expr(ctx, val)
            val_name = _reserved_prefix + '_' + str(i)
            r = val_expr.range()
            returns.append(Param(TensorType(r), Ident(r, val_name)))
            ret_body.append(Assign([Ident(r, val_name)], '=', val_expr))
    elif num_returns > 1:
        raise ValueError(_ret_err_msg)
    r = ctx.make_range(py_def.lineno, py_def.col_offset,
                       py_def.col_offset + len("def"))
    return Def(Ident(r, py_def.name),
               build_param_list(ctx, py_def.args),
               returns,
               build_stmt_list(ctx, body) + ret_body)


def build_param_list(ctx, py_args):
    assert py_args.vararg is None
    assert py_args.kwarg is None
    assert not py_args.defaults
    if PY2:
        # TODO: args are in py_args.args, but are expressions <sigh>
        raise RuntimeError("PY2 not supported")
    else:
        assert not py_args.kwonlyargs
        assert not py_args.kw_defaults
        return [build_param(ctx, arg) for arg in py_args.args]


def build_param(ctx, py_arg):
    assert py_arg.annotation is None  # TODO: handle annotations to get types
    name = py_arg.arg
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    return Param(TensorType(r), Ident(r, name))


def build_stmt_list(ctx, py_stmt_list):
    return [build_stmt(ctx, stmt) for stmt in py_stmt_list]


class StmtBuilder(Builder):
    augassign_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
    }

    @staticmethod
    def build_Expr(ctx, stmt):
        return ExprStmt(build_expr(ctx, stmt.value))

    @staticmethod
    def get_assign_ident(ctx, expr):
        var = build_expr(ctx, expr)
        if not isinstance(var, Var):
            raise RuntimeError("the only expressions allowed on the left hand side of "
                               "assignments are variable names")
        return var.name()

    @staticmethod
    def build_Assign(ctx, stmt):
        return Assign([StmtBuilder.get_assign_ident(ctx, e) for e in stmt.targets],
                      '=',
                      build_expr(ctx, stmt.value))

    @staticmethod
    def build_AugAssign(ctx, stmt):
        op = type(stmt.op)
        if op in StmtBuilder.augassign_map:
            op_token = StmtBuilder.augassign_map[op]
        else:
            raise RuntimeError("unsupported kind of augumented assignment: " + op.__name__)
        return Assign([StmtBuilder.get_assign_ident(ctx, stmt.target)],
                      op_token,
                      build_expr(ctx, stmt.value))

    @staticmethod
    def build_While(ctx, stmt):
        if stmt.orelse:
            raise RuntimeError("else branches of while loops aren't supported")
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("while"))
        return While(r, build_expr(ctx, stmt.test), [build_stmt(ctx, s) for s in stmt.body])

    @staticmethod
    def build_If(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("if"))
        return If(r, build_expr(ctx, stmt.test),
                  [build_stmt(ctx, s) for s in stmt.body],
                  [build_stmt(ctx, s) for s in stmt.orelse])


class ExprBuilder(Builder):
    _MethodRef = namedtuple('MethodRef', ['self', 'name'])
    binop_map = {
        ast.Add: partial(BinOp, '+'),
        ast.Sub: partial(BinOp, '-'),
        ast.Mult: partial(BinOp, '*'),
        ast.Div: partial(BinOp, '/'),
    }

    unop_map = {
        ast.Not: (UnaryOp, 'not'),
        ast.USub: (UnaryOp, '-'),
    }

    boolop_map = {
        ast.And: partial(BinOp, 'and'),
        ast.Or: partial(BinOp, 'or'),
    }

    cmpop_map = {
        ast.Eq: partial(BinOp, '=='),
        ast.NotEq: partial(BinOp, '!='),
        ast.LtE: partial(BinOp, '<='),
        ast.Lt: partial(BinOp, '<'),
        ast.GtE: partial(BinOp, '>='),
        ast.Gt: partial(BinOp, '>'),
    }

    @staticmethod
    def build_Attribute(ctx, expr):
        # NB: the only attributes we support are for getting methods
        value = build_expr(ctx, expr.value)
        return ExprBuilder._MethodRef(value, Ident(value.range(), expr.attr))

    @staticmethod
    def build_Call(ctx, expr):
        ref = build_expr(ctx, expr.func, allow_methods=True)
        assert type(ref) is ExprBuilder._MethodRef
        args = [build_expr(ctx, py_arg) for py_arg in expr.args]
        kwargs = [Attribute(Ident(name), build_expr(ctx, value)) for name, value in expr.keywords]
        return Apply(ref.name, [ref.self] + args, kwargs)

    @staticmethod
    def build_Name(ctx, expr):
        assert not expr.id.startswith(_reserved_prefix)
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(expr.id))
        return Var(Ident(r, expr.id))

    @staticmethod
    def build_BinOp(ctx, expr):
        op = type(expr.op)
        if op in ExprBuilder.binop_map:
            return ExprBuilder.binop_map[op](build_expr(ctx, expr.left), build_expr(ctx, expr.right))
        else:
            raise RuntimeError("unsupported binary operator: " + op.__name__)

    @staticmethod
    def build_UnaryOp(ctx, expr):
        op = type(expr.op)
        if op in ExprBuilder.unop_map:
            constructor, token = ExprBuilder.unop_map[op]
            r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(token))
            return constructor(r, token, build_expr(ctx, expr.operand))
        else:
            raise RuntimeError("unsupported unary operator: " + op.__name__)

    @staticmethod
    def build_BoolOp(ctx, expr):
        op = type(expr.op)
        if op in ExprBuilder.boolop_map:
            constr = ExprBuilder.boolop_map[op]
        else:
            raise RuntimeError("unsupported boolean operator: " + op.__name__)
        if len(expr.values) < 2:
            raise AssertionError("expected at least 2 values in BoolOp, but got " + str(len(expr.values)))
        lhs = build_expr(ctx, expr.values[0])
        for rhs in expr.values[1:]:
            lhs = constr(lhs, build_expr(ctx, rhs))
        return lhs

    @staticmethod
    def build_IfExp(ctx, expr):
        return TernaryIf(build_expr(ctx, expr.test),
                         build_expr(ctx, expr.body),
                         build_expr(ctx, expr.orelse))

    @staticmethod
    def build_Compare(ctx, expr):
        operands = [build_expr(ctx, e) for e in [expr.left] + list(expr.comparators)]
        # NB: Python allows doing x < y.add_(2) < z, and we need to ensure that y.add_(2)
        # will execute only once. Not sure how to do this without an assignment to a temporary,
        # but this will require some redesigns since this function will not only need to return
        # an expression, but also a few assignment statements that should happen beforehand.
        # Since it's not all that common, we're restricting the usage for now.
        if len(operands) > 2 and not all(isinstance(operand, Var) for operand in operands):
            raise RuntimeError("the only expressions allowed in comparisons of more than 2 elements are variable names")
        result = None
        for lhs, op_, rhs in zip(operands, expr.ops, operands[1:]):
            op = type(op_)
            if op in ExprBuilder.cmpop_map:
                cmp_expr = ExprBuilder.cmpop_map[op](lhs, rhs)
            else:
                raise RuntimeError("unsupported comparison operator: " + op.__name__)
            if result is None:
                result = cmp_expr
            else:
                result = BinOp('and', result, cmp_expr)
        return result

    @staticmethod
    def build_Num(ctx, expr):
        # TODO: fix this once we have a nice Number node in our AST
        raise RuntimeError("scalar constants aren't supported")

    def __call__(self, ctx, expr, allow_methods=False):
        result = super(ExprBuilder, self).__call__(ctx, expr)
        if type(result) is ExprBuilder._MethodRef and not allow_methods:
            raise TypeError("taking attributes/function values isn't supported")
        return result


build_expr = ExprBuilder()
build_stmt = StmtBuilder()
