import torch
import sys
import ast
import inspect
import string
from textwrap import dedent
from functools import partial
from collections import namedtuple
from torch._C._jit_tree_views import *

PY2 = sys.version_info[0] == 2
_reserved_prefix = '__jit'
_reserved_names = {'print'}
_identifier_chars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)


def is_reserved_name(name):
    return name.startswith(_reserved_prefix) or name in _reserved_names


pretty_node_names = {
    ast.FunctionDef: "function definitions",
    ast.For: "for loops",
    ast.Delete: "del statements",
    ast.ClassDef: "class definitions",
    ast.With: "with statements",
    ast.Raise: "raise statements",
    ast.Assert: "assertions",
    ast.Import: "import statements",
    ast.ImportFrom: "import statements",
    ast.Global: "global variables",
    ast.Break: "break statements",
    ast.Continue: "continue statements",
}

node_start_tokens = {
    ast.FunctionDef: "def",
    ast.For: "for",
    ast.Delete: "del",
    ast.ClassDef: "class",
    ast.With: "with",
    ast.Raise: "raise",
    ast.Assert: "assert",
    ast.Import: "import",
    ast.ImportFrom: "from",
    ast.Global: "global",
    ast.Break: "break",
    ast.Continue: "continue",
}

if PY2:
    pretty_node_names.update({
        ast.Print: "print statements",
        ast.TryExcept: "try blocks",
        ast.TryFinally: "try blocks",
        ast.Exec: "exec statements",
    })

    node_start_tokens.update({
        ast.Print: "print",
        ast.TryExcept: "try",
        ast.TryFinally: "try",
        ast.Exec: "exec",
    })
else:
    pretty_node_names.update({
        ast.AsyncFunctionDef: "async function definitions",
        ast.AsyncFor: "async for loops",
        ast.AsyncWith: "async with statements",
        ast.Try: "try blocks",
        ast.Nonlocal: "nonlocal variables",
    })

    node_start_tokens.update({
        ast.AsyncFunctionDef: "async def",
        ast.AsyncFor: "async for",
        ast.AsyncWith: "async with",
        ast.Try: "try",
        ast.Nonlocal: "nonlocal",
    })

if sys.version_info >= (3, 6):
    pretty_node_names.update({
        ast.AnnAssign: "annotated assignments",
    })
    # NB: no specific token for AnnAssign


class FrontendError(Exception):
    def __init__(self, source_range, msg):
        self.source_range = source_range
        self.msg = msg

    def __str__(self):
        result = self.msg
        if self.source_range is not None:
            result += '\n' + self.source_range.highlight()
        return result


class NotSupportedError(FrontendError):
    pass


class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node):
        # If we don't have a specific token, we default to length of 1
        node_type = type(offending_node)
        range_len = len(node_start_tokens.get(node_type, ' '))
        source_range = ctx.make_range(offending_node.lineno,
                                      offending_node.col_offset,
                                      offending_node.col_offset + range_len)
        feature_name = pretty_node_names.get(node_type, node_type.__name__)
        msg = "{} aren't supported".format(feature_name)
        super(NotSupportedError, self).__init__(source_range, msg)


class FrontendTypeError(FrontendError):
    pass


def get_jit_ast(fn):
    source = dedent(inspect.getsource(fn))
    py_ast = ast.parse(source)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError("expected a single top-level function")
    return build_def(SourceRangeFactory(source), py_ast.body[0])


class Builder(object):
    def __call__(self, ctx, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise UnsupportedNodeError(ctx, node)
        return method(ctx, node)


def build_def(ctx, py_def):
    returns = []
    ret_body = []
    body = py_def.body
    r = ctx.make_range(py_def.lineno, py_def.col_offset,
                       py_def.col_offset + len("def"))
    return Def(Ident(r, py_def.name),
               build_param_list(ctx, py_def.args),
               [build_stmt(ctx, stmt) for stmt in body])


_vararg_kwarg_err = ("Compiled functions can't take variable number of arguments, "
                     "have default values for arguments, nor keyword-only arguments")


def build_param_list(ctx, py_args):
    if py_args.vararg is not None or py_args.kwarg is not None or py_args.defaults:
        raise ValueError(_vararg_kwarg_err)
    if not PY2 and (py_args.kw_defaults or py_args.kwonlyargs):
        raise ValueError(_vararg_kwarg_err)
    return [build_param(ctx, arg) for arg in py_args.args]


def build_param(ctx, py_arg):
    # NB: In Python3 py_arg is a pair of (str arg, expr? annotation)
    #     In Python2 py_arg is a Name (Expr subclass)
    if getattr(py_arg, 'annotation', None) is not None:
        raise ValueError("Compiled functions don't support annotations")
    name = py_arg.id if PY2 else py_arg.arg
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    return Param(TensorType(r), Ident(r, name))


class StmtBuilder(Builder):
    augassign_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
    }

    @staticmethod
    def build_Expr(ctx, stmt):
        return ExprStmt([build_expr(ctx, stmt.value)])

    @staticmethod
    def get_assign_lhs_expr(ctx, expr):
        var = build_expr(ctx, expr)
        if not isinstance(var, Var) and not isinstance(var, Starred):
            raise NotSupportedError(var.range(),
                                    "the only expressions allowed on the left hand side of "
                                    "assignments are variable names and starred expressions")
        return var

    @staticmethod
    def build_Assign(ctx, stmt):
        rhs = build_expr(ctx, stmt.value)
        if len(stmt.targets) > 1:
            start_point = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + 1)
            raise NotSupportedError(ctx.make_raw_range(start_point.start, rhs.range().end),
                                    "Performing multiple assignments in a single line isn't supported")
        py_lhs = stmt.targets[0]
        py_lhs_exprs = py_lhs.elts if isinstance(py_lhs, ast.Tuple) else [py_lhs]
        return Assign([StmtBuilder.get_assign_lhs_expr(ctx, e) for e in py_lhs_exprs], '=', rhs)

    @staticmethod
    def build_Return(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("return"))
        values = (stmt.value,) if not isinstance(stmt.value, ast.Tuple) else stmt.value.elts
        return Return(r, [build_expr(ctx, val) for val in values if val is not None])

    @staticmethod
    def build_AugAssign(ctx, stmt):
        lhs = [StmtBuilder.get_assign_lhs_expr(ctx, stmt.target)]
        rhs = build_expr(ctx, stmt.value)
        op = type(stmt.op)
        if op in StmtBuilder.augassign_map:
            op_token = StmtBuilder.augassign_map[op]
        else:
            raise NotSupportedError(
                find_before(ctx, rhs.range().start, '=', offsets=(-1, 0)),
                "unsupported kind of augumented assignment: " + op.__name__)
        return Assign(lhs, op_token, rhs)

    @staticmethod
    def build_While(ctx, stmt):
        if stmt.orelse:
            # TODO: try to recover the location of else:? Python doesn't give us useful
            # annotations in this case
            raise NotSupportedError(None, "else branches of while loops aren't supported")
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("while"))
        return While(r, build_expr(ctx, stmt.test), [build_stmt(ctx, s) for s in stmt.body])

    @staticmethod
    def build_For(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("for"))
        return For(
            r, [StmtBuilder.get_assign_lhs_expr(ctx, stmt.target)],
            [build_expr(ctx, stmt.iter)], [build_stmt(ctx, s) for s in stmt.body])

    @staticmethod
    def build_If(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("if"))
        return If(r, build_expr(ctx, stmt.test),
                  [build_stmt(ctx, s) for s in stmt.body],
                  [build_stmt(ctx, s) for s in stmt.orelse])

    @staticmethod
    def build_Print(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("print"))
        if stmt.dest:
            raise NotSupportedError(r, "print statements with non-default destinations aren't supported")
        args = [build_expr(ctx, val) for val in stmt.values]
        return ExprStmt([Apply(Var(Ident(r, "print")), args, [])])


class ExprBuilder(Builder):
    binop_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
    }

    unop_map = {
        ast.Not: 'not',
        ast.USub: '-',
    }

    boolop_map = {
        ast.And: 'and',
        ast.Or: 'or',
    }

    cmpop_map = {
        ast.Eq: '==',
        ast.NotEq: '!=',
        ast.LtE: '<=',
        ast.Lt: '<',
        ast.GtE: '>=',
        ast.Gt: '>',
    }

    @staticmethod
    def build_Attribute(ctx, expr):
        # NB: the only attributes we support are for getting methods
        value = build_expr(ctx, expr.value)
        # <sigh> name is just a string, so it's not annotated in any way.
        source = ctx.source
        pos = find_after(ctx, value.range().end, '.').end  # Start with the dot
        while source[pos] in string.whitespace:  # Skip whitespace
            pos += 1
        start_pos = pos
        while source[pos] in _identifier_chars:  # Find the identifier itself
            pos += 1
        name_range = ctx.make_raw_range(start_pos, pos)
        return Select(value, Ident(name_range, expr.attr))

    @staticmethod
    def build_Call(ctx, expr):
        func = build_expr(ctx, expr.func)
        args = [build_expr(ctx, py_arg) for py_arg in expr.args]
        if hasattr(expr, 'starargs') and expr.starargs:
            stararg_expr = build_expr(ctx, expr.starargs)
            args += [Starred(stararg_expr.range(), stararg_expr)]
        kwargs = []
        for kw in expr.keywords:
            kw_expr = build_expr(ctx, kw.value)
            # XXX: we could do a better job at figuring out the range for the name here
            kwargs.append(Attribute(Ident(kw_expr.range(), kw.arg), kw_expr))
        return Apply(func, args, kwargs)

    @staticmethod
    def build_Name(ctx, expr):
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(expr.id))
        if expr.id.startswith(_reserved_prefix):
            raise NotSupportedError(r, "names of variables used in JIT-ed functions "
                                       "can't start with " + _reserved_prefix)
        if expr.id == "True":
            return TrueLiteral(r)
        elif expr.id == "False":
            return FalseLiteral(r)
        return Var(Ident(r, expr.id))

    @staticmethod
    def build_NameConstant(ctx, expr):
        text = "True" if expr.value else "False"
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(text))
        if expr.value:
            return TrueLiteral(r)
        else:
            return FalseLiteral(r)

    @staticmethod
    def build_BinOp(ctx, expr):
        lhs = build_expr(ctx, expr.left)
        rhs = build_expr(ctx, expr.right)
        op = type(expr.op)
        op_token = ExprBuilder.binop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise NotSupportedError(err_range, "unsupported binary operator: " + op.__name__)
        return BinOp(op_token, lhs, rhs)

    @staticmethod
    def build_UnaryOp(ctx, expr):
        sub_expr = build_expr(ctx, expr.operand)
        op = type(expr.op)
        op_token = ExprBuilder.unop_map.get(op)
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(op_token))
        if op_token is None:
            err_range = ctx.make_raw_range(r.start, sub_expr.range().end)
            raise NotSupportedError(err_range, "unsupported unary operator: " + op.__name__)
        return UnaryOp(r, op_token, sub_expr)

    @staticmethod
    def build_BoolOp(ctx, expr):
        if len(expr.values) < 2:
            raise AssertionError("expected at least 2 values in BoolOp, but got " + str(len(expr.values)))
        sub_exprs = [build_expr(ctx, sub_expr) for sub_expr in expr.values]
        op = type(expr.op)
        op_token = ExprBuilder.boolop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(sub_exprs[0].range().end, sub_exprs[1].range().start)
            raise NotSupportedError(err_range, "unsupported boolean operator: " + op.__name__)
        lhs = sub_exprs[0]
        for rhs in sub_exprs[1:]:
            lhs = BinOp(op_token, lhs, rhs)
        return lhs

    @staticmethod
    def build_IfExp(ctx, expr):
        return TernaryIf(build_expr(ctx, expr.test),
                         build_expr(ctx, expr.body),
                         build_expr(ctx, expr.orelse))

    @staticmethod
    def build_Compare(ctx, expr):
        operands = [build_expr(ctx, e) for e in [expr.left] + list(expr.comparators)]
        result = None
        for lhs, op_, rhs in zip(operands, expr.ops, operands[1:]):
            op = type(op_)
            op_token = ExprBuilder.cmpop_map.get(op)
            if op_token is None:
                err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
                raise NotSupportedError(err_range, "unsupported comparison operator: " + op.__name__)
            cmp_expr = BinOp(op_token, lhs, rhs)
            if result is None:
                result = cmp_expr
            else:
                result = BinOp('and', result, cmp_expr)
        return result

    @staticmethod
    def build_Subscript(ctx, expr):
        base = build_expr(ctx, expr.value)
        sub_type = type(expr.slice)
        if sub_type is ast.Index:
            index = build_expr(ctx, expr.slice.value)
            return Gather(base, index)
        elif sub_type is ast.Slice:
            lower = build_expr(ctx, expr.slice.lower) if expr.slice.lower is not None else None
            upper = build_expr(ctx, expr.slice.upper) if expr.slice.upper is not None else None
            if expr.slice.step is not None:
                step = build_expr(ctx, expr.slice.step)
                raise NotSupportedError(step.range(), "slices with ranges are not supported yet")
            return Slice(base, lower, upper)
        elif sub_type is ast.ExtSlice:
            raise NotSupportedError(base.range(), "slicing multiple dimensions at the same time isn't supported yet")
        else:  # Ellipsis (can only happen in Python 2)
            raise NotSupportedError(base.range(), "ellipsis is not supported")

    @staticmethod
    def build_List(ctx, expr):
        return ListLiteral(ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
                           [build_expr(ctx, e) for e in expr.elts])

    @staticmethod
    def build_Num(ctx, expr):
        value = str(expr.n)
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(value))
        return Const(r, value)

    @staticmethod
    def build_Starred(ctx, expr):
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        return Starred(r, build_expr(ctx, expr.value))

build_expr = ExprBuilder()
build_stmt = StmtBuilder()


def find_after(ctx, pos, substr, offsets=(0, 0)):
    new_pos = pos + ctx.source[pos:].index(substr)
    return ctx.make_raw_range(new_pos + offsets[0], new_pos + len(substr) + offsets[1])


def find_before(ctx, pos, substr, offsets=(0, 0)):
    new_pos = ctx.source[:pos].rindex(substr)
    return ctx.make_raw_range(new_pos + offsets[0], new_pos + len(substr) + offsets[1])
