# mypy: allow-untyped-defs
import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401

import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
    Apply,
    Assert,
    Assign,
    Attribute,
    AugAssign,
    BinOp,
    Break,
    ClassDef,
    Const,
    Continue,
    Decl,
    Def,
    Delete,
    DictComp,
    DictLiteral,
    Dots,
    EmptyTypeAnnotation,
    ExprStmt,
    FalseLiteral,
    For,
    Ident,
    If,
    ListComp,
    ListLiteral,
    NoneLiteral,
    Param,
    Pass,
    Property,
    Raise,
    Return,
    Select,
    SliceExpr,
    Starred,
    Stmt,
    StringLiteral,
    Subscript,
    TernaryIf,
    TrueLiteral,
    TupleLiteral,
    UnaryOp,
    Var,
    While,
    With,
    WithItem,
)
from torch._jit_internal import (  # noqa: F401
    _is_drop_fn,
    FunctionModifiers,
    is_static_fn,
    should_drop,
)
from torch._sources import (
    get_source_lines_and_file,
    make_source_context,
    parse_def,
    ParsedDef as _ParsedDef,
)
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace

_IS_ASTUNPARSE_INSTALLED = False
try:
    import astunparse  # type: ignore[import]

    _IS_ASTUNPARSE_INSTALLED = True
except ImportError:
    pass

# Borrowed from cPython implementation
# https://github.com/python/cpython/blob/561612d8456cfab5672c9b445521113b847bd6b3/Lib/textwrap.py#L411#

_reserved_prefix = "__jit"
_reserved_names = {"print"}
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

pretty_node_names.update(
    {
        ast.AsyncFunctionDef: "async function definitions",
        ast.AsyncFor: "async for loops",
        ast.AsyncWith: "async with statements",
        ast.Try: "try blocks",
        ast.Nonlocal: "nonlocal variables",
    }
)

node_start_tokens.update(
    {
        ast.AsyncFunctionDef: "async def",
        ast.AsyncFor: "async for",
        ast.AsyncWith: "async with",
        ast.Try: "try",
        ast.Nonlocal: "nonlocal",
    }
)

pretty_node_names.update(
    {
        ast.AnnAssign: "annotated assignments",
    }
)
# NB: no specific token for AnnAssign


class FrontendError(Exception):
    def __init__(self, source_range, msg):
        self.source_range = source_range
        self.msg = msg

        # This has to be instantiated here so the ErrorReport is accurate to the
        # call stack when the FrontendError was raised
        self.error_report = torch._C.ErrorReport(self.source_range)

    def __str__(self):
        return self.msg + self.error_report.what().lstrip()


class NotSupportedError(FrontendError):
    pass


class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason=""):
        # If we don't have a specific token, we default to length of 1
        node_type = type(offending_node)
        range_len = len(node_start_tokens.get(node_type, " "))
        source_range = ctx.make_range(
            offending_node.lineno,
            offending_node.col_offset,
            offending_node.col_offset + range_len,
        )
        feature_name = pretty_node_names.get(node_type, node_type.__name__)
        msg = f"{feature_name} {reason + ' ' if reason else ''}aren't supported"
        super().__init__(source_range, msg)


class FrontendTypeError(FrontendError):
    pass


def build_withitems(ctx, items):
    items = [build_withitem(ctx, i) for i in items]
    return list(items)


def build_stmts(ctx, stmts):
    stmts = [build_stmt(ctx, s) for s in stmts]
    return list(filter(None, stmts))


def get_class_properties(cls, self_name):
    """
    Get a list of Property objects representing the properties of a class.

    Args:
        cls:  The class to get properties of.
        self_name: The name of the class that the properties should belong to.
    Returns:
        A list of Property objects corresponding to the properties of cls. Property
        here refers to the subclass of TreeView.
    """
    props = inspect.getmembers(cls, predicate=lambda m: isinstance(m, property))
    # Any property that should not compiled must be in this list on the Module.
    unused_properties = getattr(cls, "__jit_unused_properties__", [])

    # Create Property TreeView objects from inspected property objects.
    properties = []
    for prop in props:
        if prop[0] not in unused_properties and not should_drop(prop[1].fget):
            getter = get_jit_def(
                prop[1].fget, f"__{prop[0]}_getter", self_name=self_name
            )
            setter = (
                get_jit_def(prop[1].fset, f"__{prop[0]}_setter", self_name=self_name)
                if prop[1].fset
                else None
            )
            properties.append(
                Property(getter.range(), Ident(getter.range(), prop[0]), getter, setter)
            )

    return properties


def get_class_assigns(ctx, cls_ast):
    assigns = []

    def maybe_build_assign(builder, entry):
        nonlocal assigns
        try:
            assigns.append(builder(ctx, entry))
        except NotSupportedError:
            pass

    for entry in cls_ast.body:
        if isinstance(entry, ast.Assign):
            maybe_build_assign(StmtBuilder.build_Assign, entry)
        elif isinstance(entry, ast.AnnAssign):
            maybe_build_assign(StmtBuilder.build_AnnAssign, entry)
    return assigns


def get_jit_class_def(cls, self_name):
    """Get definitions for each method within the current class independently.

    Args:
        cls: The class to get definition of.
        self_name: The name of the class that the properties should belong to.

    Returns:
        torch._C._jit_tree_views.ClassDef: A representation of the class,
            the methods in the class and their definition as a tree.
    """
    # TODO: proper overriding analysis when implementing class inheritance
    methods = inspect.getmembers(
        cls,
        predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m))
        and not is_static_fn(cls, m.__name__)
        and m.__name__ in cls.__dict__
        and not _is_drop_fn(m),
    )

    def is_classmethod(fn):
        return inspect.ismethod(fn) and getattr(fn, "__self__", None) == cls

    # Get and parse the source code for this class
    sourcelines, file_lineno, filename = get_source_lines_and_file(
        cls, torch._C.ErrorReport.call_stack()
    )
    source = "".join(sourcelines)

    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)

    class_ast = py_ast.body[0]
    assert isinstance(class_ast, ast.ClassDef)

    # Special case for dataclasses. In general we need access to the source code for
    # an object in order to JIT compile it. But the dataclasses module dynamically synthesizes
    # magic methods for classes, and we can't get the source code for these methods. As a
    # workaround, we synthesize TorchScript-friendly implementations ourselves.
    if dataclasses.is_dataclass(cls):
        # Detect whether the user manually implemented any of the magic methods. If they did,
        # we don't want to synthesize/override them.
        overrides = {
            method.name
            for method in class_ast.body
            if isinstance(method, ast.FunctionDef)
            and method.name in DATACLASS_MAGIC_METHODS
        }
        for i, (name, _) in enumerate(methods):
            # Is this a magic method we can synthesize?
            synthesizer_fn = DATACLASS_MAGIC_METHODS.get(name)
            if synthesizer_fn and name not in overrides:
                parsed_def = synthesizer_fn(cls)
                methods[i] = name, parsed_def
                func = getattr(cls, name)
                _jit_internal.loader.cache(func, parsed_def.source)

    method_defs = [
        get_jit_def(obj, name, self_name=self_name, is_classmethod=is_classmethod(obj))
        for (name, obj) in methods
    ]
    properties = get_class_properties(cls, self_name)

    leading_whitespace_len = len(source.split("\n", 1)[0]) - len(
        dedent_src.split("\n", 1)[0]
    )
    ctx = make_source_context(
        source, filename, file_lineno, leading_whitespace_len, False
    )
    assigns = get_class_assigns(ctx, class_ast)

    return build_class_def(ctx, class_ast, method_defs, properties, self_name, assigns)


def get_jit_def(fn, def_name, self_name=None, is_classmethod=False):
    """
    Build a JIT AST (TreeView) from the given function.

    Args:
        fn: A function object to compile or a pre-parsed ParsedDef object
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: If this function is a method, what the type name of `self` is.
    """
    parsed_def = parse_def(fn) if not isinstance(fn, _ParsedDef) else fn
    type_line = torch.jit.annotations.get_type_line(parsed_def.source)
    fn_def = parsed_def.ast.body[0]

    if is_classmethod:
        arg_name = fn_def.args.args[0].arg
        # Insert a statement that assigns the first argument to the class
        assign_stmt = ast.parse(f"{arg_name} = {self_name}").body[0]
        fn_def.body.insert(0, assign_stmt)

    # Swap out the function signature and body if it is unused
    if should_drop(fn):
        unused_fn_def = ast.parse(
            'def unused_fn(self: Any):\n\traise RuntimeError("Cannot call @unused methods")'
        )
        if len(unused_fn_def.body) != 1 or not isinstance(
            unused_fn_def.body[0], ast.FunctionDef
        ):
            raise RuntimeError(
                f"Expected a single top-level function: {parsed_def.filename}:{parsed_def.file_lineno}"
            )
        unused_def = unused_fn_def.body[0]
        fn_def.body = unused_def.body
        # kwarg/vararg not supported by `build_def`
        fn_def.args.kwarg = fn_def.args.vararg = None
        for arg in fn_def.args.args + fn_def.args.kwonlyargs:
            # Replace potentially unsupported type annotations by "Any"
            arg.annotation = unused_def.args.args[0].annotation
        if _is_drop_fn(fn):
            # Dropping potentially unsupported return type annotation for jit._drop
            fn_def.returns = None
            fn_def.type_comment = None

    # If MonkeyType is installed, get all the consolidated type traces
    # for the arguments from type_trace_db
    type_trace_db = torch.jit._script._get_type_trace_db()
    pdt_arg_types = None
    if monkeytype_trace and not isinstance(fn, _ParsedDef):  # type: ignore[truthy-function]
        qualname = get_qualified_name(fn)
        pdt_arg_types = type_trace_db.get_args_types(qualname)

    return build_def(
        parsed_def.ctx,
        fn_def,
        type_line,
        def_name,
        self_name=self_name,
        pdt_arg_types=pdt_arg_types,
    )


# TODO: more robust handling of recognizing ignore context manager
def is_torch_jit_ignore_context_manager(stmt):
    # checks if the statement is torch.jit.ignore context manager
    if isinstance(stmt.items[0].context_expr, ast.Call):
        # extract torch part
        function = stmt.items[0].context_expr.func
        if isinstance(function, ast.Attribute):
            attr_name = function.attr
            attr_value = function.value
            if attr_name == "_IgnoreContextManager" and isinstance(
                attr_value, ast.Attribute
            ):
                # there should be at most two nested attributes (e.g torch.jit._IgnoreContextManager)
                if attr_value.attr == "jit" and isinstance(attr_value.value, ast.Name):
                    if attr_value.value.id == "torch":
                        return True
    return False


class Builder:
    def __call__(self, ctx, node):
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            raise UnsupportedNodeError(ctx, node)
        return method(ctx, node)


def build_class_def(ctx, py_def, methods, properties, self_name, assigns):
    r = ctx.make_range(
        py_def.lineno, py_def.col_offset, py_def.col_offset + len("class")
    )
    return ClassDef(
        Ident(r, self_name), [Stmt(method) for method in methods], properties, assigns
    )


def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None):
    body = py_def.body
    r = ctx.make_range(py_def.lineno, py_def.col_offset, py_def.col_offset + len("def"))

    param_list = build_param_list(ctx, py_def.args, self_name, pdt_arg_types)
    return_type = None
    if getattr(py_def, "returns", None) is not None:
        return_type = build_expr(ctx, py_def.returns)

    decl = Decl(r, param_list, return_type)
    is_method = self_name is not None
    if type_line is not None:
        type_comment_decl = torch._C.parse_type_comment(type_line)
        decl = torch._C.merge_type_from_type_comment(decl, type_comment_decl, is_method)

    return Def(Ident(r, def_name), decl, build_stmts(ctx, body))


_vararg_kwarg_err = (
    "Compiled functions can't take variable number of arguments "
    "or use keyword-only arguments with defaults"
)


def build_param_list(ctx, py_args, self_name, pdt_arg_types=None):
    if py_args.kwarg is not None:
        expr = py_args.kwarg
        ctx_range = ctx.make_range(
            expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg)
        )
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if py_args.vararg is not None:
        expr = py_args.vararg
        ctx_range = ctx.make_range(
            expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg)
        )
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if len(py_args.kw_defaults) > 0:
        # kw_defaults is a list of the values for the kwargs (which default to None),
        # so they don't actually have line numbers.
        for arg in py_args.kw_defaults:
            if arg is not None:
                ctx_range = build_expr(ctx, arg).range()
                raise NotSupportedError(ctx_range, _vararg_kwarg_err)

    # List of Tuple of args and type as inferred by profile directed typing
    arg_and_types = [
        (
            arg,
            pdt_arg_types[arg.arg]
            if pdt_arg_types and bool(pdt_arg_types[arg.arg])
            else None,
        )
        for arg in py_args.args
    ]
    arg_and_types_kwonlyargs = [
        (
            arg,
            pdt_arg_types[arg.arg]
            if pdt_arg_types and bool(pdt_arg_types[arg.arg])
            else None,
        )
        for arg in py_args.kwonlyargs
    ]

    result = [
        build_param(ctx, arg, self_name, kwarg_only=False, pdt_arg_type=arg_type)
        for arg, arg_type in arg_and_types
    ]
    result += [
        build_param(ctx, arg, self_name, kwarg_only=True, pdt_arg_type=arg_type)
        for arg, arg_type in arg_and_types_kwonlyargs
    ]
    return result


def build_param(ctx, py_arg, self_name, kwarg_only, pdt_arg_type=None):
    # NB: In Python3 py_arg is a pair of (str arg, expr? annotation)
    name = py_arg.arg
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    if getattr(py_arg, "annotation", None) is not None:
        annotation_expr = build_expr(ctx, py_arg.annotation)
    elif pdt_arg_type:
        annotation_expr = Var(Ident(r, pdt_arg_type))
    elif self_name is not None and name == "self":
        annotation_expr = Var(Ident(r, self_name))
    else:
        annotation_expr = EmptyTypeAnnotation(r)
    return Param(annotation_expr, Ident(r, name), kwarg_only)


def build_ignore_context_manager(ctx, stmt):
    InputType = namedtuple("InputType", ["name", "ann"])
    OutputType = namedtuple("OutputType", ["name", "ann"])

    def process_ins_outs(args):
        # parse the context manager to figure out inputs and outputs
        # with their annotated types
        # TODO: add input, output validator
        inputs = []
        outputs = []
        for arg in args:
            var_name = arg.arg
            var_ann = arg.value.value
            var_decl_type, var_ann = var_ann.split(":")
            if var_decl_type == "inp":
                inputs.append(InputType(var_name, var_ann))
            if var_decl_type == "out":
                outputs.append(OutputType(var_name, var_ann))
        return inputs, outputs

    def create_unique_name_ext(ctx, stmt):
        # extension will be based on the full path filename plus
        # the line number of original context manager
        fn = re.sub(r"[^a-zA-Z0-9_]", "_", ctx.filename)
        return f"{fn}_{stmt.lineno}"

    def build_return_ann_stmt(outputs):
        return_type_ann = ""
        return_statement_str = "return "
        if len(outputs) == 0:
            return_type_ann += " -> None"
        if len(outputs) == 1:
            return_type_ann = " -> " + outputs[0].ann
            return_statement_str += outputs[0].name
        if len(outputs) > 1:
            return_type_ann = " -> Tuple"
            return_type_ann += "[" + ", ".join([var.ann for var in outputs]) + "]"
            return_statement_str += ", ".join([var.name for var in outputs])
        return return_type_ann, return_statement_str

    def build_args(args):
        return ", ".join([arg.name for arg in args])

    inputs, outputs = process_ins_outs(stmt.items[0].context_expr.keywords)

    # build the replacement function str with given inputs and outputs
    ignore_function_name = "func_ignore_" + create_unique_name_ext(ctx, stmt)
    ignore_function_str = "\ndef " + ignore_function_name
    ignore_function_str += (
        "(" + ", ".join([var.name + " :" + var.ann for var in inputs]) + ")"
    )

    return_ann, return_stmt = build_return_ann_stmt(outputs)
    ignore_function_str += return_ann + ": pass"

    # first create the functionDef object from just declaration
    ignore_function = ast.parse(ignore_function_str).body[0]

    # dump the body of context manager to dummy function
    ignore_function.body = stmt.body  # type: ignore[attr-defined]

    # insert return statement to the function
    return_stmt = ast.parse(return_stmt).body[0]
    ignore_function.body.append(return_stmt)  # type: ignore[attr-defined]

    # registers the custom function in the global context
    ignore_func_str = "@torch.jit.ignore\n" + astunparse.unparse(ignore_function)
    ignore_func_str += f'\nglobals()["{ignore_function_name}"] = {ignore_function_name}'
    exec(ignore_func_str)  # noqa: P204

    # build the statements as:
    # <out_1>, <out_2>, ... = torch.jit.frontend.<func>(<in_1>, <in_2>)
    assign_str_lhs = build_args(outputs)
    # this function will be registered in torch.jit.frontend module by default
    assign_str_rhs = (
        f"torch.jit.frontend.{ignore_function_name}(" + build_args(inputs) + ")"
    )

    if len(outputs) > 0:
        assign_str = assign_str_lhs + " = " + assign_str_rhs
    else:
        assign_str = assign_str_rhs
    assign_ast = ast.parse(assign_str).body[0]
    return assign_ast


def get_default_args(fn):
    """
    Get a dictionary of default arguments for a function.

    Args:
        fn: Callable - The function to inspect for default arguments.
    Returns:
        (Dict[str, Any]): mapping argument names to their default values if
        :attr:`fn` is not None, else empty dictionary.
    """
    if fn is None:
        return {}

    signature = inspect.signature(fn)

    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_default_args_for_class(cls):
    """
    Get default arguments for all methods in a class (except for static methods).

    Args:
        cls: type - The class type to inspect for default arguments.
    Returns:
        A Dict[str, Dict[str, Any]] which maps each method name to a Dict[str, Any]
        that maps each argument name to its default value.
    """
    # Get methods (except static methods because those are compiled separately as
    # if they were independent script functions).
    methods = inspect.getmembers(
        cls,
        predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m))
        and not is_static_fn(cls, m.__name__)
        and m.__name__ in cls.__dict__,
    )

    # Get method defaults. Property defaults do not need to be considered
    # because setters cannot be invoked without a value.
    defaults = {
        method_name: get_default_args(method_impl)
        for method_name, method_impl in methods
    }

    return defaults


class WithItemBuilder(Builder):
    @staticmethod
    def build_withitem(ctx, item):
        lineno = item.context_expr.lineno
        start = item.context_expr.col_offset
        end = start + len(pretty_node_names[ast.With])
        op_vars = item.optional_vars
        r = ctx.make_range(lineno, start, end)

        return WithItem(
            r,
            build_expr(ctx, item.context_expr),
            build_expr(ctx, op_vars) if op_vars else None,
        )


class StmtBuilder(Builder):
    augassign_map = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.BitOr: "|",
        ast.BitAnd: "&",
        ast.BitXor: "^",
        ast.LShift: "<<",
        ast.RShift: ">>",
        ast.Pow: "**",
    }

    @staticmethod
    def build_Expr(ctx, stmt):
        value = stmt.value
        if value.__class__.__name__ == "Str":
            # If a statement is a string literal expression,
            # then it is a docstring. Just ignore it.
            return None
        else:
            return ExprStmt(build_expr(ctx, value))

    @staticmethod
    def build_Assign(ctx, stmt):
        rhs = build_expr(ctx, stmt.value)
        lhs = [build_expr(ctx, x) for x in stmt.targets]
        return Assign(lhs, rhs)

    @staticmethod
    def build_AnnAssign(ctx, stmt):
        if stmt.value is None:
            raise UnsupportedNodeError(ctx, stmt, reason="without assigned value")

        # Disallow type annotations on instance attributes outside of __init__
        if (
            type(stmt.target) == ast.Attribute
            and stmt.target.value.id == "self"  # type: ignore[attr-defined]
            and ctx.funcname != "__init__"
        ):
            start = stmt.col_offset
            end = start + len(f"self.{stmt.target.attr}")
            if hasattr(stmt.annotation, "id"):
                end += len(f": {stmt.annotation.id}")
            sr = ctx.make_range(stmt.lineno, start, end)
            raise ValueError(
                "Type annotations on instance attributes must be declared in "
                f"__init__, not '{ctx.funcname}': {sr}"
            )

        rhs = build_expr(ctx, stmt.value)
        lhs = build_expr(ctx, stmt.target)
        the_type = build_expr(ctx, stmt.annotation)
        return Assign([lhs], rhs, the_type)

    @staticmethod
    def build_Delete(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("del"))

        return Delete(r, [build_expr(ctx, target) for target in stmt.targets])

    @staticmethod
    def build_Return(ctx, stmt):
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("return")
        )
        return Return(r, None if stmt.value is None else build_expr(ctx, stmt.value))

    @staticmethod
    def build_Raise(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("raise"))
        expr = build_expr(ctx, stmt.exc)
        return Raise(r, expr)

    @staticmethod
    def build_Assert(ctx, stmt):
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("assert")
        )
        test = build_expr(ctx, stmt.test)
        msg = build_expr(ctx, stmt.msg) if stmt.msg is not None else None
        return Assert(r, test, msg)

    @staticmethod
    def build_AugAssign(ctx, stmt):
        lhs = build_expr(ctx, stmt.target)
        rhs = build_expr(ctx, stmt.value)
        op = type(stmt.op)
        if op in StmtBuilder.augassign_map:
            op_token = StmtBuilder.augassign_map[op]
        else:
            raise NotSupportedError(
                find_before(ctx, rhs.range().start, "=", offsets=(-1, 0)),
                "unsupported kind of augmented assignment: " + op.__name__,
            )
        return AugAssign(lhs, op_token, rhs)

    @staticmethod
    def build_While(ctx, stmt):
        if stmt.orelse:
            # TODO: try to recover the location of else:? Python doesn't give us useful
            # annotations in this case
            raise NotSupportedError(
                None, "else branches of while loops aren't supported"
            )
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("while"))
        return While(r, build_expr(ctx, stmt.test), build_stmts(ctx, stmt.body))

    @staticmethod
    def build_For(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("for"))
        if stmt.orelse:
            raise NotSupportedError(r, "else branches of for loops aren't supported")

        return For(
            r,
            [build_expr(ctx, stmt.target)],
            [build_expr(ctx, stmt.iter)],
            build_stmts(ctx, stmt.body),
        )

    @staticmethod
    def build_If(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("if"))
        return If(
            r,
            build_expr(ctx, stmt.test),
            build_stmts(ctx, stmt.body),
            build_stmts(ctx, stmt.orelse),
        )

    @staticmethod
    def build_Print(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("print"))
        if stmt.dest:
            raise NotSupportedError(
                r, "print statements with non-default destinations aren't supported"
            )
        args = [build_expr(ctx, val) for val in stmt.values]
        return ExprStmt(Apply(Var(Ident(r, "print")), args, []))

    @staticmethod
    def build_Pass(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("pass"))
        return Pass(r)

    @staticmethod
    def build_Break(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("break"))
        return Break(r)

    @staticmethod
    def build_Continue(ctx, stmt):
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("continue")
        )
        return Continue(r)

    @staticmethod
    def build_With(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("with"))
        # Handle ignore context manager
        if is_torch_jit_ignore_context_manager(stmt):
            if not _IS_ASTUNPARSE_INSTALLED:
                raise RuntimeError(
                    "torch.jit._IgnoreContextManager requires installing Python library `astunparse`, \
                                   please install it in your Python environment"
                )
            assign_ast = build_ignore_context_manager(ctx, stmt)
            return build_stmt(ctx, assign_ast)
        return With(r, build_withitems(ctx, stmt.items), build_stmts(ctx, stmt.body))


class ExprBuilder(Builder):
    binop_map = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Pow: "**",
        ast.Mod: "%",
        ast.FloorDiv: "//",
        ast.BitAnd: "&",
        ast.BitXor: "^",
        ast.BitOr: "|",
        ast.LShift: "<<",
        ast.RShift: ">>",
    }

    binop_map[ast.MatMult] = "@"

    unop_map = {
        ast.Not: "not",
        ast.USub: "-",
        ast.Invert: "~",
    }

    boolop_map = {
        ast.And: "and",
        ast.Or: "or",
    }

    cmpop_map = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.LtE: "<=",
        ast.Lt: "<",
        ast.GtE: ">=",
        ast.Gt: ">",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }

    @staticmethod
    def build_Attribute(ctx, expr):
        base = build_expr(ctx, expr.value)
        # expr.attr is just a string, so it's not annotated in any way, so we have
        # to build the range manually
        source = ctx.source.encode("utf-8")

        def get_char(index):
            return chr(source[index])

        start_pos = base.range().end + 1
        while get_char(start_pos) in string.whitespace:  # Skip whitespace
            start_pos += 1
        end_pos = start_pos + len(expr.attr)
        name_range = ctx.make_raw_range(start_pos, end_pos)
        return Select(base, Ident(name_range, expr.attr))

    @staticmethod
    def build_Call(ctx, expr):
        func = build_expr(ctx, expr.func)
        args = [build_expr(ctx, py_arg) for py_arg in expr.args]
        if hasattr(expr, "starargs") and expr.starargs:
            stararg_expr = build_expr(ctx, expr.starargs)
            args += [Starred(stararg_expr.range(), stararg_expr)]
        kwargs = []
        for kw in expr.keywords:
            kw_expr = build_expr(ctx, kw.value)
            # XXX: we could do a better job at figuring out the range for the name here
            if not kw.arg:
                raise NotSupportedError(
                    kw_expr.range(), "keyword-arg expansion is not supported"
                )
            kwargs.append(Attribute(Ident(kw_expr.range(), kw.arg), kw_expr))
        return Apply(func, args, kwargs)

    @staticmethod
    def build_Ellipsis(ctx, expr):
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + 3
        )  # len("...") == 3
        return Dots(r)

    @staticmethod
    def build_Name(ctx, expr):
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(expr.id))
        if expr.id.startswith(_reserved_prefix):
            raise NotSupportedError(
                r,
                "names of variables used in JIT-ed functions "
                "can't start with " + _reserved_prefix,
            )
        if expr.id == "True":
            return TrueLiteral(r)
        elif expr.id == "False":
            return FalseLiteral(r)
        elif expr.id == "None":
            return NoneLiteral(r)
        elif expr.id == "Ellipsis":
            return Dots(r)
        return Var(Ident(r, expr.id))

    @staticmethod
    def build_NameConstant(ctx, expr):
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(str(expr.value))
        )
        if expr.value is True:
            return TrueLiteral(r)
        elif expr.value is False:
            return FalseLiteral(r)
        elif expr.value is None:
            return NoneLiteral(r)
        elif expr.value == Ellipsis:
            return Dots(r)
        else:
            raise ValueError("Name constant value unsupported: " + str(expr.value))

    @staticmethod
    def build_BinOp(ctx, expr):
        lhs = build_expr(ctx, expr.left)
        rhs = build_expr(ctx, expr.right)
        op = type(expr.op)

        if op == ast.Div and not ctx.uses_true_division:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise FrontendError(
                err_range,
                "Division of ints in TorchScript uses Python 3 true "
                "division semantics. Please put `from __future__ "
                "import division` at the top of your file",
            )
        op_token = ExprBuilder.binop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise NotSupportedError(
                err_range, "unsupported binary operator: " + op.__name__
            )
        return BinOp(op_token, lhs, rhs)

    @staticmethod
    def build_UnaryOp(ctx, expr):
        sub_expr = build_expr(ctx, expr.operand)
        op = type(expr.op)
        op_token = ExprBuilder.unop_map.get(op)
        if op_token is None:
            raise NotSupportedError(
                expr.range(), "unsupported unary operator: " + op.__name__
            )
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(op_token)
        )
        return UnaryOp(r, op_token, sub_expr)

    @staticmethod
    def build_BoolOp(ctx, expr):
        if len(expr.values) < 2:
            raise AssertionError(
                "expected at least 2 values in BoolOp, but got " + str(len(expr.values))
            )
        sub_exprs = [build_expr(ctx, sub_expr) for sub_expr in expr.values]
        op = type(expr.op)
        op_token = ExprBuilder.boolop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(
                sub_exprs[0].range().end, sub_exprs[1].range().start
            )
            raise NotSupportedError(
                err_range, "unsupported boolean operator: " + op.__name__
            )
        lhs = sub_exprs[0]
        for rhs in sub_exprs[1:]:
            lhs = BinOp(op_token, lhs, rhs)
        return lhs

    @staticmethod
    def build_IfExp(ctx, expr):
        return TernaryIf(
            build_expr(ctx, expr.test),
            build_expr(ctx, expr.body),
            build_expr(ctx, expr.orelse),
        )

    @staticmethod
    def build_Compare(ctx, expr):
        operands = [build_expr(ctx, e) for e in [expr.left] + list(expr.comparators)]
        result = None
        for lhs, op_, rhs in zip(operands, expr.ops, operands[1:]):
            op = type(op_)
            op_token = ExprBuilder.cmpop_map.get(op)
            r = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            if op_token is None:
                raise NotSupportedError(
                    r, "unsupported comparison operator: " + op.__name__
                )

            if op == ast.NotIn:
                # NB: `not in` is just `not( in )`, so we don't introduce new tree view
                # but just make it a nested call in our tree view structure
                in_expr = BinOp("in", lhs, rhs)
                cmp_expr = UnaryOp(r, "not", in_expr)
            else:
                cmp_expr = BinOp(op_token, lhs, rhs)

            if result is None:
                result = cmp_expr
            else:
                result = BinOp("and", result, cmp_expr)
        return result

    @staticmethod
    def build_Subscript(ctx, expr):
        def build_SliceExpr(ctx, base, slice_expr):
            lower = (
                build_expr(ctx, slice_expr.lower)
                if slice_expr.lower is not None
                else None
            )
            upper = (
                build_expr(ctx, slice_expr.upper)
                if slice_expr.upper is not None
                else None
            )
            step = (
                build_expr(ctx, slice_expr.step)
                if slice_expr.step is not None
                else None
            )
            return SliceExpr(base.range(), lower, upper, step)

        def build_Index(ctx, base, index_expr):
            if isinstance(index_expr.value, ast.Tuple):
                raise NotSupportedError(
                    base.range(),
                    "slicing multiple dimensions with tuples not supported yet",
                )
            return build_expr(ctx, index_expr.value)

        def build_ExtSlice(ctx, base, extslice):
            sub_exprs = []
            for expr in extslice.dims:
                sub_type = type(expr)
                if sub_type is ast.Index:
                    sub_exprs.append(build_Index(ctx, base, expr))
                elif sub_type is ast.Slice:
                    sub_exprs.append(build_SliceExpr(ctx, base, expr))
                elif sub_type is ast.Constant and expr.value is Ellipsis:
                    sub_exprs.append(Dots(base.range()))
                else:
                    raise NotSupportedError(
                        base.range(),
                        f"slicing multiple dimensions with {sub_type} not supported",
                    )
            return sub_exprs

        base = build_expr(ctx, expr.value)
        sub_type = type(expr.slice)
        if sub_type is ast.Index:
            if isinstance(expr.slice.value, ast.Tuple):
                # N-dimensional indexing using Tuple: x[(i, j, k)] is equivalent to x[i, j, k]
                # XXX: Indexing using a list is **different**! It triggers advanced indexing.
                indices = [
                    build_expr(ctx, index_expr) for index_expr in expr.slice.value.elts
                ]
                if not indices:
                    # `col_offset` is an int, but `end_col_offset` is
                    # `Optional[int]`. The magic number is here to make
                    # sure we can parse `()` on any machine
                    r = ctx.make_range(
                        expr.lineno,
                        expr.slice.value.col_offset,
                        expr.slice.value.col_offset + 2,
                    )
                    tup = TupleLiteral(r, [])
                    indices.append(tup)
                return Subscript(base, indices)
            else:
                return Subscript(base, [build_expr(ctx, expr.slice.value)])
        elif sub_type is ast.Slice:
            return Subscript(base, [build_SliceExpr(ctx, base, expr.slice)])
        elif sub_type is ast.ExtSlice:
            return Subscript(base, build_ExtSlice(ctx, base, expr.slice))
        elif sys.version_info >= (
            3,
            9,
        ):  # In Python3.9 array indicies are not wrapped in ast.Index
            if sub_type is ast.Tuple:
                # N-dimensional indexing using Tuple: x[(i, j, k)] is equivalent to x[i, j, k]
                indices = []
                for index_expr in expr.slice.elts:
                    if isinstance(index_expr, ast.Slice):
                        indices.append(build_SliceExpr(ctx, base, index_expr))
                    else:
                        indices.append(build_expr(ctx, index_expr))
                # Special-case logic for `typing.Tuple[()]`
                if not indices:
                    # See note above r.e. magic number
                    r = ctx.make_range(
                        expr.lineno, expr.slice.col_offset, expr.slice.col_offset + 2
                    )
                    tup = TupleLiteral(r, [])
                    indices.append(tup)
                return Subscript(base, indices)
            return Subscript(base, [build_expr(ctx, expr.slice)])
        else:  # Ellipsis (can only happen in Python 2)
            raise NotSupportedError(base.range(), "ellipsis is not supported")

    @staticmethod
    def build_List(ctx, expr):
        return ListLiteral(
            ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
            [build_expr(ctx, e) for e in expr.elts],
        )

    @staticmethod
    def build_Tuple(ctx, expr):
        return TupleLiteral(
            ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
            [build_expr(ctx, e) for e in expr.elts],
        )

    @staticmethod
    def build_Dict(ctx, expr):
        range = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        if expr.keys and not expr.keys[0]:
            raise NotSupportedError(
                range, "Dict expansion (e.g. `{**dict}`) is not supported"
            )
        return DictLiteral(
            range,
            [build_expr(ctx, e) for e in expr.keys],
            [build_expr(ctx, e) for e in expr.values],
        )

    @staticmethod
    def build_Num(ctx, expr):
        value = str(expr.value)
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(value))
        return Const(r, value)

    @staticmethod
    def build_Constant(ctx, expr):
        value = expr.value
        if value is None or isinstance(value, bool):
            # NB: this check has to happen before the int check because bool is
            # a subclass of int
            return ExprBuilder.build_NameConstant(ctx, expr)
        if isinstance(value, (int, float, complex)):
            return ExprBuilder.build_Num(ctx, expr)
        elif isinstance(value, str):
            return ExprBuilder.build_Str(ctx, expr)
        elif isinstance(value, type(Ellipsis)):
            return ExprBuilder.build_Ellipsis(ctx, expr)
        else:
            error_range = ctx.make_range(
                expr.lineno, expr.col_offset, expr.col_offset + len(str(value))
            )
            raise FrontendError(error_range, "Unknown Constant expression type")

    @staticmethod
    def build_Str(ctx, expr):
        value = str(expr.value)
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(value) + 1
        )
        return StringLiteral(r, value)

    @staticmethod
    def build_JoinedStr(ctx, expr):
        s = ""
        args = []
        for value in expr.values:
            r = ctx.make_range(value.lineno, value.col_offset, value.col_offset + 1)
            if isinstance(value, ast.FormattedValue):
                if value.conversion != -1:
                    raise NotSupportedError(r, "Don't support conversion in JoinedStr")
                if value.format_spec is not None:
                    raise NotSupportedError(r, "Don't support formatting in JoinedStr")
                s += "{}"
                args.append(build_expr(ctx, value.value))
            elif isinstance(value, ast.Constant):
                s += value.value
            else:
                raise NotSupportedError(r, "Unsupported value in JoinedStr")

        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        return Apply(Select(StringLiteral(r, s), Ident(r, "format")), args, [])

    @staticmethod
    def build_ListComp(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
        if len(stmt.generators) != 1:
            raise NotSupportedError(r, "Only a single generator is currently supported")

        if len(stmt.generators[0].ifs) != 0:
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        elt_expr = build_expr(ctx, stmt.elt)
        target_expr = build_expr(ctx, stmt.generators[0].target)
        iter_expr = build_expr(ctx, stmt.generators[0].iter)

        return ListComp(r, elt_expr, target_expr, iter_expr)

    @staticmethod
    def build_GeneratorExp(ctx, stmt):
        # Convert Generator expression to ListComp
        return ExprBuilder.build_ListComp(ctx, stmt)

    @staticmethod
    def build_DictComp(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
        if len(stmt.generators) != 1:
            raise NotSupportedError(r, "Only a single generator is currently supported")

        if len(stmt.generators[0].ifs) != 0:
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        key_expr = build_expr(ctx, stmt.key)
        value_expr = build_expr(ctx, stmt.value)
        target_expr = build_expr(ctx, stmt.generators[0].target)
        iter_expr = build_expr(ctx, stmt.generators[0].iter)

        return DictComp(r, key_expr, value_expr, target_expr, iter_expr)

    @staticmethod
    def build_Starred(ctx, expr):
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        return Starred(r, build_expr(ctx, expr.value))


build_expr = ExprBuilder()
build_stmt = StmtBuilder()
build_withitem = WithItemBuilder()


def find_before(ctx, pos, substr, offsets=(0, 0)):
    new_pos = ctx.source[:pos].rindex(substr)
    return ctx.make_raw_range(new_pos + offsets[0], new_pos + len(substr) + offsets[1])
