"""This module provides the following tools:

- triton_jit - an enhanced triton.jit decorator that provides a
    high-level API for creating triton kernels.
"""

import inspect
import re
import sys
import typing

if sys.version_info[:2] >= (3, 9):
    Annotated = typing.Annotated
else:

    class Annotated:
        # A simple backport of typing.Annotated to Python <3.9
        def __class_getitem__(cls, key):
            T, metadata = key[0], key[1:]
            r = typing._alias(cls, T)  # type: ignore[attr-defined]
            r.__metadata__ = metadata
            return r


def triton_jit(func):
    """A decorator that patches Python function `func` by making it
    triton.jit-able, that is, all torch.Tensor arguments are replaced
    with pointer and strides arguments.

    The patching involves applying the following rules:

    1. Tensor arguments must be annotated with `Annotated[Tensor,
       dict(ndims=...)]` that specifies the corresponding
       dimensionalities of the arguments.
       Otherwise, the argument is passed on as it is.

       For example, the following function

         def foo(x: Annotated[Tensor, dict(ndim=2)], y:triton.language.constexpr): ...

       is transformed to

         def foo(x_PTR, x_STRIDE_0, x_STRIDE_1, y:triton.language.constexpr): ...

    2. Expressions using Tensor arguments are patched by applying the
       following transformations:

         x[<int-expr0>, <int-expr1>] -> tl.load(x_PTR + x_STRIDE_0 * <int-expr0> + x_STRIDE_1 * <int-expr1>)
         x[<int-expr0>, <int-expr1>].data_ptr() -> x_PTR + x_STRIDE_0 * <int-expr0> + x_STRIDE_1 * <int-expr1>
         x[<int-expr0>, <int-expr1>].copy_(y)  -> tl.store(x_PTR + x_STRIDE_0 * <int-expr0> + x_STRIDE_1 * <int-expr1>, y)
         x.stride(<int-literal>) -> x_STRIDE_<int-literal>
         x.stride() -> (x_STRIDE_0, x_STRIDE_1)
         x.dtype -> x_PTR.dtype

       For example, the expression

         x[i, j + 1].data_ptr()

       is replaced with

         triton.language.load(x_PTR + i * x_STRIDE_0 + (j + 1) * x_STRIDE_1)

       Hint: to avoid recurring loads, assign an indexing expression
       to a variable and use the variable in expressions. For
       instance, instead of

         r = x[i, j + 1] ** 2 - 2 * x[i, j + 1] 1 1

       use

         x_ijp1 = x[i, j + 1]
         r = x_ijp1 ** 2 - 2 * x_ijp1 + 1

    """

    def _mark_parenthesis(text, mark):
        m = mark.search(text)
        if m is None:
            return text
        paren = {"(": "()", "[": "[]"}[mark.pattern[-1]]
        left, right = paren[0], paren[1]
        j = m.end() + 1
        d = 1
        while d != 0:
            d += {left: 1, right: -1}.get(text[j], 0)
            j += 1

        return (
            text[: m.end() - 1]
            + "!"
            + left
            + "!"
            + text[m.end() : j - 1]
            + "!"
            + right
            + "!"
            + _mark_parenthesis(text[j:], mark)
        )

    def _split_outer_comma(text):
        """
        For example, transform `"i + m[j, 1], k - 1"` to `["i + m[j, 1]", "k - 1"]`
        """
        lst = []
        d = 0
        i0 = 0
        for i, c in enumerate(text + ","):
            d += {"[": 1, "(": 1, "{": 1, "]": -1, ")": -1, "}": -1}.get(c, 0)
            if d == 0 and c == ",":
                lst.append(text[i0:i].strip())
                i0 = i + 1
        return lst

    sig = inspect.signature(func)

    new_params = []
    new_ann = dict()
    replace_lst = []
    mark_lst = []

    for name, param in sig.parameters.items():
        assert name == param.name
        if typing.get_origin(param.annotation) is Annotated:
            ndim = param.annotation.__metadata__[0]["ndim"]

            new_params.append(inspect.Parameter(f"{name}_PTR", param.kind))
            for i in range(ndim):
                new_params.append(inspect.Parameter(f"{name}_STRIDE_{i}", param.kind))

            # Marks `x[...]` as `x![!...!]!`
            mark_lst.append(re.compile(r"\b" + name + r"\s*\[", re.MULTILINE))
            # Marks `x.stride(...)` as `x.stride!(!...!)!`
            mark_lst.append(
                re.compile(r"\b" + name + r"\s*[.]\s*stride\s*\(", re.MULTILINE)
            )

            # Transformers:

            # `x.data_ptr()` -> `x_PTR`
            data_ptr_matcher = re.compile(
                r"\b" + name + r"\s*[.]\s*data_ptr\s*\(\s*\)", re.MULTILINE
            )
            data_ptr_repl = f"{name}_PTR"
            replace_lst.append(
                lambda src, match=data_ptr_matcher, repl=data_ptr_repl: match.sub(
                    repl, src
                )
            )

            # `x[i, j].copy_(...)` -> `tl.store(x_PTR + x_STRIDE_0 * i + x_STRIDE_1 * j, ...)`
            copy_matcher = re.compile(
                r"\b"
                + name
                + r"\s*!\[!(?P<indices>[^!]*)!\]!\s*[.]\s*copy_\s*!\(!(?P<expr>[^!]*)!\)!",
                re.MULTILINE,
            )

            def copy_repl(m, name=name):
                indices = _split_outer_comma(m.group("indices"))
                expr = m.group("expr")
                return (
                    "triton.language.store("
                    + " + ".join(
                        [f"{name}_PTR"]
                        + [f"{name}_STRIDE_{i} * ({w})" for i, w in enumerate(indices)]
                    )
                    + f", {expr})"
                )

            replace_lst.append(
                lambda src, match=copy_matcher, repl=copy_repl: match.sub(repl, src)
            )

            # `x[i, j].data_ptr()` -> `x_PTR + x_STRIDE_0 * i + x_STRIDE_1 * j`
            indexing_ptr_matcher = re.compile(
                r"\b"
                + name
                + r"\s*!\[!(?P<indices>[^!]*)!\]!\s*[.]\s*data_ptr\s*\(\s*\)",
                re.MULTILINE,
            )

            def index_ptr_repl(m, name=name):
                indices = _split_outer_comma(m.group("indices"))
                return " + ".join(
                    [f"{name}_PTR"]
                    + [f"{name}_STRIDE_{i} * ({w})" for i, w in enumerate(indices)]
                )

            replace_lst.append(
                lambda src, match=indexing_ptr_matcher, repl=index_ptr_repl: match.sub(
                    repl, src
                )
            )

            # `x[i, j]` -> `tl.load(x_PTR + x_STRIDE_0 * i + x_STRIDE_1 * j)`
            indexing_matcher = re.compile(
                r"\b" + name + r"\s*(!\[!(?P<indices>[^!]*)!\]!)", re.MULTILINE
            )

            def index_repl(m, name=name):
                indices = _split_outer_comma(m.group("indices"))
                return (
                    "triton.language.load("
                    + " + ".join(
                        [f"{name}_PTR"]
                        + [f"{name}_STRIDE_{i} * ({w})" for i, w in enumerate(indices)]
                    )
                    + ")"
                )

            replace_lst.append(
                lambda src, match=indexing_matcher, repl=index_repl: match.sub(
                    repl, src
                )
            )

            # `x.stride(<int-literal>)` -> `x_STRIDE_<int-literal>`
            # `x.stride()` -> `(x_STRIDE_0, x_STRIDE_1, ..., x_STRIDE_<ndim-1>)`
            # `x.stride(<int-expr>)` -> `(x_STRIDE_0, x_STRIDE_1, ..., x_STRIDE_<ndim-1>)[<int-expr>]`
            stride_matcher = re.compile(
                r"\b" + name + r"\s*[.]\s*stride\s*!\(!(?P<index>[^!]*)!\)!",
                re.MULTILINE,
            )

            def stride_repl(m, name=name, ndim=ndim):
                index = m.group("index").strip()
                if not index:
                    # index argument is not specified
                    return (
                        "("
                        + ", ".join([f"{name}_STRIDE_{i}" for i in range(ndim)])
                        + ",)"
                    )
                if index.isdigit():
                    # index is int-literal
                    return f"{name}_STRIDE_{index}"
                # index is int-expression:
                return (
                    "("
                    + ", ".join([f"{name}_STRIDE_{i}" for i in range(ndim)])
                    + f",)[{index}]"
                )

            replace_lst.append(
                lambda src, match=stride_matcher, repl=stride_repl: match.sub(repl, src)
            )

            # `x.dtype` -> `x_PTR.dtype`
            dtype_matcher = re.compile(r"\b" + name + r"\s*[.]\s*dtype\b", re.MULTILINE)
            dtype_repl = f"{name}_PTR.dtype"
            replace_lst.append(
                lambda src, match=dtype_matcher, repl=dtype_repl: match.sub(repl, src)
            )
        else:
            new_params.append(param)
            new_ann[name] = param.annotation

    new_sig = inspect.Signature(parameters=new_params)

    # Marks `!]!.copy_(...)` as `!]!.copy_!(!...!)!`
    mark_lst.append(re.compile(r"!\]!\s*[.]\s*copy_\s*\(", re.MULTILINE))

    # Marks `def foo(...)` as `def foo!(!...!)!`
    mark_lst.append(re.compile(r"def\s+" + func.__name__ + r"\s*\(", re.MULTILINE))

    # Fix up function definition:
    def_matcher = re.compile(
        r"def\s+" + func.__name__ + r"\s*!\(![^!]*!\)!", re.MULTILINE
    )

    def def_repl(m):
        return f"def {func.__name__}{new_sig}"

    replace_lst.append(
        lambda src, match=def_matcher, repl=def_repl: match.sub(repl, src)
    )

    import triton

    func.__signature__ = new_sig
    func.__annotations__ = new_ann
    jitfunc = triton.jit(func)

    # Patch func source
    src = jitfunc.src

    # Prepare source:
    for mark in mark_lst:
        src = _mark_parenthesis(src, mark)

    # Apply transformations:
    for replace in replace_lst:
        src = replace(src)

    jitfunc.src = src

    return jitfunc
