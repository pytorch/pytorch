from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from re import compile as rcompile

from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark # noqa

from .errors import LaTeXParsingError  # noqa


IGNORE_L = r"\s*[{]*\s*"
IGNORE_R = r"\s*[}]*\s*"
NO_LEFT = r"(?<!\\left)"
BEGIN_AMS_MAT = r"\\begin{matrix}"
END_AMS_MAT = r"\\end{matrix}"
BEGIN_ARR = r"\\begin{array}{.*?}"
END_ARR = r"\\end{array}"

# begin_delim_regex: end_delim_regex
MATRIX_DELIMS = {fr"\\left\({IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\)",
                 fr"{NO_LEFT}\({IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\)",
                 fr"\\left\[{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\]",
                 fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\]",
                 fr"\\left\|{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\|",
                 fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\|",
                 r"\\begin{pmatrix}": r"\\end{pmatrix}",
                 r"\\begin{bmatrix}": r"\\end{bmatrix}",
                 r"\\begin{vmatrix}": r"\\end{vmatrix}",
                 fr"\\left\({IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\)",
                 fr"{NO_LEFT}\({IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\)",
                 fr"\\left\[{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\]",
                 fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\]",
                 fr"\\left\|{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\|",
                 fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\|"
                 }

MATRIX_DELIMS_INV = {v: k for k, v in MATRIX_DELIMS.items()}

# begin_delim_regex: ideal_begin_delim_representative
BEGIN_DELIM_REPR = {fr"\\left\({IGNORE_L}{BEGIN_AMS_MAT}": "\\left(\\begin{matrix}",
                    fr"{NO_LEFT}\({IGNORE_L}{BEGIN_AMS_MAT}": "(\\begin{matrix}",
                    fr"\\left\[{IGNORE_L}{BEGIN_AMS_MAT}": "\\left[\\begin{matrix}",
                    fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_AMS_MAT}": "[\\begin{matrix}",
                    fr"\\left\|{IGNORE_L}{BEGIN_AMS_MAT}": "\\left|\\begin{matrix}",
                    fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_AMS_MAT}": "|\\begin{matrix}",
                    r"\\begin{pmatrix}": "\\begin{pmatrix}",
                    r"\\begin{bmatrix}": "\\begin{bmatrix}",
                    r"\\begin{vmatrix}": "\\begin{vmatrix}",
                    fr"\\left\({IGNORE_L}{BEGIN_ARR}": "\\left(\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\({IGNORE_L}{BEGIN_ARR}": "(\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"\\left\[{IGNORE_L}{BEGIN_ARR}": "\\left[\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_ARR}": "[\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"\\left\|{IGNORE_L}{BEGIN_ARR}": "\\left|\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_ARR}": "|\\begin{array}{COLUMN_SPECIFIERS}"
                    }

# end_delim_regex: ideal_end_delim_representative
END_DELIM_REPR = {fr"{END_AMS_MAT}{IGNORE_R}\\right\)": "\\end{matrix}\\right)",
                  fr"{END_AMS_MAT}{IGNORE_R}\)": "\\end{matrix})",
                  fr"{END_AMS_MAT}{IGNORE_R}\\right\]": "\\end{matrix}\\right]",
                  fr"{END_AMS_MAT}{IGNORE_R}\]": "\\end{matrix}]",
                  fr"{END_AMS_MAT}{IGNORE_R}\\right\|": "\\end{matrix}\\right|",
                  fr"{END_AMS_MAT}{IGNORE_R}\|": "\\end{matrix}|",
                  r"\\end{pmatrix}": "\\end{pmatrix}",
                  r"\\end{bmatrix}": "\\end{bmatrix}",
                  r"\\end{vmatrix}": "\\end{vmatrix}",
                  fr"{END_ARR}{IGNORE_R}\\right\)": "\\end{array}\\right)",
                  fr"{END_ARR}{IGNORE_R}\)": "\\end{array})",
                  fr"{END_ARR}{IGNORE_R}\\right\]": "\\end{array}\\right]",
                  fr"{END_ARR}{IGNORE_R}\]": "\\end{array}]",
                  fr"{END_ARR}{IGNORE_R}\\right\|": "\\end{array}\\right|",
                  fr"{END_ARR}{IGNORE_R}\|": "\\end{array}|"
                  }


def check_matrix_delimiters(latex_str):
    """Report mismatched, excess, or missing matrix delimiters."""
    spans = []
    for begin_delim in MATRIX_DELIMS:
        end_delim = MATRIX_DELIMS[begin_delim]

        p = rcompile(begin_delim)
        q = rcompile(end_delim)

        spans.extend([(*m.span(), m.group(),
                       begin_delim) for m in p.finditer(latex_str)])
        spans.extend([(*m.span(), m.group(),
                       end_delim) for m in q.finditer(latex_str)])

    spans.sort(key=(lambda x: x[0]))
    if len(spans) % 2 == 1:
        # Odd number of delimiters; therefore something
        # is wrong. We do not complain yet; let's see if
        # we can pinpoint the actual error.
        spans.append((None, None, None, None))

    spans = [(*x, *y) for (x, y) in zip(spans[::2], spans[1::2])]
    for x in spans:
        # x is supposed to be an 8-tuple of the following form:
        #
        # (begin_delim_span_start, begin_delim_span_end,
        # begin_delim_match, begin_delim_regex,
        # end_delim_span_start, end_delim_span_end,
        # end_delim_match, end_delim_regex)

        sellipsis = "..."
        s = x[0] - 10
        if s < 0:
            s = 0
            sellipsis = ""

        eellipsis = "..."
        e = x[1] + 10
        if e > len(latex_str):
            e = len(latex_str)
            eellipsis = ""

        if x[3] in END_DELIM_REPR:
            err = (f"Extra '{x[2]}' at index {x[0]} or "
                   "missing corresponding "
                   f"'{BEGIN_DELIM_REPR[MATRIX_DELIMS_INV[x[3]]]}' "
                   f"in LaTeX string: {sellipsis}{latex_str[s:e]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

        if x[7] is None:
            err = (f"Extra '{x[2]}' at index {x[0]} or "
                   "missing corresponding "
                   f"'{END_DELIM_REPR[MATRIX_DELIMS[x[3]]]}' "
                   f"in LaTeX string: {sellipsis}{latex_str[s:e]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

        correct_end_regex = MATRIX_DELIMS[x[3]]
        sellipsis = "..." if x[0] > 0 else ""
        eellipsis = "..." if x[5] < len(latex_str) else ""
        if x[7] != correct_end_regex:
            err = ("Expected "
                   f"'{END_DELIM_REPR[correct_end_regex]}' "
                   f"to close the '{x[2]}' at index {x[0]} but "
                   f"found '{x[6]}' at index {x[4]} of LaTeX "
                   f"string instead: {sellipsis}{latex_str[x[0]:x[5]]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

__doctest_requires__ = {('parse_latex',): ['antlr4', 'lark']}


@doctest_depends_on(modules=('antlr4', 'lark'))
def parse_latex(s, strict=False, backend="antlr"):
    r"""Converts the input LaTeX string ``s`` to a SymPy ``Expr``.

    Parameters
    ==========

    s : str
        The LaTeX string to parse. In Python source containing LaTeX,
        *raw strings* (denoted with ``r"``, like this one) are preferred,
        as LaTeX makes liberal use of the ``\`` character, which would
        trigger escaping in normal Python strings.
    backend : str, optional
        Currently, there are two backends supported: ANTLR, and Lark.
        The default setting is to use the ANTLR backend, which can be
        changed to Lark if preferred.

        Use ``backend="antlr"`` for the ANTLR-based parser, and
        ``backend="lark"`` for the Lark-based parser.

        The ``backend`` option is case-sensitive, and must be in
        all lowercase.
    strict : bool, optional
        This option is only available with the ANTLR backend.

        If True, raise an exception if the string cannot be parsed as
        valid LaTeX. If False, try to recover gracefully from common
        mistakes.

    Examples
    ========

    >>> from sympy.parsing.latex import parse_latex
    >>> expr = parse_latex(r"\frac {1 + \sqrt {\a}} {\b}")
    >>> expr
    (sqrt(a) + 1)/b
    >>> expr.evalf(4, subs=dict(a=5, b=2))
    1.618
    >>> func = parse_latex(r"\int_1^\alpha \dfrac{\mathrm{d}t}{t}", backend="lark")
    >>> func.evalf(subs={"alpha": 2})
    0.693147180559945
    """

    check_matrix_delimiters(s)

    if backend == "antlr":
        _latex = import_module(
            'sympy.parsing.latex._parse_latex_antlr',
            import_kwargs={'fromlist': ['X']})

        if _latex is not None:
            return _latex.parse_latex(s, strict)
    elif backend == "lark":
        return parse_latex_lark(s)
    else:
        raise NotImplementedError(f"Using the '{backend}' backend in the LaTeX" \
                                   " parser is not supported.")
