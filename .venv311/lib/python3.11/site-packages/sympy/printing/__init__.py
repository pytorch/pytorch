"""Printing subsystem"""

from .pretty import pager_print, pretty, pretty_print, pprint, pprint_use_unicode, pprint_try_use_unicode

from .latex import latex, print_latex, multiline_latex

from .mathml import mathml, print_mathml

from .python import python, print_python

from .pycode import pycode

from .codeprinter import print_ccode, print_fcode

from .codeprinter import ccode, fcode, cxxcode, rust_code # noqa:F811

from .smtlib import smtlib_code

from .glsl import glsl_code, print_glsl

from .rcode import rcode, print_rcode

from .jscode import jscode, print_jscode

from .julia import julia_code

from .mathematica import mathematica_code

from .octave import octave_code

from .gtk import print_gtk

from .preview import preview

from .repr import srepr

from .tree import print_tree

from .str import StrPrinter, sstr, sstrrepr

from .tableform import TableForm

from .dot import dotprint

from .maple import maple_code, print_maple_code

__all__ = [
    # sympy.printing.pretty
    'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode',

    # sympy.printing.latex
    'latex', 'print_latex', 'multiline_latex',

    # sympy.printing.mathml
    'mathml', 'print_mathml',

    # sympy.printing.python
    'python', 'print_python',

    # sympy.printing.pycode
    'pycode',

    # sympy.printing.codeprinter
    'ccode', 'print_ccode', 'cxxcode', 'fcode', 'print_fcode', 'rust_code',

    # sympy.printing.smtlib
    'smtlib_code',

    # sympy.printing.glsl
    'glsl_code', 'print_glsl',

    # sympy.printing.rcode
    'rcode', 'print_rcode',

    # sympy.printing.jscode
    'jscode', 'print_jscode',

    # sympy.printing.julia
    'julia_code',

    # sympy.printing.mathematica
    'mathematica_code',

    # sympy.printing.octave
    'octave_code',

    # sympy.printing.gtk
    'print_gtk',

    # sympy.printing.preview
    'preview',

    # sympy.printing.repr
    'srepr',

    # sympy.printing.tree
    'print_tree',

    # sympy.printing.str
    'StrPrinter', 'sstr', 'sstrrepr',

    # sympy.printing.tableform
    'TableForm',

    # sympy.printing.dot
    'dotprint',

    # sympy.printing.maple
    'maple_code', 'print_maple_code',
]
