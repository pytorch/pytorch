from itertools import chain
from sympy.codegen.fnodes import Module
from sympy.core.symbol import Dummy
from sympy.printing.fortran import FCodePrinter

""" This module collects utilities for rendering Fortran code. """


def render_as_module(definitions, name, declarations=(), printer_settings=None):
    """ Creates a ``Module`` instance and renders it as a string.

    This generates Fortran source code for a module with the correct ``use`` statements.

    Parameters
    ==========

    definitions : iterable
        Passed to :class:`sympy.codegen.fnodes.Module`.
    name : str
        Passed to :class:`sympy.codegen.fnodes.Module`.
    declarations : iterable
        Passed to :class:`sympy.codegen.fnodes.Module`. It will be extended with
        use statements, 'implicit none' and public list generated from ``definitions``.
    printer_settings : dict
        Passed to ``FCodePrinter`` (default: ``{'standard': 2003, 'source_format': 'free'}``).

    """
    printer_settings = printer_settings or {'standard': 2003, 'source_format': 'free'}
    printer = FCodePrinter(printer_settings)
    dummy = Dummy()
    if isinstance(definitions, Module):
        raise ValueError("This function expects to construct a module on its own.")
    mod = Module(name, chain(declarations, [dummy]), definitions)
    fstr = printer.doprint(mod)
    module_use_str = '   %s\n' % '   \n'.join(['use %s, only: %s' % (k, ', '.join(v)) for
                                                k, v in printer.module_uses.items()])
    module_use_str += '   implicit none\n'
    module_use_str += '   private\n'
    module_use_str += '   public %s\n' % ', '.join([str(node.name) for node in definitions if getattr(node, 'name', None)])
    return fstr.replace(printer.doprint(dummy), module_use_str)
