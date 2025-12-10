from sympy.printing.pycode import PythonCodePrinter

""" This module collects utilities for rendering Python code. """


def render_as_module(content, standard='python3'):
    """Renders Python code as a module (with the required imports).

    Parameters
    ==========

    standard :
        See the parameter ``standard`` in
        :meth:`sympy.printing.pycode.pycode`
    """

    printer = PythonCodePrinter({'standard':standard})
    pystr = printer.doprint(content)
    if printer._settings['fully_qualified_modules']:
        module_imports_str = '\n'.join('import %s' % k for k in printer.module_imports)
    else:
        module_imports_str = '\n'.join(['from %s import %s' % (k, ', '.join(v)) for
                                        k, v in printer.module_imports.items()])
    return module_imports_str + '\n\n' + pystr
