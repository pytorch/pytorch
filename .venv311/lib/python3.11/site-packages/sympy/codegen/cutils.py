from sympy.printing.c import C99CodePrinter

def render_as_source_file(content, Printer=C99CodePrinter, settings=None):
    """ Renders a C source file (with required #include statements) """
    printer = Printer(settings or {})
    code_str = printer.doprint(content)
    includes = '\n'.join(['#include <%s>' % h for h in printer.headers])
    return includes + '\n\n' + code_str
