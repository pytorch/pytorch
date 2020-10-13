"""
A pretty-printing dump function for the ast module.  The code was copied from
the ast.dump function and modified slightly to pretty-print.

Alex Leone (acleone ~AT~ gmail.com), 2010-01-30

From http://alexleone.blogspot.co.uk/2010/01/python-ast-pretty-printer.html
"""

import sys
from ast import *

def dump(node, annotate_fields=True, include_attributes=False, indent='  '):
    """
    Return a formatted dump of the tree in *node*.  This is mainly useful for
    debugging purposes.  The returned string will show the names and the values
    for fields.  This makes the code impossible to evaluate, so if evaluation is
    wanted *annotate_fields* must be set to False.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    *include_attributes* can be set to True.
    """
    def _format(node, level=0):
        if isinstance(node, AST):
            fields = [(a, _format(b, level)) for a, b in iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(a, _format(getattr(node, a), level))
                               for a in node._attributes])
            return ''.join([
                node.__class__.__name__,
                '(',
                ', '.join(('%s=%s' % field for field in fields)
                           if annotate_fields else
                           (b for a, b in fields)),
                ')'])
        elif isinstance(node, list):
            lines = ['[']
            lines.extend((indent * (level + 2) + _format(x, level + 2) + ','
                         for x in node))
            if len(lines) > 1:
                lines.append(indent * (level + 1) + ']')
            else:
                lines[-1] += ']'
            return '\n'.join(lines)
        return repr(node)

    if not isinstance(node, AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _format(node)

def parseprint(code, filename="<string>", mode="exec", type_comments=False,
               **kwargs):
    """Parse some code from a string and pretty-print it."""
    if sys.version_info >= (3, 8):
        node = parse(code, mode=mode, type_comments=type_comments)
    else:
        node = parse(code, mode=mode)   # An ode to the code
    print(dump(node, **kwargs))

# Short name: pdp = parse, dump, print
pdp = parseprint

def load_ipython_extension(ip):
    from IPython.core.magic import Magics, magics_class, cell_magic
    from IPython.core import magic_arguments

    @magics_class
    class AstMagics(Magics):

        @magic_arguments.magic_arguments()
        @magic_arguments.argument(
            '-m', '--mode', default='exec',
            help="The mode in which to parse the code. Can be exec (the default), "
                 "eval or single."
        )
        @magic_arguments.argument(
            '-t', '--type-comments', default=False,
            help="Whether to preserve the type comments in the AST"
        )
        @cell_magic
        def dump_ast(self, line, cell):
            """Parse the code in the cell, and pretty-print the AST."""
            args = magic_arguments.parse_argstring(self.dump_ast, line)
            parseprint(cell, mode=args.mode, type_comments=args.type_comments)

    ip.register_magics(AstMagics)

if __name__ == '__main__':
    import sys, tokenize
    for filename in sys.argv[1:]:
        print('=' * 50)
        print('AST tree for', filename)
        print('=' * 50)
        with tokenize.open(filename) as f:
            fstr = f.read()

        parseprint(fstr, filename=filename, include_attributes=True)
        print()


parseprint("torch.Assert(True, 'foo')")
