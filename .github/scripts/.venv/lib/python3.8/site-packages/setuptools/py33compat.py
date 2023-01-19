import dis
import array
import collections

try:
    import html
except ImportError:
    html = None

from setuptools.extern import six
from setuptools.extern.six.moves import html_parser

__metaclass__ = type

OpArg = collections.namedtuple('OpArg', 'opcode arg')


class Bytecode_compat:
    def __init__(self, code):
        self.code = code

    def __iter__(self):
        """Yield '(op,arg)' pair for each operation in code object 'code'"""

        bytes = array.array('b', self.code.co_code)
        eof = len(self.code.co_code)

        ptr = 0
        extended_arg = 0

        while ptr < eof:

            op = bytes[ptr]

            if op >= dis.HAVE_ARGUMENT:

                arg = bytes[ptr + 1] + bytes[ptr + 2] * 256 + extended_arg
                ptr += 3

                if op == dis.EXTENDED_ARG:
                    long_type = six.integer_types[-1]
                    extended_arg = arg * long_type(65536)
                    continue

            else:
                arg = None
                ptr += 1

            yield OpArg(op, arg)


Bytecode = getattr(dis, 'Bytecode', Bytecode_compat)


unescape = getattr(html, 'unescape', None)
if unescape is None:
    # HTMLParser.unescape is deprecated since Python 3.4, and will be removed
    # from 3.9.
    unescape = html_parser.HTMLParser().unescape
