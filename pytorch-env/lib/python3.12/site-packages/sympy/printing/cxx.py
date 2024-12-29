"""
C++ code printer
"""

from itertools import chain
from sympy.codegen.ast import Type, none
from .codeprinter import requires
from .c import C89CodePrinter, C99CodePrinter

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import cxxcode # noqa:F401

# from https://en.cppreference.com/w/cpp/keyword
reserved = {
    'C++98': [
        'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break',
        'case', 'catch,', 'char', 'class', 'compl', 'const', 'const_cast',
        'continue', 'default', 'delete', 'do', 'double', 'dynamic_cast',
        'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float',
        'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
        'namespace', 'new', 'not', 'not_eq', 'operator', 'or', 'or_eq',
        'private', 'protected', 'public', 'register', 'reinterpret_cast',
        'return', 'short', 'signed', 'sizeof', 'static', 'static_cast',
        'struct', 'switch', 'template', 'this', 'throw', 'true', 'try',
        'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
        'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
    ]
}

reserved['C++11'] = reserved['C++98'][:] + [
    'alignas', 'alignof', 'char16_t', 'char32_t', 'constexpr', 'decltype',
    'noexcept', 'nullptr', 'static_assert', 'thread_local'
]
reserved['C++17'] = reserved['C++11'][:]
reserved['C++17'].remove('register')
# TM TS: atomic_cancel, atomic_commit, atomic_noexcept, synchronized
# concepts TS: concept, requires
# module TS: import, module


_math_functions = {
    'C++98': {
        'Mod': 'fmod',
        'ceiling': 'ceil',
    },
    'C++11': {
        'gamma': 'tgamma',
    },
    'C++17': {
        'beta': 'beta',
        'Ei': 'expint',
        'zeta': 'riemann_zeta',
    }
}

# from https://en.cppreference.com/w/cpp/header/cmath
for k in ('Abs', 'exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan',  # 'Pow'
          'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'floor'):
    _math_functions['C++98'][k] = k.lower()


for k in ('asinh', 'acosh', 'atanh', 'erf', 'erfc'):
    _math_functions['C++11'][k] = k.lower()


def _attach_print_method(cls, sympy_name, func_name):
    meth_name = '_print_%s' % sympy_name
    if hasattr(cls, meth_name):
        raise ValueError("Edit method (or subclass) instead of overwriting.")
    def _print_method(self, expr):
        return '{}{}({})'.format(self._ns, func_name, ', '.join(map(self._print, expr.args)))
    _print_method.__doc__ = "Prints code for %s" % k
    setattr(cls, meth_name, _print_method)


def _attach_print_methods(cls, cont):
    for sympy_name, cxx_name in cont[cls.standard].items():
        _attach_print_method(cls, sympy_name, cxx_name)


class _CXXCodePrinterBase:
    printmethod = "_cxxcode"
    language = 'C++'
    _ns = 'std::'  # namespace

    def __init__(self, settings=None):
        super().__init__(settings or {})

    @requires(headers={'algorithm'})
    def _print_Max(self, expr):
        from sympy.functions.elementary.miscellaneous import Max
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "%smax(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Max(*expr.args[1:])))

    @requires(headers={'algorithm'})
    def _print_Min(self, expr):
        from sympy.functions.elementary.miscellaneous import Min
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "%smin(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Min(*expr.args[1:])))

    def _print_using(self, expr):
        if expr.alias == none:
            return 'using %s' % expr.type
        else:
            raise ValueError("C++98 does not support type aliases")

    def _print_Raise(self, rs):
        arg, = rs.args
        return 'throw %s' % self._print(arg)

    @requires(headers={'stdexcept'})
    def _print_RuntimeError_(self, re):
        message, = re.args
        return "%sruntime_error(%s)" % (self._ns, self._print(message))


class CXX98CodePrinter(_CXXCodePrinterBase, C89CodePrinter):
    standard = 'C++98'
    reserved_words = set(reserved['C++98'])


# _attach_print_methods(CXX98CodePrinter, _math_functions)


class CXX11CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++11'
    reserved_words = set(reserved['C++11'])
    type_mappings = dict(chain(
        CXX98CodePrinter.type_mappings.items(),
        {
            Type('int8'): ('int8_t', {'cstdint'}),
            Type('int16'): ('int16_t', {'cstdint'}),
            Type('int32'): ('int32_t', {'cstdint'}),
            Type('int64'): ('int64_t', {'cstdint'}),
            Type('uint8'): ('uint8_t', {'cstdint'}),
            Type('uint16'): ('uint16_t', {'cstdint'}),
            Type('uint32'): ('uint32_t', {'cstdint'}),
            Type('uint64'): ('uint64_t', {'cstdint'}),
            Type('complex64'): ('std::complex<float>', {'complex'}),
            Type('complex128'): ('std::complex<double>', {'complex'}),
            Type('bool'): ('bool', None),
        }.items()
    ))

    def _print_using(self, expr):
        if expr.alias == none:
            return super()._print_using(expr)
        else:
            return 'using %(alias)s = %(type)s' % expr.kwargs(apply=self._print)

# _attach_print_methods(CXX11CodePrinter, _math_functions)


class CXX17CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++17'
    reserved_words = set(reserved['C++17'])

    _kf = dict(C99CodePrinter._kf, **_math_functions['C++17'])

    def _print_beta(self, expr):
        return self._print_math_func(expr)

    def _print_Ei(self, expr):
        return self._print_math_func(expr)

    def _print_zeta(self, expr):
        return self._print_math_func(expr)


# _attach_print_methods(CXX17CodePrinter, _math_functions)

cxx_code_printers = {
    'c++98': CXX98CodePrinter,
    'c++11': CXX11CodePrinter,
    'c++17': CXX17CodePrinter
}
