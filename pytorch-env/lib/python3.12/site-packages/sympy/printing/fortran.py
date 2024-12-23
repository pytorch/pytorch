"""
Fortran code printer

The FCodePrinter converts single SymPy expressions into single Fortran
expressions, using the functions defined in the Fortran 77 standard where
possible. Some useful pointers to Fortran can be found on wikipedia:

https://en.wikipedia.org/wiki/Fortran

Most of the code below is based on the "Professional Programmer\'s Guide to
Fortran77" by Clive G. Page:

https://www.star.le.ac.uk/~cgp/prof77.html

Fortran is a case-insensitive language. This might cause trouble because
SymPy is case sensitive. So, fcode adds underscores to variable names when
it is necessary to make them different for Fortran.
"""

from __future__ import annotations
from typing import Any

from collections import defaultdict
from itertools import chain
import string

from sympy.codegen.ast import (
    Assignment, Declaration, Pointer, value_const,
    float32, float64, float80, complex64, complex128, int8, int16, int32,
    int64, intc, real, integer,  bool_, complex_, none, stderr, stdout
)
from sympy.codegen.fnodes import (
    allocatable, isign, dsign, cmplx, merge, literal_dp, elemental, pure,
    intent_in, intent_out, intent_inout
)
from sympy.core import S, Add, N, Float, Symbol
from sympy.core.function import Function
from sympy.core.numbers import equal_valued
from sympy.core.relational import Eq
from sympy.sets import Range
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.printer import printer_context

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import fcode, print_fcode # noqa:F401

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "abs",
    "conjugate": "conjg",
    "Max": "max",
    "Min": "min",
}


class FCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    type_aliases = {
        integer: int32,
        real: float64,
        complex_: complex128,
    }

    type_mappings = {
        intc: 'integer(c_int)',
        float32: 'real*4',  # real(kind(0.e0))
        float64: 'real*8',  # real(kind(0.d0))
        float80: 'real*10', # real(kind(????))
        complex64: 'complex*8',
        complex128: 'complex*16',
        int8: 'integer*1',
        int16: 'integer*2',
        int32: 'integer*4',
        int64: 'integer*8',
        bool_: 'logical'
    }

    type_modules = {
        intc: {'iso_c_binding': 'c_int'}
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'source_format': 'fixed',
        'contract': True,
        'standard': 77,
        'name_mangling': True,
    })

    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    _relationals = {
        '!=': '/=',
    }

    def __init__(self, settings=None):
        if not settings:
            settings = {}
        self.mangled_symbols = {}         # Dict showing mapping of all words
        self.used_name = []
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        # leading columns depend on fixed or free format
        standards = {66, 77, 90, 95, 2003, 2008}
        if self._settings['standard'] not in standards:
            raise ValueError("Unknown Fortran standard: %s" % self._settings[
                             'standard'])
        self.module_uses = defaultdict(set)  # e.g.: use iso_c_binding, only: c_int

    @property
    def _lead(self):
        if self._settings['source_format'] == 'fixed':
            return {'code': "      ", 'cont': "     @ ", 'comment': "C     "}
        elif self._settings['source_format'] == 'free':
            return {'code': "", 'cont': "      ", 'comment': "! "}
        else:
            raise ValueError("Unknown source format: %s" % self._settings['source_format'])

    def _print_Symbol(self, expr):
        if self._settings['name_mangling'] == True:
            if expr not in self.mangled_symbols:
                name = expr.name
                while name.lower() in self.used_name:
                    name += '_'
                self.used_name.append(name.lower())
                if name == expr.name:
                    self.mangled_symbols[expr] = expr
                else:
                    self.mangled_symbols[expr] = Symbol(name)

            expr = expr.xreplace(self.mangled_symbols)

        name = super()._print_Symbol(expr)
        return name

    def _rate_index_position(self, p):
        return -p*5

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {}".format(text)

    def _declare_number_const(self, name, value):
        return "parameter ({} = {})".format(name, self._print(value))

    def _print_NumberSymbol(self, expr):
        # A Number symbol that is not implemented here or with _printmethod
        # is registered and evaluated
        self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
        return str(expr)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # fortran arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("do %s = %s, %s" % (var, start, stop))
            close_lines.append("end do")
        return open_lines, close_lines

    def _print_sign(self, expr):
        from sympy.functions.elementary.complexes import Abs
        arg, = expr.args
        if arg.is_integer:
            new_expr = merge(0, isign(1, arg), Eq(arg, 0))
        elif (arg.is_complex or arg.is_infinite):
            new_expr = merge(cmplx(literal_dp(0), literal_dp(0)), arg/Abs(arg), Eq(Abs(arg), literal_dp(0)))
        else:
            new_expr = merge(literal_dp(0), dsign(literal_dp(1), arg), Eq(arg, literal_dp(0)))
        return self._print(new_expr)


    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) then" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("else if (%s) then" % self._print(c))
                lines.append(self._print(e))
            lines.append("end if")
            return "\n".join(lines)
        elif self._settings["standard"] >= 95:
            # Only supported in F95 and newer:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            pattern = "merge({T}, {F}, {COND})"
            code = self._print(expr.args[-1].expr)
            terms = list(expr.args[:-1])
            while terms:
                e, c = terms.pop()
                expr = self._print(e)
                cond = self._print(c)
                code = pattern.format(T=expr, F=code, COND=cond)
            return code
        else:
            # `merge` is not supported prior to F95
            raise NotImplementedError("Using Piecewise as an expression using "
                                      "inline operators is not supported in "
                                      "standards earlier than Fortran95.")

    def _print_MatrixElement(self, expr):
        return "{}({}, {})".format(self.parenthesize(expr.parent,
                PRECEDENCE["Atom"], strict=True), expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if pure_imaginary:
            if mixed:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_Mod(self, expr):
        # NOTE : Fortran has the functions mod() and modulo(). modulo() behaves
        # the same wrt to the sign of the arguments as Python and SymPy's
        # modulus computations (% and Mod()) but is not available in Fortran 66
        # or Fortran 77, thus we raise an error.
        if self._settings['standard'] in [66, 77]:
            msg = ("Python % operator and SymPy's Mod() function are not "
                   "supported by Fortran 66 or 77 standards.")
            raise NotImplementedError(msg)
        else:
            x, y = expr.args
            return "      modulo({}, {})".format(self._print(x), self._print(y))

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (
                self._print(literal_dp(1)),
                self.parenthesize(expr.base, PREC)
            )
        elif equal_valued(expr.exp, 0.5):
            if expr.base.is_integer:
                # Fortran intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "%d.0d0/%d.0d0" % (p, q)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        op = op if op not in self._relationals else self._relationals[op]
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{0} = {0} {1} {2}".format(
            self._print(lhs_code), self._print(expr.binop), self._print(rhs_code)))

    def _print_sum_(self, sm):
        params = self._print(sm.array)
        if sm.dim != None: # Must use '!= None', cannot use 'is not None'
            params += ', ' + self._print(sm.dim)
        if sm.mask != None: # Must use '!= None', cannot use 'is not None'
            params += ', mask=' + self._print(sm.mask)
        return '%s(%s)' % (sm.__class__.__name__.rstrip('_'), params)

    def _print_product_(self, prod):
        return self._print_sum_(prod)

    def _print_Do(self, do):
        excl = ['concurrent']
        if do.step == 1:
            excl.append('step')
            step = ''
        else:
            step = ', {step}'

        return (
            'do {concurrent}{counter} = {first}, {last}'+step+'\n'
            '{body}\n'
            'end do\n'
        ).format(
            concurrent='concurrent ' if do.concurrent else '',
            **do.kwargs(apply=lambda arg: self._print(arg), exclude=excl)
        )

    def _print_ImpliedDoLoop(self, idl):
        step = '' if idl.step == 1 else ', {step}'
        return ('({expr}, {counter} = {first}, {last}'+step+')').format(
            **idl.kwargs(apply=lambda arg: self._print(arg))
        )

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop - 1,
                        step=step, body=body)

    def _print_Type(self, type_):
        type_ = self.type_aliases.get(type_, type_)
        type_str = self.type_mappings.get(type_, type_.name)
        module_uses = self.type_modules.get(type_)
        if module_uses:
            for k, v in module_uses:
                self.module_uses[k].add(v)
        return type_str

    def _print_Element(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.symbol),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )

    def _print_Extent(self, ext):
        return str(ext)

    def _print_Declaration(self, expr):
        var = expr.variable
        val = var.value
        dim = var.attr_params('dimension')
        intents = [intent in var.attrs for intent in (intent_in, intent_out, intent_inout)]
        if intents.count(True) == 0:
            intent = ''
        elif intents.count(True) == 1:
            intent = ', intent(%s)' % ['in', 'out', 'inout'][intents.index(True)]
        else:
            raise ValueError("Multiple intents specified for %s" % self)

        if isinstance(var, Pointer):
            raise NotImplementedError("Pointers are not available by default in Fortran.")
        if self._settings["standard"] >= 90:
            result = '{t}{vc}{dim}{intent}{alloc} :: {s}'.format(
                t=self._print(var.type),
                vc=', parameter' if value_const in var.attrs else '',
                dim=', dimension(%s)' % ', '.join((self._print(arg) for arg in dim)) if dim else '',
                intent=intent,
                alloc=', allocatable' if allocatable in var.attrs else '',
                s=self._print(var.symbol)
            )
            if val != None: # Must be "!= None", cannot be "is not None"
                result += ' = %s' % self._print(val)
        else:
            if value_const in var.attrs or val:
                raise NotImplementedError("F77 init./parameter statem. req. multiple lines.")
            result = ' '.join((self._print(arg) for arg in [var.type, var.symbol]))

        return result


    def _print_Infinity(self, expr):
        return '(huge(%s) + 1)' % self._print(literal_dp(0))

    def _print_While(self, expr):
        return 'do while ({condition})\n{body}\nend do'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    def _print_BooleanTrue(self, expr):
        return '.true.'

    def _print_BooleanFalse(self, expr):
        return '.false.'

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append(self._lead['comment'] + line[1:].lstrip())
            else:
                result.append(self._lead['code'] + line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos
        # split line by line and add the split lines to result
        result = []
        if self._settings['source_format'] == 'free':
            trailing = ' &'
        else:
            trailing = ''
        for line in lines:
            if line.startswith(self._lead['comment']):
                # comment line
                if len(line) > 72:
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while line:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % (self._lead['comment'], hunk))
                else:
                    result.append(line)
            elif line.startswith(self._lead['code']):
                # code line
                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while line:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s" % (self._lead['cont'], hunk))
            else:
                result.append(line)
        return result

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        free = self._settings['source_format'] == 'free'
        code = [ line.lstrip(' \t') for line in code ]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n', 'else', 'program', 'interface')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif', 'else', 'end program', 'end interface')

        increase = [ int(any(map(line.startswith, inc_keyword)))
                     for line in code ]
        decrease = [ int(any(map(line.startswith, dec_keyword)))
                     for line in code ]
        continuation = [ int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code ]

        level = 0
        cont_padding = 0
        tabwidth = 3
        new_code = []
        for i, line in enumerate(code):
            if line in ('', '\n'):
                new_code.append(line)
                continue
            level -= decrease[i]

            if free:
                padding = " "*(level*tabwidth + cont_padding)
            else:
                padding = " "*level*tabwidth

            line = "%s%s" % (padding, line)
            if not free:
                line = self._pad_leading_columns([line])[0]

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        if not free:
            return self._wrap_fortran(new_code)
        return new_code

    def _print_GoTo(self, goto):
        if goto.expr:  # computed goto
            return "go to ({labels}), {expr}".format(
                labels=', '.join((self._print(arg) for arg in goto.labels)),
                expr=self._print(goto.expr)
            )
        else:
            lbl, = goto.labels
            return "go to %s" % self._print(lbl)

    def _print_Program(self, prog):
        return (
            "program {name}\n"
            "{body}\n"
            "end program\n"
        ).format(**prog.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Module(self, mod):
        return (
            "module {name}\n"
            "{declarations}\n"
            "\ncontains\n\n"
            "{definitions}\n"
            "end module\n"
        ).format(**mod.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Stream(self, strm):
        if strm.name == 'stdout' and self._settings["standard"] >= 2003:
            self.module_uses['iso_c_binding'].add('stdint=>input_unit')
            return 'input_unit'
        elif strm.name == 'stderr' and self._settings["standard"] >= 2003:
            self.module_uses['iso_c_binding'].add('stdint=>error_unit')
            return 'error_unit'
        else:
            if strm.name == 'stdout':
                return '*'
            else:
                return strm.name

    def _print_Print(self, ps):
        if ps.format_string == none: # Must be '!= None', cannot be 'is not None'
            template = "print {fmt}, {iolist}"
            fmt = '*'
        else:
            template = 'write(%(out)s, fmt="{fmt}", advance="no"), {iolist}' % {
                'out': {stderr: '0', stdout: '6'}.get(ps.file, '*')
            }
            fmt = self._print(ps.format_string)
        return template.format(fmt=fmt, iolist=', '.join(
            (self._print(arg) for arg in ps.print_args)))

    def _print_Return(self, rs):
        arg, = rs.args
        return "{result_name} = {arg}".format(
            result_name=self._context.get('result_name', 'sympy_result'),
            arg=self._print(arg)
        )

    def _print_FortranReturn(self, frs):
        arg, = frs.args
        if arg:
            return 'return %s' % self._print(arg)
        else:
            return 'return'

    def _head(self, entity, fp, **kwargs):
        bind_C_params = fp.attr_params('bind_C')
        if bind_C_params is None:
            bind = ''
        else:
            bind = ' bind(C, name="%s")' % bind_C_params[0] if bind_C_params else ' bind(C)'
        result_name = self._settings.get('result_name', None)
        return (
            "{entity}{name}({arg_names}){result}{bind}\n"
            "{arg_declarations}"
        ).format(
            entity=entity,
            name=self._print(fp.name),
            arg_names=', '.join([self._print(arg.symbol) for arg in fp.parameters]),
            result=(' result(%s)' % result_name) if result_name else '',
            bind=bind,
            arg_declarations='\n'.join((self._print(Declaration(arg)) for arg in fp.parameters))
        )

    def _print_FunctionPrototype(self, fp):
        entity = "{} function ".format(self._print(fp.return_type))
        return (
            "interface\n"
            "{function_head}\n"
            "end function\n"
            "end interface"
        ).format(function_head=self._head(entity, fp))

    def _print_FunctionDefinition(self, fd):
        if elemental in fd.attrs:
            prefix = 'elemental '
        elif pure in fd.attrs:
            prefix = 'pure '
        else:
            prefix = ''

        entity = "{} function ".format(self._print(fd.return_type))
        with printer_context(self, result_name=fd.name):
            return (
                "{prefix}{function_head}\n"
                "{body}\n"
                "end function\n"
            ).format(
                prefix=prefix,
                function_head=self._head(entity, fd),
                body=self._print(fd.body)
            )

    def _print_Subroutine(self, sub):
        return (
            '{subroutine_head}\n'
            '{body}\n'
            'end subroutine\n'
        ).format(
            subroutine_head=self._head('subroutine ', sub),
            body=self._print(sub.body)
        )

    def _print_SubroutineCall(self, scall):
        return 'call {name}({args})'.format(
            name=self._print(scall.name),
            args=', '.join((self._print(arg) for arg in scall.subroutine_args))
        )

    def _print_use_rename(self, rnm):
        return "%s => %s" % tuple((self._print(arg) for arg in rnm.args))

    def _print_use(self, use):
        result = 'use %s' % self._print(use.namespace)
        if use.rename != None: # Must be '!= None', cannot be 'is not None'
            result += ', ' + ', '.join([self._print(rnm) for rnm in use.rename])
        if use.only != None: # Must be '!= None', cannot be 'is not None'
            result += ', only: ' + ', '.join([self._print(nly) for nly in use.only])
        return result

    def _print_BreakToken(self, _):
        return 'exit'

    def _print_ContinueToken(self, _):
        return 'cycle'

    def _print_ArrayConstructor(self, ac):
        fmtstr = "[%s]" if self._settings["standard"] >= 2003 else '(/%s/)'
        return fmtstr % ', '.join((self._print(arg) for arg in ac.elements))

    def _print_ArrayElement(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.name),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )
