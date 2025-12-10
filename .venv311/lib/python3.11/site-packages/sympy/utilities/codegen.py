"""
module for generating C, C++, Fortran77, Fortran90, Julia, Rust
and Octave/Matlab routines that evaluate SymPy expressions.
This module is work in progress.
Only the milestones with a '+' character in the list below have been completed.

--- How is sympy.utilities.codegen different from sympy.printing.ccode? ---

We considered the idea to extend the printing routines for SymPy functions in
such a way that it prints complete compilable code, but this leads to a few
unsurmountable issues that can only be tackled with dedicated code generator:

- For C, one needs both a code and a header file, while the printing routines
  generate just one string. This code generator can be extended to support
  .pyf files for f2py.

- SymPy functions are not concerned with programming-technical issues, such
  as input, output and input-output arguments. Other examples are contiguous
  or non-contiguous arrays, including headers of other libraries such as gsl
  or others.

- It is highly interesting to evaluate several SymPy functions in one C
  routine, eventually sharing common intermediate results with the help
  of the cse routine. This is more than just printing.

- From the programming perspective, expressions with constants should be
  evaluated in the code generator as much as possible. This is different
  for printing.

--- Basic assumptions ---

* A generic Routine data structure describes the routine that must be
  translated into C/Fortran/... code. This data structure covers all
  features present in one or more of the supported languages.

* Descendants from the CodeGen class transform multiple Routine instances
  into compilable code. Each derived class translates into a specific
  language.

* In many cases, one wants a simple workflow. The friendly functions in the
  last part are a simple api on top of the Routine/CodeGen stuff. They are
  easier to use, but are less powerful.

--- Milestones ---

+ First working version with scalar input arguments, generating C code,
  tests
+ Friendly functions that are easier to use than the rigorous
  Routine/CodeGen workflow.
+ Integer and Real numbers as input and output
+ Output arguments
+ InputOutput arguments
+ Sort input/output arguments properly
+ Contiguous array arguments (numpy matrices)
+ Also generate .pyf code for f2py (in autowrap module)
+ Isolate constants and evaluate them beforehand in double precision
+ Fortran 90
+ Octave/Matlab

- Common Subexpression Elimination
- User defined comments in the generated code
- Optional extra include lines for libraries/objects that can eval special
  functions
- Test other C compilers and libraries: gcc, tcc, libtcc, gcc+gsl, ...
- Contiguous array arguments (SymPy matrices)
- Non-contiguous array arguments (SymPy matrices)
- ccode must raise an error when it encounters something that cannot be
  translated into c. ccode(integrate(sin(x)/x, x)) does not make sense.
- Complex numbers as input and output
- A default complex datatype
- Include extra information in the header: date, user, hostname, sha1
  hash, ...
- Fortran 77
- C++
- Python
- Julia
- Rust
- ...

"""

import os
import textwrap
from io import StringIO

from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)
from sympy.utilities.iterables import is_sequence


__all__ = [
    # description of routines
    "Routine", "DataType", "default_datatypes", "get_default_datatype",
    "Argument", "InputArgument", "OutputArgument", "Result",
    # routines -> code
    "CodeGen", "CCodeGen", "FCodeGen", "JuliaCodeGen", "OctaveCodeGen",
    "RustCodeGen",
    # friendly functions
    "codegen", "make_routine",
]


#
# Description of routines
#


class Routine:
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """

    def __init__(self, name, arguments, results, local_vars, global_vars):
        """Initialize a Routine instance.

        Parameters
        ==========

        name : string
            Name of the routine.

        arguments : list of Arguments
            These are things that appear in arguments of a routine, often
            appearing on the right-hand side of a function call.  These are
            commonly InputArguments but in some languages, they can also be
            OutputArguments or InOutArguments (e.g., pass-by-reference in C
            code).

        results : list of Results
            These are the return values of the routine, often appearing on
            the left-hand side of a function call.  The difference between
            Results and OutputArguments and when you should use each is
            language-specific.

        local_vars : list of Results
            These are variables that will be defined at the beginning of the
            function.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set()
        symbols = set()
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            elif isinstance(arg, InputArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)
            symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))

        local_symbols = set()
        for r in local_vars:
            if isinstance(r, Result):
                symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))
                local_symbols.add(r.name)
            else:
                local_symbols.add(r)

        symbols = {s.label if isinstance(s, Idx) else s for s in symbols}

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        notcovered = symbols.difference(
            input_symbols.union(local_symbols).union(global_vars))
        if notcovered != set():
            raise ValueError("Symbols needed for output are not in input " +
                             ", ".join([str(x) for x in notcovered]))

        self.name = name
        self.arguments = arguments
        self.results = results
        self.local_vars = local_vars
        self.global_vars = global_vars

    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {results}, {local_vars}, {global_vars})".format(**self.__dict__)

    __repr__ = __str__

    @property
    def variables(self):
        """Returns a set of all variables possibly used in the routine.

        For routines with unnamed return values, the dummies that may or
        may not be used will be included in the set.

        """
        v = set(self.local_vars)
        v.update(arg.name for arg in self.arguments)
        v.update(res.result_var for res in self.results)
        return v

    @property
    def result_variables(self):
        """Returns a list of OutputArgument, InOutArgument and Result.

        If return values are present, they are at the end of the list.
        """
        args = [arg for arg in self.arguments if isinstance(
            arg, (OutputArgument, InOutArgument))]
        args.extend(self.results)
        return args


class DataType:
    """Holds strings for a certain datatype in different languages."""
    def __init__(self, cname, fname, pyname, jlname, octname, rsname):
        self.cname = cname
        self.fname = fname
        self.pyname = pyname
        self.jlname = jlname
        self.octname = octname
        self.rsname = rsname


default_datatypes = {
    "int": DataType("int", "INTEGER*4", "int", "", "", "i32"),
    "float": DataType("double", "REAL*8", "float", "", "", "f64"),
    "complex": DataType("double", "COMPLEX*16", "complex", "", "", "float") #FIXME:
       # complex is only supported in fortran, python, julia, and octave.
       # So to not break c or rust code generation, we stick with double or
       # float, respectively (but actually should raise an exception for
       # explicitly complex variables (x.is_complex==True))
}


COMPLEX_ALLOWED = False
def get_default_datatype(expr, complex_allowed=None):
    """Derives an appropriate datatype based on the expression."""
    if complex_allowed is None:
        complex_allowed = COMPLEX_ALLOWED
    if complex_allowed:
        final_dtype = "complex"
    else:
        final_dtype = "float"
    if expr.is_integer:
        return default_datatypes["int"]
    elif expr.is_real:
        return default_datatypes["float"]
    elif isinstance(expr, MatrixBase):
        #check all entries
        dt = "int"
        for element in expr:
            if dt == "int" and not element.is_integer:
                dt = "float"
            if dt == "float" and not element.is_real:
                return default_datatypes[final_dtype]
        return default_datatypes[dt]
    else:
        return default_datatypes[final_dtype]


class Variable:
    """Represents a typed variable."""

    def __init__(self, name, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol or MatrixSymbol

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimensions : sequence containing tuples, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        if not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("The first argument must be a SymPy symbol.")
        if datatype is None:
            datatype = get_default_datatype(name)
        elif not isinstance(datatype, DataType):
            raise TypeError("The (optional) `datatype' argument must be an "
                            "instance of the DataType class.")
        if dimensions and not isinstance(dimensions, (tuple, list)):
            raise TypeError(
                "The dimensions argument must be a sequence of tuples")

        self._name = name
        self._datatype = {
            'C': datatype.cname,
            'FORTRAN': datatype.fname,
            'JULIA': datatype.jlname,
            'OCTAVE': datatype.octname,
            'PYTHON': datatype.pyname,
            'RUST': datatype.rsname,
        }
        self.dimensions = dimensions
        self.precision = precision

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)

    __repr__ = __str__

    @property
    def name(self):
        return self._name

    def get_datatype(self, language):
        """Returns the datatype string for the requested language.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.utilities.codegen import Variable
        >>> x = Variable(Symbol('x'))
        >>> x.get_datatype('c')
        'double'
        >>> x.get_datatype('fortran')
        'REAL*8'

        """
        try:
            return self._datatype[language.upper()]
        except KeyError:
            raise CodeGenError("Has datatypes for languages: %s" %
                    ", ".join(self._datatype))


class Argument(Variable):
    """An abstract Argument data structure: a name and a data type.

    This structure is refined in the descendants below.

    """
    pass


class InputArgument(Argument):
    pass


class ResultBase:
    """Base class for all "outgoing" information from a routine.

    Objects of this class stores a SymPy expression, and a SymPy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """
    def __init__(self, expr, result_var):
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
            self.result_var)

    __repr__ = __str__


class OutputArgument(Argument, ResultBase):
    """OutputArgument are always initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol, MatrixSymbol
            The name of this variable.  When used for code generation, this
            might appear, for example, in the prototype of function in the
            argument list.

        result_var : Symbol, Indexed
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".

        expr : object
            The expression that should be output, typically a SymPy
            expression.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimensions : sequence containing tuples, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """

        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.result_var, self.expr)

    __repr__ = __str__


class InOutArgument(Argument, ResultBase):
    """InOutArgument are never initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        if not datatype:
            datatype = get_default_datatype(expr)
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)
    __init__.__doc__ = OutputArgument.__init__.__doc__


    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__


class Result(Variable, ResultBase):
    """An expression for a return value.

    The name result is used to avoid conflicts with the reserved word
    "return" in the Python language.  It is also shorter than ReturnValue.

    These may or may not need a name in the destination (e.g., "return(x*y)"
    might return a value without ever naming it).

    """

    def __init__(self, expr, name=None, result_var=None, datatype=None,
                 dimensions=None, precision=None):
        """Initialize a return value.

        Parameters
        ==========

        expr : SymPy expression

        name : Symbol, MatrixSymbol, optional
            The name of this return variable.  When used for code generation,
            this might appear, for example, in the prototype of function in a
            list of return values.  A dummy name is generated if omitted.

        result_var : Symbol, Indexed, optional
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".  Defaults to
            `name` if omitted.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the expr argument.

        dimensions : sequence containing tuples, optional
            If present, this variable is interpreted as an array,
            where this sequence of tuples specifies (lower, upper)
            bounds for each index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        # Basic because it is the base class for all types of expressions
        if not isinstance(expr, (Basic, MatrixBase)):
            raise TypeError("The first argument must be a SymPy expression.")

        if name is None:
            name = 'result_%d' % abs(hash(expr))

        if datatype is None:
            #try to infer data type from the expression
            datatype = get_default_datatype(expr)

        if isinstance(name, str):
            if isinstance(expr, (MatrixBase, MatrixExpr)):
                name = MatrixSymbol(name, *expr.shape)
            else:
                name = Symbol(name)

        if result_var is None:
            result_var = name

        Variable.__init__(self, name, datatype=datatype,
                          dimensions=dimensions, precision=precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.expr, self.name,
            self.result_var)

    __repr__ = __str__


#
# Transformation of routine objects into code
#

class CodeGen:
    """Abstract class for the code generators."""

    printer = None  # will be set to an instance of a CodePrinter subclass

    def _indent_code(self, codelines):
        return self.printer.indent_code(codelines)

    def _printer_method_with_settings(self, method, settings=None, *args, **kwargs):
        settings = settings or {}
        ori = {k: self.printer._settings[k] for k in settings}
        for k, v in settings.items():
            self.printer._settings[k] = v
        result = getattr(self.printer, method)(*args, **kwargs)
        for k, v in ori.items():
            self.printer._settings[k] = v
        return result

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        if self.printer._settings['human']:
            expr_str = self.printer.doprint(s)
        else:
            constants, not_supported, expr_str = self.printer.doprint(s)
            if constants or not_supported:
                raise ValueError("Failed to print %s" % str(s))
        return expr_str.strip()

    def __init__(self, project="project", cse=False):
        """Initialize a code generator.

        Derived classes will offer more options that affect the generated
        code.

        """
        self.project = project
        self.cse = cse

    def routine(self, name, expr, argument_sequence=None, global_vars=None):
        """Creates an Routine object that is appropriate for this language.

        This implementation is appropriate for at least C/Fortran.  Subclasses
        can override this if necessary.

        Here, we assume at most one return value (the l-value) which must be
        scalar.  Additional outputs are OutputArguments (e.g., pointers on
        right-hand-side or pass-by-reference).  Matrices are always returned
        via OutputArguments.  If ``argument_sequence`` is None, arguments will
        be ordered alphabetically, but with all InputArguments first, and then
        OutputArgument and InOutArguments.

        """

        if self.cse:
            from sympy.simplify.cse_main import cse

            if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
                if not expr:
                    raise ValueError("No expression given")
                for e in expr:
                    if not e.is_Equality:
                        raise CodeGenError("Lists of expressions must all be Equalities. {} is not.".format(e))

                # create a list of right hand sides and simplify them
                rhs = [e.rhs for e in expr]
                common, simplified = cse(rhs)

                # pack the simplified expressions back up with their left hand sides
                expr = [Equality(e.lhs, rhs) for e, rhs in zip(expr, simplified)]
            else:
                if isinstance(expr, Equality):
                    common, simplified = cse(expr.rhs) #, ignore=in_out_args)
                    expr = Equality(expr.lhs, simplified[0])
                else:
                    common, simplified = cse(expr)
                    expr = simplified

            local_vars = [Result(b,a) for a,b in common]
            local_symbols = {a for a,_ in common}
            local_expressions = Tuple(*[b for _,b in common])
        else:
            local_expressions = Tuple()

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        if self.cse:
            if {i.label for i in expressions.atoms(Idx)} != set():
                raise CodeGenError("CSE and Indexed expressions do not play well together yet")
        else:
            # local variables for indexed expressions
            local_vars = {i.label for i in expressions.atoms(Idx)}
            local_symbols = local_vars

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = (expressions.free_symbols | local_expressions.free_symbols) - local_symbols - global_vars
        new_symbols = set()
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                new_symbols.update(symbol.args[1].free_symbols)
            if isinstance(symbol, Indexed):
                new_symbols.remove(symbol)
        symbols = new_symbols

        # Decide whether to use output argument or return value
        return_val = []
        output_args = []
        for expr in expressions:
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                if expr.has(symbol):
                    output_args.append(
                        InOutArgument(symbol, out_arg, expr, dimensions=dims))
                else:
                    output_args.append(
                        OutputArgument(symbol, out_arg, expr, dimensions=dims))

                # remove duplicate arguments when they are not local variables
                if symbol not in local_vars:
                    # avoid duplicate arguments
                    symbols.remove(symbol)
            elif isinstance(expr, (ImmutableMatrix, MatrixSlice)):
                # Create a "dummy" MatrixSymbol to use as the Output arg
                out_arg = MatrixSymbol('out_%s' % abs(hash(expr)), *expr.shape)
                dims = tuple([(S.Zero, dim - 1) for dim in out_arg.shape])
                output_args.append(
                    OutputArgument(out_arg, out_arg, expr, dimensions=dims))
            else:
                return_val.append(Result(expr))

        arg_list = []

        # setup input argument list

        # helper to get dimensions for data for array-like args
        def dimensions(s):
            return [(S.Zero, dim - 1) for dim in s.shape]

        array_symbols = {}
        for array in expressions.atoms(Indexed) | local_expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol) | local_expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                array = array_symbols[symbol]
                metadata = {'dimensions': dimensions(array)}
            else:
                metadata = {}

            arg_list.append(InputArgument(symbol, **metadata))

        output_args.sort(key=lambda x: str(x.name))
        arg_list.extend(output_args)

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    if isinstance(symbol, (IndexedBase, MatrixSymbol)):
                        metadata = {'dimensions': dimensions(symbol)}
                    else:
                        metadata = {}
                    new_args.append(InputArgument(symbol, **metadata))
            arg_list = new_args

        return Routine(name, arg_list, return_val, local_vars, global_vars)

    def write(self, routines, prefix, to_files=False, header=True, empty=True):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        """
        if to_files:
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                with open(filename, "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                contents = StringIO()
                dump_fn(self, routines, contents, prefix, header, empty)
                result.append((filename, contents.getvalue()))
            return result

    def dump_code(self, routines, f, prefix, header=True, empty=True):
        """Write the code by calling language specific methods.

        The generated file contains all the definitions of the routines in
        low-level code and refers to the header file if appropriate.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """

        code_lines = self._preprocessor_statements(prefix)

        for routine in routines:
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if header:
            code_lines = ''.join(self._get_header() + [code_lines])

        if code_lines:
            f.write(code_lines)


class CodeGenError(Exception):
    pass


class CodeGenArgumentListError(Exception):
    @property
    def missing_args(self):
        return self.args[1]


header_comment = """Code generated with SymPy %(version)s

See http://www.sympy.org/ for more information.

This file is part of '%(project)s'
"""


class CCodeGen(CodeGen):
    """Generator for C code.

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.c and <prefix>.h respectively.

    """

    code_extension = "c"
    interface_extension = "h"
    standard = 'c99'

    def __init__(self, project="project", printer=None,
                 preprocessor_statements=None, cse=False):
        super().__init__(project=project, cse=cse)
        self.printer = printer or c_code_printers[self.standard.lower()]()

        self.preprocessor_statements = preprocessor_statements
        if preprocessor_statements is None:
            self.preprocessor_statements = ['#include <math.h>']

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append(" *%s*\n" % line.center(76))
        code_lines.append(" " + "*"*78 + "/\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        if len(routine.results) > 1:
            raise CodeGenError("C only supports a single or no return value.")
        elif len(routine.results) == 1:
            ctype = routine.results[0].get_datatype('C')
        else:
            ctype = "void"

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append((arg.get_datatype('C'), "*%s" % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        arguments = ", ".join([ "%s %s" % t for t in type_args])
        return "%s %s(%s)" % (ctype, routine.name, arguments)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.append('#include "{}.h"'.format(os.path.basename(prefix)))
        code_lines.extend(self.preprocessor_statements)
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        code_lines = []
        for result in routine.local_vars:

            # local variables that are simple symbols such as those used as indices into
            # for loops are defined declared elsewhere.
            if not isinstance(result, Result):
                continue

            if result.name != result.result_var:
                raise CodeGen("Result variable and name should match: {}".format(result))
            assign_to = result.name
            t = result.get_datatype('c')
            if isinstance(result.expr, (MatrixBase, MatrixExpr)):
                dims = result.expr.shape
                code_lines.append("{} {}[{}];\n".format(t, str(assign_to), dims[0]*dims[1]))
                prefix = ""
            else:
                prefix = "const {} ".format(t)

            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "dereference": dereference, "strict": False},
                result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))

            code_lines.append("{}{}\n".format(prefix, c_expr))

        return code_lines

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + "_result"
                t = result.get_datatype('c')
                code_lines.append("{} {};\n".format(t, str(assign_to)))
                return_val = assign_to
            else:
                assign_to = result.result_var

            try:
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_c(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_c.extension = code_extension  # type: ignore
    dump_c.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the C header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to construct the include guards.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        guard_name = "%s__%s__H" % (self.project.replace(
            " ", "_").upper(), prefix.replace("/", "_").upper())
        # include guards
        if empty:
            print(file=f)
        print("#ifndef %s" % guard_name, file=f)
        print("#define %s" % guard_name, file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_prototype(routine)
            print("%s;" % prototype, file=f)
        # end if include guards
        if empty:
            print(file=f)
        print("#endif", file=f)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension  # type: ignore

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_c, dump_h]

class C89CodeGen(CCodeGen):
    standard = 'C89'

class C99CodeGen(CCodeGen):
    standard = 'C99'

class FCodeGen(CodeGen):
    """Generator for Fortran 95 code

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.f90 and <prefix>.h respectively.

    """

    code_extension = "f90"
    interface_extension = "h"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or FCodePrinter()

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("!" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            code_lines.append("!*%s*\n" % line.center(76))
        code_lines.append("!" + "*"*78 + '\n')
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the fortran routine."""
        code_list = []
        if len(routine.results) > 1:
            raise CodeGenError(
                "Fortran only supports a single or no return value.")
        elif len(routine.results) == 1:
            result = routine.results[0]
            code_list.append(result.get_datatype('fortran'))
            code_list.append("function")
        else:
            code_list.append("subroutine")

        args = ", ".join("%s" % self._get_symbol(arg.name)
                        for arg in routine.arguments)

        call_sig = "{}({})\n".format(routine.name, args)
        # Fortran 95 requires all lines be less than 132 characters, so wrap
        # this line before appending.
        call_sig = ' &\n'.join(textwrap.wrap(call_sig,
                                             width=60,
                                             break_long_words=False)) + '\n'
        code_list.append(call_sig)
        code_list = [' '.join(code_list)]
        code_list.append('implicit none\n')
        return code_list

    def _declare_arguments(self, routine):
        # argument type declarations
        code_list = []
        array_list = []
        scalar_list = []
        for arg in routine.arguments:

            if isinstance(arg, InputArgument):
                typeinfo = "%s, intent(in)" % arg.get_datatype('fortran')
            elif isinstance(arg, InOutArgument):
                typeinfo = "%s, intent(inout)" % arg.get_datatype('fortran')
            elif isinstance(arg, OutputArgument):
                typeinfo = "%s, intent(out)" % arg.get_datatype('fortran')
            else:
                raise CodeGenError("Unknown Argument type: %s" % type(arg))

            fprint = self._get_symbol

            if arg.dimensions:
                # fortran arrays start at 1
                dimstr = ", ".join(["%s:%s" % (
                    fprint(dim[0] + 1), fprint(dim[1] + 1))
                    for dim in arg.dimensions])
                typeinfo += ", dimension(%s)" % dimstr
                array_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))
            else:
                scalar_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))

        # scalars first, because they can be used in array declarations
        code_list.extend(scalar_list)
        code_list.extend(array_list)

        return code_list

    def _declare_globals(self, routine):
        # Global variables not explicitly declared within Fortran 90 functions.
        # Note: a future F77 mode may need to generate "common" blocks.
        return []

    def _declare_locals(self, routine):
        code_list = []
        for var in sorted(routine.local_vars, key=str):
            typeinfo = get_default_datatype(var)
            code_list.append("%s :: %s\n" % (
                typeinfo.fname, self._get_symbol(var)))
        return code_list

    def _get_routine_ending(self, routine):
        """Returns the closing statements of the fortran routine."""
        if len(routine.results) == 1:
            return ["end function\n"]
        else:
            return ["end subroutine\n"]

    def get_interface(self, routine):
        """Returns a string for the function interface.

        The routine should have a single result object, which can be None.
        If the routine has multiple result objects, a CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        prototype = [ "interface\n" ]
        prototype.extend(self._get_routine_opening(routine))
        prototype.extend(self._declare_arguments(routine))
        prototype.extend(self._get_routine_ending(routine))
        prototype.append("end interface\n")

        return "".join(prototype)

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name
            elif isinstance(result, (OutputArgument, InOutArgument)):
                assign_to = result.result_var

            constants, not_fortran, f_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "source_format": 'free', "standard": 95, "strict": False},
                result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                t = get_default_datatype(obj)
                declarations.append(
                    "%s, parameter :: %s = %s\n" % (t.fname, obj, v))
            for obj in sorted(not_fortran, key=str):
                t = get_default_datatype(obj)
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("%s :: %s\n" % (t.fname, name))

            code_lines.append("%s\n" % f_expr)
        return declarations + code_lines

    def _indent_code(self, codelines):
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "source_format": 'free', "strict": False}, codelines)

    def dump_f95(self, routines, f, prefix, header=True, empty=True):
        # check that symbols are unique with ignorecase
        for r in routines:
            lowercase = {str(x).lower() for x in r.variables}
            orig_case = {str(x) for x in r.variables}
            if len(lowercase) < len(orig_case):
                raise CodeGenError("Fortran ignores case. Got symbols: %s" %
                        (", ".join([str(var) for var in r.variables])))
        self.dump_code(routines, f, prefix, header, empty)
    dump_f95.extension = code_extension  # type: ignore
    dump_f95.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the interface to a header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_interface(routine)
            f.write(prototype)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension  # type: ignore

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_f95, dump_h]


class JuliaCodeGen(CodeGen):
    """Generator for Julia code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.jl.

    """

    code_extension = "jl"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or JuliaCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Julia."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)

        # Julia supports multiple return values
        return_vals = []
        output_args = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("#\n")
            else:
                code_lines.append("#   %s\n" % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("function ")

        # Inputs
        args = []
        for arg in routine.arguments:
            if isinstance(arg, OutputArgument):
                raise CodeGenError("Julia: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, (InputArgument, InOutArgument)):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y[i]` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        return ["return " + ", ".join(outs) + "\nend\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, jl_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "%s = %s\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "# unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (jl_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        # Note that indenting seems to happen twice, first
        # statement-by-statement by JuliaPrinter then again here.
        p = JuliaCodePrinter({'human': False, "strict": False})
        return p.indent_code(codelines)

    def dump_jl(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_jl.extension = code_extension  # type: ignore
    dump_jl.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_jl]


class OctaveCodeGen(CodeGen):
    """Generator for Octave code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.m.

    Octave .m files usually contain one function.  That function name should
    match the filename (``prefix``).  If you pass multiple ``name_expr`` pairs,
    the latter ones are presumed to be private functions accessed by the
    primary function.

    You should only pass inputs to ``argument_sequence``: outputs are ordered
    according to their order in ``name_expr``.

    """

    code_extension = "m"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or OctaveCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Octave."""

        # FIXME: this is probably general enough for other high-level
        # languages, perhaps its the C/Fortran one that is specialized!

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)

        # Octave supports multiple return values
        return_vals = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    symbol = out_arg.base.label
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        arg_list = []
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("%\n")
            else:
                code_lines.append("%%   %s\n" % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("function ")

        # Outputs
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y(i)` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        if len(outs) > 1:
            code_list.append("[" + (", ".join(outs)) + "]")
        else:
            code_list.append("".join(outs))
        code_list.append(" = ")

        # Inputs
        args = []
        for arg in routine.arguments:
            if isinstance(arg, (OutputArgument, InOutArgument)):
                raise CodeGenError("Octave: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, InputArgument):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        if not routine.global_vars:
            return []
        s = " ".join(sorted([self._get_symbol(g) for g in routine.global_vars]))
        return ["global " + s + "\n"]

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        return ["end\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, oct_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "  %s = %s;  %% constant\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "  %% unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (oct_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "strict": False}, codelines)

    def dump_m(self, routines, f, prefix, header=True, empty=True, inline=True):
        # Note used to call self.dump_code() but we need more control for header

        code_lines = self._preprocessor_statements(prefix)

        for i, routine in enumerate(routines):
            if i > 0:
                if empty:
                    code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            if i == 0:
                if routine.name != prefix:
                    raise ValueError('Octave function name should match prefix')
                if header:
                    code_lines.append("%" + prefix.upper() +
                                      "  Autogenerated by SymPy\n")
                    code_lines.append(''.join(self._get_header()))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if code_lines:
            f.write(code_lines)

    dump_m.extension = code_extension  # type: ignore
    dump_m.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_m]

class RustCodeGen(CodeGen):
    """Generator for Rust code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.rs

    """

    code_extension = "rs"

    def __init__(self, project="project", printer=None):
        super().__init__(project=project)
        self.printer = printer or RustCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Rust."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = expressions.free_symbols - local_vars - global_vars - expressions.atoms(Indexed)

        # Rust supports multiple return values
        return_vals = []
        output_args = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)


    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/*\n")
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append((" *%s" % line.center(76)).rstrip() + "\n")
        code_lines.append(" */\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        results = [i.get_datatype('Rust') for i in routine.results]

        if len(results) == 1:
            rstype = " -> " + results[0]
        elif len(routine.results) > 1:
            rstype = " -> (" + ", ".join(results) + ")"
        else:
            rstype = ""

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append(("*%s" % name, arg.get_datatype('Rust')))
            else:
                type_args.append((name, arg.get_datatype('Rust')))
        arguments = ", ".join([ "%s: %s" % t for t in type_args])
        return "fn %s(%s)%s" % (routine.name, arguments, rstype)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        # code_lines.append("use std::f64::consts::*;\n")
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):
        # loop variables are declared in loop statement
        return []

    def _call_printer(self, routine):

        code_lines = []
        declarations = []
        returns = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
                returns.append(str(result.result_var))
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, rs_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                declarations.append("const %s: f64 = %s;\n" % (name, value))

            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("// unsupported: %s\n" % (name))

            code_lines.append("let %s\n" % rs_expr)

        if len(returns) > 1:
            returns = ['(' + ', '.join(returns) + ')']

        returns.append('\n')

        return declarations + code_lines + returns

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_rs(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_rs.extension = code_extension  # type: ignore
    dump_rs.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_rs]




def get_code_generator(language, project=None, standard=None, printer = None):
    if language == 'C':
        if standard is None:
            pass
        elif standard.lower() == 'c89':
            language = 'C89'
        elif standard.lower() == 'c99':
            language = 'C99'
    CodeGenClass = {"C": CCodeGen, "C89": C89CodeGen, "C99": C99CodeGen,
                    "F95": FCodeGen, "JULIA": JuliaCodeGen,
                    "OCTAVE": OctaveCodeGen,
                    "RUST": RustCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project, printer)


#
# Friendly functions
#


def codegen(name_expr, language=None, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, standard=None, code_gen=None, printer=None):
    """Generate source code for expressions in a given language.

    Parameters
    ==========

    name_expr : tuple, or list of tuples
        A single (name, expression) tuple or a list of (name, expression)
        tuples.  Each tuple corresponds to a routine.  If the expression is
        an equality (an instance of class Equality) the left hand side is
        considered an output argument.  If expression is an iterable, then
        the routine will have multiple outputs.

    language : string,
        A string that indicates the source code language.  This is case
        insensitive.  Currently, 'C', 'F95' and 'Octave' are supported.
        'Octave' generates code compatible with both Octave and Matlab.

    prefix : string, optional
        A prefix for the names of the files that contain the source code.
        Language-dependent suffixes will be appended.  If omitted, the name
        of the first name_expr tuple is used.

    project : string, optional
        A project name, used for making unique preprocessor instructions.
        [default: "project"]

    to_files : bool, optional
        When True, the code will be written to one or more files with the
        given prefix, otherwise strings with the names and contents of
        these files are returned. [default: False]

    header : bool, optional
        When True, a header is written on top of each source file.
        [default: True]

    empty : bool, optional
        When True, empty lines are used to structure the code.
        [default: True]

    argument_sequence : iterable, optional
        Sequence of arguments for the routine in a preferred order.  A
        CodeGenError is raised if required arguments are missing.
        Redundant arguments are used without warning.  If omitted,
        arguments will be ordered alphabetically, but with all input
        arguments first, and then output or in-out arguments.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    standard : string, optional

    code_gen : CodeGen instance, optional
        An instance of a CodeGen subclass. Overrides ``language``.

    printer : Printer instance, optional
        An instance of a Printer subclass.

    Examples
    ========

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...     ("f", x+y*z), "C89", "test", header=False, empty=False)
    >>> print(c_name)
    test.c
    >>> print(c_code)
    #include "test.h"
    #include <math.h>
    double f(double x, double y, double z) {
       double f_result;
       f_result = x + y*z;
       return f_result;
    }
    <BLANKLINE>
    >>> print(h_name)
    test.h
    >>> print(c_header)
    #ifndef PROJECT__TEST__H
    #define PROJECT__TEST__H
    double f(double x, double y, double z);
    #endif
    <BLANKLINE>

    Another example using Equality objects to give named outputs.  Here the
    filename (prefix) is taken from the first (name, expr) pair.

    >>> from sympy.abc import f, g
    >>> from sympy import Eq
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...      [("myfcn", x + y), ("fcn2", [Eq(f, 2*x), Eq(g, y)])],
    ...      "C99", header=False, empty=False)
    >>> print(c_name)
    myfcn.c
    >>> print(c_code)
    #include "myfcn.h"
    #include <math.h>
    double myfcn(double x, double y) {
       double myfcn_result;
       myfcn_result = x + y;
       return myfcn_result;
    }
    void fcn2(double x, double y, double *f, double *g) {
       (*f) = 2*x;
       (*g) = y;
    }
    <BLANKLINE>

    If the generated function(s) will be part of a larger project where various
    global variables have been defined, the 'global_vars' option can be used
    to remove the specified variables from the function signature

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(f_name, f_code), header] = codegen(
    ...     ("f", x+y*z), "F95", header=False, empty=False,
    ...     argument_sequence=(x, y), global_vars=(z,))
    >>> print(f_code)
    REAL*8 function f(x, y)
    implicit none
    REAL*8, intent(in) :: x
    REAL*8, intent(in) :: y
    f = x + y*z
    end function
    <BLANKLINE>

    """

    # Initialize the code generator.
    if language is None:
        if code_gen is None:
            raise ValueError("Need either language or code_gen")
    else:
        if code_gen is not None:
            raise ValueError("You cannot specify both language and code_gen.")
        code_gen = get_code_generator(language, project, standard, printer)

    if isinstance(name_expr[0], str):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    if prefix is None:
        prefix = name_expr[0][0]

    # Construct Routines appropriate for this code_gen from (name, expr) pairs.
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence,
                                         global_vars))

    # Write the code.
    return code_gen.write(routines, prefix, to_files, header, empty)


def make_routine(name, expr, argument_sequence=None,
                 global_vars=None, language="F95"):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    Notes
    =====

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    >>> [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    [1, Matrix([[x, 2]])]

    >>> [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    [g]
    >>> [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    [g + x]

    """

    # initialize a new code generator
    code_gen = get_code_generator(language)

    return code_gen.routine(name, expr, argument_sequence, global_vars)
