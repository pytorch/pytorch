"""Module for compiling codegen output, and wrap the binary for use in
python.

.. note:: To use the autowrap module it must first be imported

   >>> from sympy.utilities.autowrap import autowrap

This module provides a common interface for different external backends, such
as f2py, fwrap, Cython, SWIG(?) etc. (Currently only f2py and Cython are
implemented) The goal is to provide access to compiled binaries of acceptable
performance with a one-button user interface, e.g.,

    >>> from sympy.abc import x,y
    >>> expr = (x - y)**25
    >>> flat = expr.expand()
    >>> binary_callable = autowrap(flat)
    >>> binary_callable(2, 3)
    -1.0

Although a SymPy user might primarily be interested in working with
mathematical expressions and not in the details of wrapping tools
needed to evaluate such expressions efficiently in numerical form,
the user cannot do so without some understanding of the
limits in the target language. For example, the expanded expression
contains large coefficients which result in loss of precision when
computing the expression:

    >>> binary_callable(3, 2)
    0.0
    >>> binary_callable(4, 5), binary_callable(5, 4)
    (-22925376.0, 25165824.0)

Wrapping the unexpanded expression gives the expected behavior:

    >>> e = autowrap(expr)
    >>> e(4, 5), e(5, 4)
    (-1.0, 1.0)

The callable returned from autowrap() is a binary Python function, not a
SymPy object.  If it is desired to use the compiled function in symbolic
expressions, it is better to use binary_function() which returns a SymPy
Function object.  The binary callable is attached as the _imp_ attribute and
invoked when a numerical evaluation is requested with evalf(), or with
lambdify().

    >>> from sympy.utilities.autowrap import binary_function
    >>> f = binary_function('f', expr)
    >>> 2*f(x, y) + y
    y + 2*f(x, y)
    >>> (2*f(x, y) + y).evalf(2, subs={x: 1, y:2})
    0.e-110

When is this useful?

    1) For computations on large arrays, Python iterations may be too slow,
       and depending on the mathematical expression, it may be difficult to
       exploit the advanced index operations provided by NumPy.

    2) For *really* long expressions that will be called repeatedly, the
       compiled binary should be significantly faster than SymPy's .evalf()

    3) If you are generating code with the codegen utility in order to use
       it in another project, the automatic Python wrappers let you test the
       binaries immediately from within SymPy.

    4) To create customized ufuncs for use with numpy arrays.
       See *ufuncify*.

When is this module NOT the best approach?

    1) If you are really concerned about speed or memory optimizations,
       you will probably get better results by working directly with the
       wrapper tools and the low level code.  However, the files generated
       by this utility may provide a useful starting point and reference
       code. Temporary files will be left intact if you supply the keyword
       tempdir="path/to/files/".

    2) If the array computation can be handled easily by numpy, and you
       do not need the binaries for another project.

"""

import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn

from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
                                     OutputArgument, InOutArgument,
                                     InputArgument, CodeGenArgumentListError,
                                     Result, ResultBase, C99CodeGen)
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on

_doctest_depends_on = {'exe': ('f2py', 'gfortran', 'gcc'),
                       'modules': ('numpy',)}


class CodeWrapError(Exception):
    pass


class CodeWrapper:
    """Base Class for code wrappers"""
    _filename = "wrapped_code"
    _module_basename = "wrapper_module"
    _module_counter = 0

    @property
    def filename(self):
        return "%s_%s" % (self._filename, CodeWrapper._module_counter)

    @property
    def module_name(self):
        return "%s_%s" % (self._module_basename, CodeWrapper._module_counter)

    def __init__(self, generator, filepath=None, flags=[], verbose=False):
        """
        generator -- the code generator to use
        """
        self.generator = generator
        self.filepath = filepath
        self.flags = flags
        self.quiet = not verbose

    @property
    def include_header(self):
        return bool(self.filepath)

    @property
    def include_empty(self):
        return bool(self.filepath)

    def _generate_code(self, main_routine, routines):
        routines.append(main_routine)
        self.generator.write(
            routines, self.filename, True, self.include_header,
            self.include_empty)

    def wrap_code(self, routine, helpers=None):
        helpers = helpers or []
        if self.filepath:
            workdir = os.path.abspath(self.filepath)
        else:
            workdir = tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routine, helpers)
            self._prepare_files(routine)
            self._process_files(routine)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return self._get_wrapped_function(mod, routine.name)

    def _process_files(self, routine):
        command = self.command
        command.extend(self.flags)
        try:
            retoutput = check_output(command, stderr=STDOUT)
        except CalledProcessError as e:
            raise CodeWrapError(
                "Error while executing command: %s. Command output is:\n%s" % (
                    " ".join(command), e.output.decode('utf-8')))
        if not self.quiet:
            print(retoutput)


class DummyWrapper(CodeWrapper):
    """Class used for testing independent of backends """

    template = """# dummy module for testing of SymPy
def %(name)s():
    return "%(expr)s"
%(name)s.args = "%(args)s"
%(name)s.returns = "%(retvals)s"
"""

    def _prepare_files(self, routine):
        return

    def _generate_code(self, routine, helpers):
        with open('%s.py' % self.module_name, 'w') as f:
            printed = ", ".join(
                [str(res.expr) for res in routine.result_variables])
            # convert OutputArguments to return value like f2py
            args = filter(lambda x: not isinstance(
                x, OutputArgument), routine.arguments)
            retvals = []
            for val in routine.result_variables:
                if isinstance(val, Result):
                    retvals.append('nameless')
                else:
                    retvals.append(val.result_var)

            print(DummyWrapper.template % {
                'name': routine.name,
                'expr': printed,
                'args': ", ".join([str(a.name) for a in args]),
                'retvals': ", ".join([str(val) for val in retvals])
            }, end="", file=f)

    def _process_files(self, routine):
        return

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


class CythonCodeWrapper(CodeWrapper):
    """Wrapper that uses Cython"""

    setup_template = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {cythonize_options}
{np_import}
ext_mods = [Extension(
    {ext_args},
    include_dirs={include_dirs},
    library_dirs={library_dirs},
    libraries={libraries},
    extra_compile_args={extra_compile_args},
    extra_link_args={extra_link_args}
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
"""

    _cythonize_options = {'compiler_directives':{'language_level' : "3"}}

    pyx_imports = (
        "import numpy as np\n"
        "cimport numpy as np\n\n")

    pyx_header = (
        "cdef extern from '{header_file}.h':\n"
        "    {prototype}\n\n")

    pyx_func = (
        "def {name}_c({arg_string}):\n"
        "\n"
        "{declarations}"
        "{body}")

    std_compile_flag = '-std=c99'

    def __init__(self, *args, **kwargs):
        """Instantiates a Cython code wrapper.

        The following optional parameters get passed to ``setuptools.Extension``
        for building the Python extension module. Read its documentation to
        learn more.

        Parameters
        ==========
        include_dirs : [list of strings]
            A list of directories to search for C/C++ header files (in Unix
            form for portability).
        library_dirs : [list of strings]
            A list of directories to search for C/C++ libraries at link time.
        libraries : [list of strings]
            A list of library names (not filenames or paths) to link against.
        extra_compile_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            compiling the source files in 'sources'.  For platforms and
            compilers where "command line" makes sense, this is typically a
            list of command-line arguments, but for other platforms it could be
            anything. Note that the attribute ``std_compile_flag`` will be
            appended to this list.
        extra_link_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            linking object files together to create the extension (or to create
            a new static Python interpreter). Similar interpretation as for
            'extra_compile_args'.
        cythonize_options : [dictionary]
            Keyword arguments passed on to cythonize.

        """

        self._include_dirs = kwargs.pop('include_dirs', [])
        self._library_dirs = kwargs.pop('library_dirs', [])
        self._libraries = kwargs.pop('libraries', [])
        self._extra_compile_args = kwargs.pop('extra_compile_args', [])
        self._extra_compile_args.append(self.std_compile_flag)
        self._extra_link_args = kwargs.pop('extra_link_args', [])
        self._cythonize_options = kwargs.pop('cythonize_options', self._cythonize_options)

        self._need_numpy = False

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def _prepare_files(self, routine, build_dir=os.curdir):
        # NOTE : build_dir is used for testing purposes.
        pyxfilename = self.module_name + '.pyx'
        codefilename = "%s.%s" % (self.filename, self.generator.code_extension)

        # pyx
        with open(os.path.join(build_dir, pyxfilename), 'w') as f:
            self.dump_pyx([routine], f, self.filename)

        # setup.py
        ext_args = [repr(self.module_name), repr([pyxfilename, codefilename])]
        if self._need_numpy:
            np_import = 'import numpy as np\n'
            self._include_dirs.append('np.get_include()')
        else:
            np_import = ''

        with open(os.path.join(build_dir, 'setup.py'), 'w') as f:
            includes = str(self._include_dirs).replace("'np.get_include()'",
                                                       'np.get_include()')
            f.write(self.setup_template.format(
                ext_args=", ".join(ext_args),
                np_import=np_import,
                include_dirs=includes,
                library_dirs=self._library_dirs,
                libraries=self._libraries,
                extra_compile_args=self._extra_compile_args,
                extra_link_args=self._extra_link_args,
                cythonize_options=self._cythonize_options
            ))

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name + '_c')

    def dump_pyx(self, routines, f, prefix):
        """Write a Cython file with Python wrappers

        This file contains all the definitions of the routines in c code and
        refers to the header file.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.
        """
        headers = []
        functions = []
        for routine in routines:
            prototype = self.generator.get_prototype(routine)

            # C Function Header Import
            headers.append(self.pyx_header.format(header_file=prefix,
                                                  prototype=prototype))

            # Partition the C function arguments into categories
            py_rets, py_args, py_loc, py_inf = self._partition_args(routine.arguments)

            # Function prototype
            name = routine.name
            arg_string = ", ".join(self._prototype_arg(arg) for arg in py_args)

            # Local Declarations
            local_decs = []
            for arg, val in py_inf.items():
                proto = self._prototype_arg(arg)
                mat, ind = [self._string_var(v) for v in val]
                local_decs.append("    cdef {} = {}.shape[{}]".format(proto, mat, ind))
            local_decs.extend(["    cdef {}".format(self._declare_arg(a)) for a in py_loc])
            declarations = "\n".join(local_decs)
            if declarations:
                declarations = declarations + "\n"

            # Function Body
            args_c = ", ".join([self._call_arg(a) for a in routine.arguments])
            rets = ", ".join([self._string_var(r.name) for r in py_rets])
            if routine.results:
                body = '    return %s(%s)' % (routine.name, args_c)
                if rets:
                    body = body + ', ' + rets
            else:
                body = '    %s(%s)\n' % (routine.name, args_c)
                body = body + '    return ' + rets

            functions.append(self.pyx_func.format(name=name, arg_string=arg_string,
                    declarations=declarations, body=body))

        # Write text to file
        if self._need_numpy:
            # Only import numpy if required
            f.write(self.pyx_imports)
        f.write('\n'.join(headers))
        f.write('\n'.join(functions))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_args = []
        py_returns = []
        py_locals = []
        py_inferred = {}
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_returns.append(arg)
                py_locals.append(arg)
            elif isinstance(arg, InOutArgument):
                py_returns.append(arg)
                py_args.append(arg)
            else:
                py_args.append(arg)
        # Find arguments that are array dimensions. These can be inferred
        # locally in the Cython code.
            if isinstance(arg, (InputArgument, InOutArgument)) and arg.dimensions:
                dims = [d[1] + 1 for d in arg.dimensions]
                sym_dims = [(i, d) for (i, d) in enumerate(dims) if
                            isinstance(d, Symbol)]
                for (i, d) in sym_dims:
                    py_inferred[d] = (arg.name, i)
        for arg in args:
            if arg.name in py_inferred:
                py_inferred[arg] = py_inferred.pop(arg.name)
        # Filter inferred arguments from py_args
        py_args = [a for a in py_args if a not in py_inferred]
        return py_returns, py_args, py_locals, py_inferred

    def _prototype_arg(self, arg):
        mat_dec = "np.ndarray[{mtype}, ndim={ndim}] {name}"
        np_types = {'double': 'np.double_t',
                    'int': 'np.int_t'}
        t = arg.get_datatype('c')
        if arg.dimensions:
            self._need_numpy = True
            ndim = len(arg.dimensions)
            mtype = np_types[t]
            return mat_dec.format(mtype=mtype, ndim=ndim, name=self._string_var(arg.name))
        else:
            return "%s %s" % (t, self._string_var(arg.name))

    def _declare_arg(self, arg):
        proto = self._prototype_arg(arg)
        if arg.dimensions:
            shape = '(' + ','.join(self._string_var(i[1] + 1) for i in arg.dimensions) + ')'
            return proto + " = np.empty({shape})".format(shape=shape)
        else:
            return proto + " = 0"

    def _call_arg(self, arg):
        if arg.dimensions:
            t = arg.get_datatype('c')
            return "<{}*> {}.data".format(t, self._string_var(arg.name))
        elif isinstance(arg, ResultBase):
            return "&{}".format(self._string_var(arg.name))
        else:
            return self._string_var(arg.name)

    def _string_var(self, var):
        printer = self.generator.printer.doprint
        return printer(var)


class F2PyCodeWrapper(CodeWrapper):
    """Wrapper that uses f2py"""

    def __init__(self, *args, **kwargs):

        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the f2py '
               'backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        filename = self.filename + '.' + self.generator.code_extension
        args = ['-c', '-m', self.module_name, filename]
        command = [sys.executable, "-c", "import numpy.f2py as f2py2e;f2py2e.main()"]+args
        return command

    def _prepare_files(self, routine):
        pass

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


# Here we define a lookup of backends -> tuples of languages. For now, each
# tuple is of length 1, but if a backend supports more than one language,
# the most preferable language is listed first.
_lang_lookup = {'CYTHON': ('C99', 'C89', 'C'),
                'F2PY': ('F95',),
                'NUMPY': ('C99', 'C89', 'C'),
                'DUMMY': ('F95',)}     # Dummy here just for testing


def _infer_language(backend):
    """For a given backend, return the top choice of language"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    return langs[0]


def _validate_backend_language(backend, language):
    """Throws error if backend and language are incompatible"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    if language.upper() not in langs:
        raise ValueError(("Backend {} and language {} are "
                          "incompatible").format(backend, language))


@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def autowrap(expr, language=None, backend='f2py', tempdir=None, args=None,
             flags=None, verbose=False, helpers=None, code_gen=None, **kwargs):
    """Generates Python callable binaries based on the math expression.

    Parameters
    ==========

    expr
        The SymPy expression that should be wrapped as a binary routine.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'f2py' [default],
        or 'cython'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in the
        specified path.
    args : iterable, optional
        An ordered iterable of symbols. Specifies the argument sequence for the
        function.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can be
        helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary expressions needed for the main expr. If the
        main expression needs to call a specialized function it should be
        passed in via ``helpers``. Autowrap will then make sure that the
        compiled main expression can link to the helper routine. Items should
        be 3-tuples with (<function_name>, <sympy_expression>,
        <argument_tuple>). It is mandatory to supply an argument sequence to
        helper routines.
    code_gen : CodeGen instance
        An instance of a CodeGen subclass. Overrides ``language``.
    include_dirs : [string]
        A list of directories to search for C/C++ header files (in Unix form
        for portability).
    library_dirs : [string]
        A list of directories to search for C/C++ libraries at link time.
    libraries : [string]
        A list of library names (not filenames or paths) to link against.
    extra_compile_args : [string]
        Any extra platform- and compiler-specific information to use when
        compiling the source files in 'sources'.  For platforms and compilers
        where "command line" makes sense, this is typically a list of
        command-line arguments, but for other platforms it could be anything.
    extra_link_args : [string]
        Any extra platform- and compiler-specific information to use when
        linking object files together to create the extension (or to create a
        new static Python interpreter).  Similar interpretation as for
        'extra_compile_args'.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.autowrap import autowrap
    >>> expr = ((x - y + z)**(13)).expand()
    >>> binary_func = autowrap(expr)
    >>> binary_func(1, 4, 2)
    -1.0

    """
    if language:
        if not isinstance(language, type):
            _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    # two cases 1) helpers is an iterable of 3-tuples and 2) helpers is a
    # 3-tuple
    if iterable(helpers) and len(helpers) != 0 and iterable(helpers[0]):
        helpers = helpers if helpers else ()
    else:
        helpers = [helpers] if helpers else ()
    args = list(args) if iterable(args, exclude=set) else args

    if code_gen is None:
        code_gen = get_code_generator(language, "autowrap")

    CodeWrapperClass = {
        'F2PY': F2PyCodeWrapper,
        'CYTHON': CythonCodeWrapper,
        'DUMMY': DummyWrapper
    }[backend.upper()]
    code_wrapper = CodeWrapperClass(code_gen, tempdir, flags if flags else (),
                                    verbose, **kwargs)

    helps = []
    for name_h, expr_h, args_h in helpers:
        helps.append(code_gen.routine(name_h, expr_h, args_h))

    for name_h, expr_h, args_h in helpers:
        if expr.has(expr_h):
            name_h = binary_function(name_h, expr_h, backend='dummy')
            expr = expr.subs(expr_h, name_h(*args_h))
    try:
        routine = code_gen.routine('autofunc', expr, args)
    except CodeGenArgumentListError as e:
        # if all missing arguments are for pure output, we simply attach them
        # at the end and try again, because the wrappers will silently convert
        # them to return values anyway.
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, OutputArgument):
                raise
            new_args.append(missing.name)
        routine = code_gen.routine('autofunc', expr, args + new_args)

    return code_wrapper.wrap_code(routine, helpers=helps)


@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def binary_function(symfunc, expr, **kwargs):
    """Returns a SymPy function with expr as binary implementation

    This is a convenience function that automates the steps needed to
    autowrap the SymPy expression and attaching it to a Function object
    with implemented_function().

    Parameters
    ==========

    symfunc : SymPy Function
        The function to bind the callable to.
    expr : SymPy Expression
        The expression used to generate the function.
    kwargs : dict
        Any kwargs accepted by autowrap.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.utilities.autowrap import binary_function
    >>> expr = ((x - y)**(25)).expand()
    >>> f = binary_function('f', expr)
    >>> type(f)
    <class 'sympy.core.function.UndefinedFunction'>
    >>> 2*f(x, y)
    2*f(x, y)
    >>> f(x, y).evalf(2, subs={x: 1, y: 2})
    -1.0

    """
    binary = autowrap(expr, **kwargs)
    return implemented_function(symfunc, binary)

#################################################################
#                           UFUNCIFY                            #
#################################################################

_ufunc_top = Template("""\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include ${include_file}

static PyMethodDef ${module}Methods[] = {
        {NULL, NULL, 0, NULL}
};""")

_ufunc_outcalls = Template("*((double *)out${outnum}) = ${funcname}(${call_args});")

_ufunc_body = Template("""\
#ifdef NPY_1_19_API_VERSION
static void ${funcname}_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void ${funcname}_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    ${declare_args}
    ${declare_steps}
    for (i = 0; i < n; i++) {
        ${outcalls}
        ${step_increments}
    }
}
PyUFuncGenericFunction ${funcname}_funcs[1] = {&${funcname}_ufunc};
static char ${funcname}_types[${n_types}] = ${types}
static void *${funcname}_data[1] = {NULL};""")

_ufunc_bottom = Template("""\
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "${module}",
    NULL,
    -1,
    ${module}Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
    return m;
}
#else
PyMODINIT_FUNC init${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = Py_InitModule("${module}", ${module}Methods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
}
#endif\
""")

_ufunc_init_form = Template("""\
ufunc${ind} = PyUFunc_FromFuncAndData(${funcname}_funcs, ${funcname}_data, ${funcname}_types, 1, ${n_in}, ${n_out},
            PyUFunc_None, "${module}", ${docstring}, 0);
    PyDict_SetItemString(d, "${funcname}", ufunc${ind});
    Py_DECREF(ufunc${ind});""")

_ufunc_setup = Template("""\
from setuptools.extension import Extension
from setuptools import setup

from numpy import get_include

if __name__ == "__main__":
    setup(ext_modules=[
        Extension('${module}',
                  sources=['${module}.c', '${filename}.c'],
                  include_dirs=[get_include()])])
""")


class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""

    def __init__(self, *args, **kwargs):

        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the numpy'
               ' backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def wrap_code(self, routines, helpers=None):
        # This routine overrides CodeWrapper because we can't assume funcname == routines[0].name
        # Therefore we have to break the CodeWrapper private API.
        # There isn't an obvious way to extend multi-expr support to
        # the other autowrap backends, so we limit this change to ufuncify.
        helpers = helpers if helpers is not None else []
        # We just need a consistent name
        funcname = 'wrapped_' + str(id(routines) + id(helpers))

        workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routines, helpers)
            self._prepare_files(routines, funcname)
            self._process_files(routines)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return self._get_wrapped_function(mod, funcname)

    def _generate_code(self, main_routines, helper_routines):
        all_routines = main_routines + helper_routines
        self.generator.write(
            all_routines, self.filename, True, self.include_header,
            self.include_empty)

    def _prepare_files(self, routines, funcname):

        # C
        codefilename = self.module_name + '.c'
        with open(codefilename, 'w') as f:
            self.dump_c(routines, f, self.filename, funcname=funcname)

        # setup.py
        with open('setup.py', 'w') as f:
            self.dump_setup(f)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)

    def dump_setup(self, f):
        setup = _ufunc_setup.substitute(module=self.module_name,
                                        filename=self.filename)
        f.write(setup)

    def dump_c(self, routines, f, prefix, funcname=None):
        """Write a C file with Python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
        if funcname is None:
            if len(routines) == 1:
                funcname = routines[0].name
            else:
                msg = 'funcname must be specified for multiple output routines'
                raise ValueError(msg)
        functions = []
        function_creation = []
        ufunc_init = []
        module = self.module_name
        include_file = "\"{}.h\"".format(prefix)
        top = _ufunc_top.substitute(include_file=include_file, module=module)

        name = funcname

        # Partition the C function arguments into categories
        # Here we assume all routines accept the same arguments
        r_index = 0
        py_in, _ = self._partition_args(routines[0].arguments)
        n_in = len(py_in)
        n_out = len(routines)

        # Declare Args
        form = "char *{0}{1} = args[{2}];"
        arg_decs = [form.format('in', i, i) for i in range(n_in)]
        arg_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_args = '\n    '.join(arg_decs)

        # Declare Steps
        form = "npy_intp {0}{1}_step = steps[{2}];"
        step_decs = [form.format('in', i, i) for i in range(n_in)]
        step_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_steps = '\n    '.join(step_decs)

        # Call Args
        form = "*(double *)in{0}"
        call_args = ', '.join([form.format(a) for a in range(n_in)])

        # Step Increments
        form = "{0}{1} += {0}{1}_step;"
        step_incs = [form.format('in', i) for i in range(n_in)]
        step_incs.extend([form.format('out', i, i) for i in range(n_out)])
        step_increments = '\n        '.join(step_incs)

        # Types
        n_types = n_in + n_out
        types = "{" + ', '.join(["NPY_DOUBLE"]*n_types) + "};"

        # Docstring
        docstring = '"Created in SymPy with Ufuncify"'

        # Function Creation
        function_creation.append("PyObject *ufunc{};".format(r_index))

        # Ufunc initialization
        init_form = _ufunc_init_form.substitute(module=module,
                                                funcname=name,
                                                docstring=docstring,
                                                n_in=n_in, n_out=n_out,
                                                ind=r_index)
        ufunc_init.append(init_form)

        outcalls = [_ufunc_outcalls.substitute(
            outnum=i, call_args=call_args, funcname=routines[i].name) for i in
            range(n_out)]

        body = _ufunc_body.substitute(module=module, funcname=name,
                                      declare_args=declare_args,
                                      declare_steps=declare_steps,
                                      call_args=call_args,
                                      step_increments=step_increments,
                                      n_types=n_types, types=types,
                                      outcalls='\n        '.join(outcalls))
        functions.append(body)

        body = '\n\n'.join(functions)
        ufunc_init = '\n    '.join(ufunc_init)
        function_creation = '\n    '.join(function_creation)
        bottom = _ufunc_bottom.substitute(module=module,
                                          ufunc_init=ufunc_init,
                                          function_creation=function_creation)
        text = [top, body, bottom]
        f.write('\n\n'.join(text))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_in = []
        py_out = []
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_out.append(arg)
            elif isinstance(arg, InOutArgument):
                raise ValueError("Ufuncify doesn't support InOutArguments")
            else:
                py_in.append(arg)
        return py_in, py_out


@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran', 'gcc'), modules=('numpy',))
def ufuncify(args, expr, language=None, backend='numpy', tempdir=None,
             flags=None, verbose=False, helpers=None, **kwargs):
    """Generates a binary function that supports broadcasting on numpy arrays.

    Parameters
    ==========

    args : iterable
        Either a Symbol or an iterable of symbols. Specifies the argument
        sequence for the function.
    expr
        A SymPy expression that defines the element wise operation.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'numpy' [default],
        'cython', or 'f2py'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in
        the specified path.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can
        be helpful for debugging.
    helpers : iterable, optional
        Used to define auxiliary expressions needed for the main expr. If
        the main expression needs to call a specialized function it should
        be put in the ``helpers`` iterable. Autowrap will then make sure
        that the compiled main expression can link to the helper routine.
        Items should be tuples with (<funtion_name>, <sympy_expression>,
        <arguments>). It is mandatory to supply an argument sequence to
        helper routines.
    kwargs : dict
        These kwargs will be passed to autowrap if the `f2py` or `cython`
        backend is used and ignored if the `numpy` backend is used.

    Notes
    =====

    The default backend ('numpy') will create actual instances of
    ``numpy.ufunc``. These support ndimensional broadcasting, and implicit type
    conversion. Use of the other backends will result in a "ufunc-like"
    function, which requires equal length 1-dimensional arrays for all
    arguments, and will not perform any type conversions.

    References
    ==========

    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html

    Examples
    ========

    >>> from sympy.utilities.autowrap import ufuncify
    >>> from sympy.abc import x, y
    >>> import numpy as np
    >>> f = ufuncify((x, y), y + x**2)
    >>> type(f)
    <class 'numpy.ufunc'>
    >>> f([1, 2, 3], 2)
    array([  3.,   6.,  11.])
    >>> f(np.arange(5), 3)
    array([  3.,   4.,   7.,  12.,  19.])

    For the 'f2py' and 'cython' backends, inputs are required to be equal length
    1-dimensional arrays. The 'f2py' backend will perform type conversion, but
    the Cython backend will error if the inputs are not of the expected type.

    >>> f_fortran = ufuncify((x, y), y + x**2, backend='f2py')
    >>> f_fortran(1, 2)
    array([ 3.])
    >>> f_fortran(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
    array([  2.,   6.,  12.])
    >>> f_cython = ufuncify((x, y), y + x**2, backend='Cython')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument '_x' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    """

    if isinstance(args, Symbol):
        args = (args,)
    else:
        args = tuple(args)

    if language:
        _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    helpers = helpers if helpers else ()
    flags = flags if flags else ()

    if backend.upper() == 'NUMPY':
        # maxargs is set by numpy compile-time constant NPY_MAXARGS
        # If a future version of numpy modifies or removes this restriction
        # this variable should be changed or removed
        maxargs = 32
        helps = []
        for name, expr, args in helpers:
            helps.append(make_routine(name, expr, args))
        code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"), tempdir,
                                           flags, verbose)
        if not isinstance(expr, (list, tuple)):
            expr = [expr]
        if len(expr) == 0:
            raise ValueError('Expression iterable has zero length')
        if len(expr) + len(args) > maxargs:
            msg = ('Cannot create ufunc with more than {0} total arguments: '
                   'got {1} in, {2} out')
            raise ValueError(msg.format(maxargs, len(args), len(expr)))
        routines = [make_routine('autofunc{}'.format(idx), exprx, args) for
                    idx, exprx in enumerate(expr)]
        return code_wrapper.wrap_code(routines, helpers=helps)
    else:
        # Dummies are used for all added expressions to prevent name clashes
        # within the original expression.
        y = IndexedBase(Dummy('y'))
        m = Dummy('m', integer=True)
        i = Idx(Dummy('i', integer=True), m)
        f_dummy = Dummy('f')
        f = implemented_function('%s_%d' % (f_dummy.name, f_dummy.dummy_index), Lambda(args, expr))
        # For each of the args create an indexed version.
        indexed_args = [IndexedBase(Dummy(str(a))) for a in args]
        # Order the arguments (out, args, dim)
        args = [y] + indexed_args + [m]
        args_with_indices = [a[i] for a in indexed_args]
        return autowrap(Eq(y[i], f(*args_with_indices)), language, backend,
                        tempdir, args, flags, verbose, helpers, **kwargs)
