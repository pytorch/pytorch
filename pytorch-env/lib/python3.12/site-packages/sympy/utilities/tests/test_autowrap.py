# Tests that require installed backends go into
# sympy/test_external/test_autowrap

import os
import tempfile
import shutil
from io import StringIO

from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (autowrap, binary_function,
            CythonCodeWrapper, UfuncifyCodeWrapper, CodeWrapper)
from sympy.utilities.codegen import (
    CCodeGen, C99CodeGen, CodeGenArgumentListError, make_routine
)
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager


def get_string(dump_fn, routines, prefix="file", **kwargs):
    """Wrapper for dump_fn. dump_fn writes its results to a stream object and
       this wrapper returns the contents of that stream as a string. This
       auxiliary function is used by many tests below.

       The header and the empty lines are not generator to facilitate the
       testing of the output.
    """
    output = StringIO()
    dump_fn(routines, output, prefix, **kwargs)
    source = output.getvalue()
    output.close()
    return source


def test_cython_wrapper_scalar_function():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = CythonCodeWrapper(CCodeGen())
    source = get_string(code_gen.dump_pyx, [routine])

    expected = (
        "cdef extern from 'file.h':\n"
        "    double test(double x, double y, double z)\n"
        "\n"
        "def test_c(double x, double y, double z):\n"
        "\n"
        "    return test(x, y, z)")
    assert source == expected


def test_cython_wrapper_outarg():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())

    routine = make_routine("test", Equality(z, x + y))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = (
        "cdef extern from 'file.h':\n"
        "    void test(double x, double y, double *z)\n"
        "\n"
        "def test_c(double x, double y):\n"
        "\n"
        "    cdef double z = 0\n"
        "    test(x, y, &z)\n"
        "    return z")
    assert source == expected


def test_cython_wrapper_inoutarg():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())
    routine = make_routine("test", Equality(z, x + y + z))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = (
        "cdef extern from 'file.h':\n"
        "    void test(double x, double y, double *z)\n"
        "\n"
        "def test_c(double x, double y, double z):\n"
        "\n"
        "    test(x, y, &z)\n"
        "    return z")
    assert source == expected


def test_cython_wrapper_compile_flags():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    routine = make_routine("test", Equality(z, x + y))

    code_gen = CythonCodeWrapper(CCodeGen())

    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'language_level': '3'}}

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=[],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

    temp_dir = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    setup_file_path = os.path.join(temp_dir, 'setup.py')

    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected

    code_gen = CythonCodeWrapper(CCodeGen(),
                                 include_dirs=['/usr/local/include', '/opt/booger/include'],
                                 library_dirs=['/user/local/lib'],
                                 libraries=['thelib', 'nilib'],
                                 extra_compile_args=['-slow-math'],
                                 extra_link_args=['-lswamp', '-ltrident'],
                                 cythonize_options={'compiler_directives': {'boundscheck': False}}
                                 )
    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'boundscheck': False}}

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=['/usr/local/include', '/opt/booger/include'],
    library_dirs=['/user/local/lib'],
    libraries=['thelib', 'nilib'],
    extra_compile_args=['-slow-math', '-std=c99'],
    extra_link_args=['-lswamp', '-ltrident']
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected

    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'boundscheck': False}}
import numpy as np

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=['/usr/local/include', '/opt/booger/include', np.get_include()],
    library_dirs=['/user/local/lib'],
    libraries=['thelib', 'nilib'],
    extra_compile_args=['-slow-math', '-std=c99'],
    extra_link_args=['-lswamp', '-ltrident']
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

    code_gen._need_numpy = True
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected

    TmpFileManager.cleanup()

def test_cython_wrapper_unique_dummyvars():
    from sympy.core.relational import Equality
    from sympy.core.symbol import Dummy
    x, y, z = Dummy('x'), Dummy('y'), Dummy('z')
    x_id, y_id, z_id = [str(d.dummy_index) for d in [x, y, z]]
    expr = Equality(z, x + y)
    routine = make_routine("test", expr)
    code_gen = CythonCodeWrapper(CCodeGen())
    source = get_string(code_gen.dump_pyx, [routine])
    expected_template = (
        "cdef extern from 'file.h':\n"
        "    void test(double x_{x_id}, double y_{y_id}, double *z_{z_id})\n"
        "\n"
        "def test_c(double x_{x_id}, double y_{y_id}):\n"
        "\n"
        "    cdef double z_{z_id} = 0\n"
        "    test(x_{x_id}, y_{y_id}, &z_{z_id})\n"
        "    return z_{z_id}")
    expected = expected_template.format(x_id=x_id, y_id=y_id, z_id=z_id)
    assert source == expected

def test_autowrap_dummy():
    x, y, z = symbols('x y z')

    # Uses DummyWrapper to test that codegen works as expected

    f = autowrap(x + y, backend='dummy')
    assert f() == str(x + y)
    assert f.args == "x, y"
    assert f.returns == "nameless"
    f = autowrap(Eq(z, x + y), backend='dummy')
    assert f() == str(x + y)
    assert f.args == "x, y"
    assert f.returns == "z"
    f = autowrap(Eq(z, x + y + z), backend='dummy')
    assert f() == str(x + y + z)
    assert f.args == "x, y, z"
    assert f.returns == "z"


def test_autowrap_args():
    x, y, z = symbols('x y z')

    raises(CodeGenArgumentListError, lambda: autowrap(Eq(z, x + y),
           backend='dummy', args=[x]))
    f = autowrap(Eq(z, x + y), backend='dummy', args=[y, x])
    assert f() == str(x + y)
    assert f.args == "y, x"
    assert f.returns == "z"

    raises(CodeGenArgumentListError, lambda: autowrap(Eq(z, x + y + z),
           backend='dummy', args=[x, y]))
    f = autowrap(Eq(z, x + y + z), backend='dummy', args=[y, x, z])
    assert f() == str(x + y + z)
    assert f.args == "y, x, z"
    assert f.returns == "z"

    f = autowrap(Eq(z, x + y + z), backend='dummy', args=(y, x, z))
    assert f() == str(x + y + z)
    assert f.args == "y, x, z"
    assert f.returns == "z"

def test_autowrap_store_files():
    x, y = symbols('x y')
    tmp = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(tmp)

    f = autowrap(x + y, backend='dummy', tempdir=tmp)
    assert f() == str(x + y)
    assert os.access(tmp, os.F_OK)

    TmpFileManager.cleanup()

def test_autowrap_store_files_issue_gh12939():
    x, y = symbols('x y')
    tmp = './tmp'
    saved_cwd = os.getcwd()
    temp_cwd = tempfile.mkdtemp()
    try:
        os.chdir(temp_cwd)
        f = autowrap(x + y, backend='dummy', tempdir=tmp)
        assert f() == str(x + y)
        assert os.access(tmp, os.F_OK)
    finally:
        os.chdir(saved_cwd)
        shutil.rmtree(temp_cwd)


def test_binary_function():
    x, y = symbols('x y')
    f = binary_function('f', x + y, backend='dummy')
    assert f._imp_() == str(x + y)


def test_ufuncify_source():
    x, y, z = symbols('x,y,z')
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"))
    routine = make_routine("test", x + y + z)
    source = get_string(code_wrapper.dump_c, [routine])
    expected = """\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "file.h"

static PyMethodDef wrapper_module_%(num)sMethods[] = {
        {NULL, NULL, 0, NULL}
};

#ifdef NPY_1_19_API_VERSION
static void test_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void test_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    char *in1 = args[1];
    char *in2 = args[2];
    char *out0 = args[3];
    npy_intp in0_step = steps[0];
    npy_intp in1_step = steps[1];
    npy_intp in2_step = steps[2];
    npy_intp out0_step = steps[3];
    for (i = 0; i < n; i++) {
        *((double *)out0) = test(*(double *)in0, *(double *)in1, *(double *)in2);
        in0 += in0_step;
        in1 += in1_step;
        in2 += in2_step;
        out0 += out0_step;
    }
}
PyUFuncGenericFunction test_funcs[1] = {&test_ufunc};
static char test_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *test_data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "wrapper_module_%(num)s",
    NULL,
    -1,
    wrapper_module_%(num)sMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "test", ufunc0);
    Py_DECREF(ufunc0);
    return m;
}
#else
PyMODINIT_FUNC initwrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "test", ufunc0);
    Py_DECREF(ufunc0);
}
#endif""" % {'num': CodeWrapper._module_counter}
    assert source == expected


def test_ufuncify_source_multioutput():
    x, y, z = symbols('x,y,z')
    var_symbols = (x, y, z)
    expr = x + y**3 + 10*z**2
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"))
    routines = [make_routine("func{}".format(i), expr.diff(var_symbols[i]), var_symbols) for i in range(len(var_symbols))]
    source = get_string(code_wrapper.dump_c, routines, funcname='multitest')
    expected = """\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "file.h"

static PyMethodDef wrapper_module_%(num)sMethods[] = {
        {NULL, NULL, 0, NULL}
};

#ifdef NPY_1_19_API_VERSION
static void multitest_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void multitest_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    char *in1 = args[1];
    char *in2 = args[2];
    char *out0 = args[3];
    char *out1 = args[4];
    char *out2 = args[5];
    npy_intp in0_step = steps[0];
    npy_intp in1_step = steps[1];
    npy_intp in2_step = steps[2];
    npy_intp out0_step = steps[3];
    npy_intp out1_step = steps[4];
    npy_intp out2_step = steps[5];
    for (i = 0; i < n; i++) {
        *((double *)out0) = func0(*(double *)in0, *(double *)in1, *(double *)in2);
        *((double *)out1) = func1(*(double *)in0, *(double *)in1, *(double *)in2);
        *((double *)out2) = func2(*(double *)in0, *(double *)in1, *(double *)in2);
        in0 += in0_step;
        in1 += in1_step;
        in2 += in2_step;
        out0 += out0_step;
        out1 += out1_step;
        out2 += out2_step;
    }
}
PyUFuncGenericFunction multitest_funcs[1] = {&multitest_ufunc};
static char multitest_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *multitest_data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "wrapper_module_%(num)s",
    NULL,
    -1,
    wrapper_module_%(num)sMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "multitest", ufunc0);
    Py_DECREF(ufunc0);
    return m;
}
#else
PyMODINIT_FUNC initwrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "multitest", ufunc0);
    Py_DECREF(ufunc0);
}
#endif""" % {'num': CodeWrapper._module_counter}
    assert source == expected
