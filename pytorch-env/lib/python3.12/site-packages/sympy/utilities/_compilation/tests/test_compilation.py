import shutil
import os
import subprocess
import tempfile
from sympy.external import import_module
from sympy.testing.pytest import skip

from sympy.utilities._compilation.compilation import compile_link_import_py_ext, compile_link_import_strings, compile_sources, get_abspath

numpy = import_module('numpy')
cython = import_module('cython')

_sources1 = [
    ('sigmoid.c', r"""
#include <math.h>

void sigmoid(int n, const double * const restrict in,
             double * const restrict out, double lim){
    for (int i=0; i<n; ++i){
        const double x = in[i];
        out[i] = x*pow(pow(x/lim, 8)+1, -1./8.);
    }
}
"""),
    ('_sigmoid.pyx', r"""
import numpy as np
cimport numpy as cnp

cdef extern void c_sigmoid "sigmoid" (int, const double * const,
                                      double * const, double)

def sigmoid(double [:] inp, double lim=350.0):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(
        inp.size, dtype=np.float64)
    c_sigmoid(inp.size, &inp[0], &out[0], lim)
    return out
""")
]


def npy(data, lim=350.0):
    return data/((data/lim)**8+1)**(1/8.)


def test_compile_link_import_strings():
    if not numpy:
        skip("numpy not installed.")
    if not cython:
        skip("cython not installed.")

    from sympy.utilities._compilation import has_c
    if not has_c():
        skip("No C compiler found.")

    compile_kw = {"std": 'c99', "include_dirs": [numpy.get_include()]}
    info = None
    try:
        mod, info = compile_link_import_strings(_sources1, compile_kwargs=compile_kw)
        data = numpy.random.random(1024*1024*8)  # 64 MB of RAM needed..
        res_mod = mod.sigmoid(data)
        res_npy = npy(data)
        assert numpy.allclose(res_mod, res_npy)
    finally:
        if info and info['build_dir']:
            shutil.rmtree(info['build_dir'])


def test_compile_sources(tmpdir):
    from sympy.utilities._compilation import has_c
    if not has_c():
        skip("No C compiler found.")

    build_dir = str(tmpdir)
    _handle, file_path = tempfile.mkstemp('.c', dir=build_dir)
    with open(file_path, 'wt') as ofh:
        ofh.write("""
        int foo(int bar) {
            return 2*bar;
        }
        """)
    obj, = compile_sources([file_path], cwd=build_dir)
    obj_path = get_abspath(obj, cwd=build_dir)
    assert os.path.exists(obj_path)
    try:
        _ = subprocess.check_output(["nm", "--help"])
    except subprocess.CalledProcessError:
        pass  # we cannot test contents of object file
    else:
        nm_out = subprocess.check_output(["nm", obj_path])
        assert 'foo' in nm_out.decode('utf-8')

    if not cython:
        return  # the final (optional) part of the test below requires Cython.

    _handle, pyx_path = tempfile.mkstemp('.pyx', dir=build_dir)
    with open(pyx_path, 'wt') as ofh:
        ofh.write(("cdef extern int foo(int)\n"
                   "def _foo(arg):\n"
                   "    return foo(arg)"))
    mod = compile_link_import_py_ext([pyx_path], extra_objs=[obj_path], build_dir=build_dir)
    assert mod._foo(21) == 42
