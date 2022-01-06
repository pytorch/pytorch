#include <Python.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>

extern "C" struct _frozen _PyImport_FrozenModules_numpy[];

extern "C" PyObject* PyInit__multiarray_umath(void);
extern "C" PyObject* PyInit__multiarray_tests(void);
extern "C" PyObject* PyInit_lapack_lite(void);
extern "C" PyObject* PyInit__umath_linalg(void);
extern "C" PyObject* PyInit__pocketfft_internal(void);
extern "C" PyObject* PyInit_mtrand(void);
extern "C" PyObject* PyInit_bit_generator(void);
extern "C" PyObject* PyInit__common(void);
extern "C" PyObject* PyInit__bounded_integers(void);
extern "C" PyObject* PyInit__mt19937(void);
extern "C" PyObject* PyInit__philox(void);
extern "C" PyObject* PyInit__pcg64(void);
extern "C" PyObject* PyInit__sfc64(void);
extern "C" PyObject* PyInit__generator(void);

REGISTER_TORCH_DEPLOY_BUILTIN(
    frozen_numpy,
    _PyImport_FrozenModules_numpy,
    "numpy.core._multiarray_umath",
    PyInit__multiarray_umath,
    "numpy.core._multiarray_tests",
    PyInit__multiarray_tests,
    "numpy.linalg.lapack_lite",
    PyInit_lapack_lite,
    "numpy.linalg._umath_linalg",
    PyInit__umath_linalg,
    "numpy.fft._pocketfft_internal",
    PyInit__pocketfft_internal,
    "numpy.random.mtrand",
    PyInit_mtrand,
    "numpy.random.bit_generator",
    PyInit_bit_generator,
    "numpy.random._common",
    PyInit__common,
    "numpy.random._bounded_integers",
    PyInit__bounded_integers,
    "numpy.random._mt19937",
    PyInit__mt19937,
    "numpy.random._philox",
    PyInit__philox,
    "numpy.random._pcg64",
    PyInit__pcg64,
    "numpy.random._sfc64",
    PyInit__sfc64,
    "numpy.random._generator",
    PyInit__generator);
