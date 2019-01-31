#include <Python.h>

extern PyObject* initNpModule();
#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_np_compat()
{
    initNpModule();
}
#else
PyMODINIT_FUNC PyInit__np_compat()
{
    return initNpModule();
}
#endif