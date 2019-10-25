#include <Python.h>

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init__global_deps()
{
}
#else
static PyMethodDef global_deps_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef global_deps_def = {
    PyModuleDef_HEAD_INIT,
    "_global_deps",
    "",
    -1,
    global_deps_methods
};

PyMODINIT_FUNC PyInit__global_deps() {
    return PyModule_Create(&global_deps_def);
}
#endif

