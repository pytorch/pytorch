#include <Python.h>

#include <array>

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init__global_deps() {}
#else
static std::array<PyMethodDef, 1> global_deps_methods = {
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef global_deps_def = {
    PyModuleDef_HEAD_INIT,
    "_global_deps",
    "",
    -1,
    global_deps_methods.data()
};

PyMODINIT_FUNC PyInit__global_deps() {
    return PyModule_Create(&global_deps_def);
}
#endif

