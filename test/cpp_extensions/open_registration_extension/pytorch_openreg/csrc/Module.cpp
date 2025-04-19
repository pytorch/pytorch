#include <OpenReg.h>

// Make this a proper CPython module
static struct PyModuleDef openreg_C_module = {
    PyModuleDef_HEAD_INIT,
    "pytorch_openreg._C",
    nullptr,
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit__C(void) {
    PyObject* mod = PyModule_Create(&openreg_C_module);

    py::object openreg_mod = py::module_::import("pytorch_openreg");
    // Only borrowed from the python side!
    openreg::set_impl_factory(openreg_mod.attr("impl_factory").ptr());

    return mod;
}
