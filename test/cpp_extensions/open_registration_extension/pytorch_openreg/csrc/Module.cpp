#include <OpenReg.h>

// Make this a proper CPython module
static struct PyModuleDef openreg_C_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pytorch_openreg._C",
};

PyMODINIT_FUNC PyInit__C(void) {
    PyObject* mod = PyModule_Create(&openreg_C_module);

    py::object openreg_mod = py::module_::import("pytorch_openreg");
    // Only borrowed from the python side!
    openreg::set_impl_registry(openreg_mod.attr("_IMPL_REGISTRY").ptr());

    return mod;
}
