#include <Python.h>
#include <dlfcn.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* module;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef TorchDlMethods[] = {
  {NULL, NULL, 0, NULL}
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static struct PyModuleDef torchdlmodule = {
    PyModuleDef_HEAD_INIT,
    "torch._dl",
    NULL,
    -1,
    TorchDlMethods
    // NOLINTNEXTLINE(clang-diagnostic-missing-field-initializers)
};

PyMODINIT_FUNC PyInit__dl(void)
{

#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL

  ASSERT_TRUE(module = PyModule_Create(&torchdlmodule));
  ASSERT_TRUE(PyModule_AddIntConstant(module, "RTLD_GLOBAL", (int64_t) RTLD_GLOBAL) == 0);
  ASSERT_TRUE(PyModule_AddIntConstant(module, "RTLD_NOW", (int64_t) RTLD_NOW) == 0);
  ASSERT_TRUE(PyModule_AddIntConstant(module, "RTLD_LAZY", (int64_t) RTLD_LAZY) == 0);

  return module;

#undef ASSERT_TRUE
}
