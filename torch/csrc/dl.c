#include <Python.h>
#include <dlfcn.h>

PyObject* module;

static PyMethodDef TorchDlMethods[] = {
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef torchdlmodule = {
   PyModuleDef_HEAD_INIT,
   "torch._dl",
   NULL,
   -1,
   TorchDlMethods
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
