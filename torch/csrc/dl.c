#include <Python.h>
#include <dlfcn.h>

PyObject* module;

static PyMethodDef TorchDlMethods[] = {
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION != 2
static struct PyModuleDef torchdlmodule = {
   PyModuleDef_HEAD_INIT,
   "torch._dl",
   NULL,
   -1,
   TorchDlMethods
};
#endif

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_dl(void)
#else
PyMODINIT_FUNC PyInit__dl(void)
#endif
{

#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._dl", TorchDlMethods));
#else
  ASSERT_TRUE(module = PyModule_Create(&torchdlmodule));
#endif
  ASSERT_TRUE(PyModule_AddIntConstant(module, "RTLD_GLOBAL", (long) RTLD_GLOBAL) == 0);
  ASSERT_TRUE(PyModule_AddIntConstant(module, "RTLD_NOW", (long) RTLD_NOW) == 0);
  ASSERT_TRUE(PyModule_AddIntConstant(module, "RTLD_LAZY", (long) RTLD_LAZY) == 0);

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif

#undef ASSERT_TRUE
}
