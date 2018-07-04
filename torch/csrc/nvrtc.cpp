#include "torch/csrc/python_headers.h"

static PyObject* module;

static PyMethodDef TorchNvrtcMethods[] = {
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION != 2
static struct PyModuleDef torchnvrtcmodule = {
   PyModuleDef_HEAD_INIT,
   "torch._nvrtc",
   NULL,
   -1,
   TorchNvrtcMethods
};
#endif

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_nvrtc(void)
#else
PyMODINIT_FUNC PyInit__nvrtc(void)
#endif
{

#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error in torch._nvrtc"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._nvrtc", TorchNvrtcMethods));
#else
  ASSERT_TRUE(module = PyModule_Create(&torchnvrtcmodule));
#endif

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif

#undef ASSERT_TRUE
}
