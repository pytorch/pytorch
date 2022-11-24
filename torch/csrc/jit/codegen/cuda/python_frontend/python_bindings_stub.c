#include <Python.h>

#ifdef _WIN32
__declspec(dllimport)
#endif
extern PyObject* initModuleNvfuser(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C_nvfuser(void);
#endif

PyMODINIT_FUNC PyInit__C_nvfuser(void)
{
  return initModuleNvfuser();
}
