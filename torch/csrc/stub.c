#include <Python.h>

extern PyObject* initModule(void);
extern void init_THPCaches();

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

PyMODINIT_FUNC PyInit__C(void)
{
  init_THPCaches();
  return initModule();
}
