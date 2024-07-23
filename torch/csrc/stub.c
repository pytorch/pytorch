#include <Python.h>

extern PyObject* initModule(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

extern "C" TORCH_API void global_kineto_init();

PyMODINIT_FUNC PyInit__C(void)
{
  PyObject* object =  initModule();
  global_kineto_init();
  return object;
}
