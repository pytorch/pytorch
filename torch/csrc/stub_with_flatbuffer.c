#include <Python.h>  // NOLINT

#ifdef _WIN32
__declspec(dllimport)
#endif
extern PyObject* initModuleFlatbuffer(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C_flatbuffer(void);
#endif

PyMODINIT_FUNC PyInit__C_flatbuffer(void)
{
  return initModuleFlatbuffer();
}
