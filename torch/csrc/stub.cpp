#include <Python.h>

#ifdef _WIN32
__declspec(dllimport)
#endif
extern PyObject* initModule();

#ifndef _WIN32
extern "C" __attribute__((visibility("default"))) PyObject* PyInit__C();
#endif

PyMODINIT_FUNC PyInit__C()
{
  return initModule();
}
