#include <Python.h>

#ifdef _WIN32
__declspec(dllimport)
#endif
extern PyObject* initModule();

PyMODINIT_FUNC PyInit__C()
{
  return initModule();
}
