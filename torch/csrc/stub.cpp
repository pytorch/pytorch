#include <Python.h>

#ifdef _WIN32
__declspec(dllimport)
#endif
extern PyObject* initModule();

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C()
{
  initModule();
}
#else
PyMODINIT_FUNC PyInit__C()
{
  return initModule();
}
#endif
