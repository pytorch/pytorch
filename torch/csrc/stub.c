#include <Python.h>

#ifdef _WIN32
__declspec(dllimport)
#endif
extern PyObject* initModule(void);

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C(void)
{
  initModule();
}
#else
PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
#endif
