#if defined(_MSC_VER)
#  pragma push_macro("_DEBUG")
#  if defined(_DEBUG) && !defined(Py_DEBUG)
#    undef _DEBUG
#  endif
#endif

#include <Python.h>

#if defined(_MSC_VER)
#  pragma pop("_DEBUG")
#endif

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
