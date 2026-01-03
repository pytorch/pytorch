#include <Python.h>

#ifdef _WIN32
#define OPENREG_EXPORT __declspec(dllexport)
#else
#define OPENREG_EXPORT __attribute__((visibility("default")))
#endif

extern OPENREG_EXPORT PyObject* initOpenRegModule(void);

#ifdef __cplusplus
extern "C"
#endif

    OPENREG_EXPORT PyObject*
    PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initOpenRegModule();
}
