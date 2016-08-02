#ifndef THP_MODULE_INC
#define THP_MODULE_INC

extern THPGenerator *THPDefaultGenerator;
bool THPModule_tensorCopy(PyObject *dst, PyObject *src);

#define STATELESS_ATTRIBUTE_NAME "_torch"

#endif
