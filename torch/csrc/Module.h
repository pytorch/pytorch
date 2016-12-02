#ifndef THP_MODULE_INC
#define THP_MODULE_INC

#define THP_STATELESS_ATTRIBUTE_NAME "_torch"

extern PyObject *THPDefaultTensorClass;
extern THPGenerator *THPDefaultGenerator;

#ifdef _THP_CORE
bool THPModule_isTensor(PyObject *obj);
#endif

#endif
