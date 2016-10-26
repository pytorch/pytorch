#ifndef THP_CUDNN_MODULE_INC
#define THP_CUDNN_MODULE_INC

#include <Python.h>

PyMethodDef* THCUDNN_methods();
bool THCUDNNModule_initModule(PyObject *self);

#endif
