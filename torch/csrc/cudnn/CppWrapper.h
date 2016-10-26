#ifndef THP_CUDNN_CPP_WRAPPER_INC
#define THP_CUDNN_CPP_WRAPPER_INC

#include <functional>

/**
 * Python wrapper around arbitrary opaque C++ class
 */

bool THPWrapper_init(PyObject *module);

PyObject * THPWrapper_New(void *data, void (*destructor)(void*));
void * THPWrapper_get(PyObject * obj);
bool THPWrapper_check(PyObject * obj);

#endif
