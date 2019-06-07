#ifndef THP_PTR_WRAPPER_H
#define THP_PTR_WRAPPER_H

#include <torch/csrc/python_headers.h>

/**
 * Python wrapper around arbitrary opaque C++ class
 */

bool THPWrapper_init(PyObject *module);

PyObject * THPWrapper_New(void *data, void (*destructor)(void*));
void * THPWrapper_get(PyObject * obj);
bool THPWrapper_check(PyObject * obj);

#endif
