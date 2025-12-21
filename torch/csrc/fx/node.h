#pragma once

#include <torch/csrc/python_headers.h>

bool NodeBase_init(PyObject* module);
bool NodeIter_init(PyObject* module);

// Fast C++ accessors for Node attributes - returns borrowed references
// These avoid the overhead of PyObject_GetAttrString by accessing struct
// members directly Returns nullptr if node is not a NodeBase instance
PyObject* NodeBase_borrow_op(PyObject* node);
PyObject* NodeBase_borrow_target(PyObject* node);
PyObject* NodeBase_borrow_name(PyObject* node);
PyObject* NodeBase_borrow_graph(PyObject* node);

// Check if a PyObject is a NodeBase instance
bool NodeBase_Check(PyObject* obj);
