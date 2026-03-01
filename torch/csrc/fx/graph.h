#pragma once

#include <torch/csrc/python_headers.h>

bool Namespace_init(PyObject* module);
bool FindNodesLookupTable_init(PyObject* module);
bool NodeList_init(PyObject* module);
bool GraphBase_init(PyObject* module);

// C++ API for direct calls from node.cpp
// These allow node.cpp to call into the lookup table without going through
// Python
bool FindNodesLookupTable_contains_impl(PyObject* lookup_table, PyObject* node);
bool FindNodesLookupTable_remove_impl(PyObject* lookup_table, PyObject* node);
bool FindNodesLookupTable_insert_impl(PyObject* lookup_table, PyObject* node);

// Fast C++ accessors for GraphBase attributes - returns borrowed references
// These avoid the overhead of PyObject_GetAttrString by accessing struct
// members directly. Returns nullptr if graph is not a GraphBase instance.
PyObject* GraphBase_borrow_owning_module(PyObject* graph);
PyObject* GraphBase_borrow_find_nodes_lookup_table(PyObject* graph);

// Check if a PyObject is a GraphBase instance
bool GraphBase_Check(PyObject* obj);
