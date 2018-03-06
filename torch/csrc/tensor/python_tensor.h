#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace tensor {

// Initializes the Python tensor type objects: torch.Tensor, torch.FloatTensor,
// etc. and binds them in their containing modules.
void initialize_python_bindings();

// Sets the concrete type constructed by calls to torch.Tensor() and most
// factory methods on the torch module.
void set_default_tensor_type(const at::Type& type);

// Same as set_default_tensor_type() but takes a PyObject*
void py_set_default_tensor_type(PyObject* type_obj);

// Gets the ATen type object for the default tensor type. Note that the
// returned value will be a VariableType instance.
at::Type& get_default_tensor_type();

}} // namespace torch::tensor
