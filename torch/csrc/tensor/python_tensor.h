#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/core/ScalarType.h>

namespace c10 {
struct Device;
class TensorOptions;
}

namespace at {
class Tensor;
} // namespace at

namespace torch { namespace tensors {

// Initializes the Python tensor type objects: torch.FloatTensor,
// torch.DoubleTensor, etc. and binds them in their containing modules.
void initialize_python_bindings();

// Same as set_default_tensor_type() but takes a PyObject*
void py_set_default_tensor_type(PyObject* type_obj);

// Same as py_set_default_tensor_type, but only changes the dtype (ScalarType).
void py_set_default_dtype(PyObject* dtype_obj);

// Gets the ATen type object for the default tensor type. Note that the
// returned value will be a VariableType instance.
c10::TensorOptions get_default_tensor_options();

}} // namespace torch::tensors
