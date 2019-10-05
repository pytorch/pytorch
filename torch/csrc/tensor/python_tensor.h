#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorTypeId.h>

namespace c10 {
struct Device;
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

// Gets the TensorTypeId for the default tensor type.
//
// TODO: This is nuts!  There is no reason to let the default tensor type id
// change.  Probably only store ScalarType, as that's the only flex point
// we support.
c10::TensorTypeId get_default_tensor_type_id();

// Gets the ScalarType for the default tensor type.
at::ScalarType get_default_scalar_type();
}} // namespace torch::tensors
