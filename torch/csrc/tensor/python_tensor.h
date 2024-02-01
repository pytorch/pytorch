#pragma once

#include <c10/core/Backend.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/python_headers.h>

namespace at {
class Tensor;
} // namespace at

namespace torch {
namespace tensors {

// Initializes the Python tensor type objects: torch.FloatTensor,
// torch.DoubleTensor, etc. and binds them in their containing modules.
// Pytorch side only register CPU and GPU tensor types, if a third-party device
// also need to register thire tensor types in torch._tensor_classes, this
// function can be called at third-party framework.
void initialize_python_bindings(
    const std::vector<std::pair<c10::Backend, c10::ScalarType>>&
        declared_types);

// Same as set_default_tensor_type() but takes a PyObject*
void py_set_default_tensor_type(PyObject* type_obj);

// Same as py_set_default_tensor_type, but only changes the dtype (ScalarType).
void py_set_default_dtype(PyObject* dtype_obj);

// Gets the DispatchKey for the default tensor type.
//
// TODO: This is nuts!  There is no reason to let the default tensor type id
// change.  Probably only store ScalarType, as that's the only flex point
// we support.
TORCH_API c10::DispatchKey get_default_dispatch_key();
at::Device get_default_device();

// Gets the ScalarType for the default tensor type.
at::ScalarType get_default_scalar_type();
} // namespace tensors
} // namespace torch
