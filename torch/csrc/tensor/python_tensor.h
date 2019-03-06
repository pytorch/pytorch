#pragma once

#include <torch/csrc/python_headers.h>
#include <type_traits>

namespace c10 {
struct Device;
}

namespace at {
struct Type;
class Tensor;
} // namespace at

struct THPDtype;
struct THPLayout;

namespace torch { namespace tensors {

struct PyTensorType {
  PyTypeObject py_type;
  at::Type* aten_type_;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_cuda;
  char name[64];
  int backend;
  int scalar_type;

  // Precondition: Access to this struct is protected by the GIL
  at::Type* aten_type();
};

static_assert(std::is_standard_layout<PyTensorType>::value, "PyTensorType must be standard layout");

// Initializes the Python tensor type objects: torch.FloatTensor,
// torch.DoubleTensor, etc. and binds them in their containing modules.
void initialize_python_bindings();

// Sets the concrete type constructed by calls to torch.Tensor() and most
// factory methods on the torch module.
void set_default_tensor_type(PyTensorType& type);

// Same as set_default_tensor_type() but takes a PyObject*
void py_set_default_tensor_type(PyObject* type_obj);

// Same as py_set_default_tensor_type, but only changes the dtype (ScalarType).
void py_set_default_dtype(PyObject* dtype_obj);

// Gets the ATen type object for the default tensor type. Note that the
// returned value will be a PyTensorType instance.
PyTensorType& get_default_tensor_type();

}} // namespace torch::tensors
