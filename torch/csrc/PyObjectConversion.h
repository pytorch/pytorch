#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/python_stub.h>
#include <torch/csrc/Export.h>

// Indirection that lets the libtorch-only Python-interop stable shims call into
// code that only libtorch_python can provide (THPVariable_* &co) without
// libtorch (or the user extension) linking libtorch_python. Today this backs
// PyObject <-> Tensor conversion; other conversions between Python objects and
// libtorch types that need libtorch_python can be added as further methods.
//
// This mirrors c10's PyInterpreterVTable: an abstract interface declared in the
// lower library, a no-op default that errors, and a concrete implementation
// registered by libtorch_python at load time. Unlike PyInterpreterVTable this
// is a single process-global (there is no tagged tensor to route through when
// converting a raw PyObject*), so it assumes a single Python interpreter.
//
// How the pieces fit together:
//   1. torch/csrc/PyObjectConversion.h (this file) - the vtable interface; in
//      libtorch.
//   2. torch/csrc/PyObjectConversion.cpp - a NoopPyObjectConversion that errors
//      by default; in libtorch. This fallback fires when libtorch_python was
//      never loaded to register a real implementation.
//   3. torch/csrc/PyObjectConversionPythonImpl.cpp - the concrete impl using
//      THPVariable_*; in libtorch_python, registers itself at load.
//   4. torch/csrc/shim_common.cpp - the stable C shims that call through the
//      vtable; in libtorch.

namespace torch::detail {

struct TORCH_API PyObjectConversionInterface {
  virtual ~PyObjectConversionInterface() = default;

  // Unpack a Python torch.Tensor (PyObject*) into an at::Tensor that shares the
  // underlying TensorImpl. The GIL must be held.
  virtual at::Tensor tensor_from_pyobject(PyObject* obj) const = 0;

  // Wrap an at::Tensor as a new-reference Python torch.Tensor. py_type, if
  // non-null, is the result's exact PyTypeObject* (e.g. torch.nn.Parameter);
  // null means the default torch.Tensor type. The GIL must be held.
  virtual PyObject* tensor_to_pyobject(const at::Tensor& t, PyObject* py_type)
      const = 0;
};

// Install the conversion implementation. Called once by libtorch_python when it
// is loaded. Passing nullptr restores the default no-op implementation.
TORCH_API void setPyObjectConversionImpl(
    const PyObjectConversionInterface* impl);

// Return the registered implementation. Before libtorch_python registers one
// this is a no-op that raises a clear error when its methods are called.
TORCH_API const PyObjectConversionInterface& getPyObjectConversionImpl();

} // namespace torch::detail
