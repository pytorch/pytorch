#pragma once

#include <c10/util/python_stub.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// Indirection that lets the libtorch-only PyObject<->Tensor stable shims call
// into code that only libtorch_python can provide (THPVariable_* &co) without
// libtorch (or the user extension) linking libtorch_python.
//
// This mirrors c10's PyInterpreterVTable: an abstract interface declared in the
// lower library, a no-op default that errors, and a concrete implementation
// registered by libtorch_python at load time. Unlike PyInterpreterVTable this
// is a single process-global (there is no tagged tensor to route through when
// converting a raw PyObject*), so it assumes a single Python interpreter.

namespace torch::detail {

struct TORCH_API PyObjectConversionInterface {
  virtual ~PyObjectConversionInterface() = default;

  // Wrap a Python torch.Tensor (PyObject*) as a new owning AtenTensorHandle
  // that shares the underlying TensorImpl. The GIL must be held.
  virtual AtenTensorHandle from_pyobject(PyObject* obj) const = 0;

  // Wrap an AtenTensorHandle as a new-reference Python torch.Tensor. py_type,
  // if non-null, is the result's exact PyTypeObject* (e.g. torch.nn.Parameter);
  // null means the default torch.Tensor type. The GIL must be held.
  virtual PyObject* to_pyobject(AtenTensorHandle ath, PyObject* py_type)
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
