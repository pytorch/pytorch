#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/core/TensorImpl.h>

namespace torch { namespace autograd {

// A PythonTorchDispatchTypeObject represents the type of a Tensor subclass that has
// a __torch_dispatch__ classmethod. Concretely, it holds the class as a
// PyObject* and a PyInterpreter* that says which python interpreter the class
// came from.
struct TORCH_API PythonTorchDispatchTypeObject : public c10::TorchDispatchTypeObject {
  // PythonTorchDispatchTypeObject takes a reference to the torch_dispatch_type_object.
  // It does this in case the user decides to delete the original type object.
  PythonTorchDispatchTypeObject(
      PyObject* torch_dispatch_type_object,
      c10::impl::PyInterpreter* pyinterpreter);
  ~PythonTorchDispatchTypeObject() override;
  PyObject* ptr() const;
  c10::impl::PyInterpreter* pyinterpreter() const override;

 private:
  PyObject* data_ = nullptr;
  c10::impl::PyInterpreter* pyinterpreter_ = nullptr;
};

struct TORCH_API PythonMode {
  // Enter python mode, causing all operators to dispatch to the type's __torch_dispatch__.
  // `type` is the type of a Tensor subclass that has __torch_dispatch__.
  static void enter(PyObject* type);

  // Exit the current python mode.
  static void exit();
};

}}
