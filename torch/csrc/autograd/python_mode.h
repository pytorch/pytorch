#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/core/TensorImpl.h>

namespace torch { namespace autograd {

struct TORCH_API PythonMode {
  // Enter python mode, causing all operators to dispatch to the type's __torch_dispatch__.
  // `type` is the type of a Tensor subclass that has __torch_dispatch__.
  static void enter(PyObject* type);

  // Exit the current python mode.
  static void exit();
};

}}
