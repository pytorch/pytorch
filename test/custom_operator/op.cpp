#include <torch/op.h>

#if defined(WITH_PYTHON_OP_BINDINGS)
#include <pybind11/pybind11.h>
#endif

#include <cstddef>
#include <vector>

std::vector<at::Tensor> custom_op(
    at::Tensor tensor,
    double scalar,
    int64_t repeat) {
  std::vector<at::Tensor> output;
  output.reserve(repeat);
  for (int64_t i = 0; i < repeat; ++i) {
    output.push_back(tensor * scalar);
  }
  return output;
}

torch::RegisterOperators registry("custom::op", &custom_op);

#if defined(WITH_PYTHON_OP_BINDINGS)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = R"doc(
    Python module to register custom::op from Python
    ------------------------------------------------
    Importing this module will register custom::op with PyTorch.
    The actual Python bindings to the operator are created automatically the
    first time you access the operator, which you can do like this::

    >>> import torch
    >>> output = torch.ops.custom.op(torch.ones(2), 2.0, 3)
  )doc";
}
#endif
