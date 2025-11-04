#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/library.h>

TORCH_LIBRARY_IMPL(_, DTensor, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&callDTensorOpDispatch>());
}
