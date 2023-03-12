#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <ATen/core/TensorBase.h>
#include <ATen/core/copy_on_write.h>

#include <algorithm>
#include <functional>

namespace torch::autograd {

void materialize_copies_on_write(
    c10::ArrayRef<std::reference_wrapper<at::TensorBase const>> tensors) {
  std::for_each(tensors.begin(), tensors.end(), at::materialize_copy_on_write);
}

} // namespace torch::autograd
