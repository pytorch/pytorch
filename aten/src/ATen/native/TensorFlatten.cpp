#include <ATen/ATen.h>
#include <ATen/core/functional.h>

namespace at {
namespace native {
  Tensor flatten_dense_tensors(TensorList tensors) {
    static auto flatten = [](const Tensor &t) { return t.contiguous().view({-1}); };
    if (tensors.size() == 1)
      return flatten(tensors[0]);
    return at::cat(fmap(tensors, flatten));
  }

  std::vector<Tensor> unflatten_dense_tensors(const Tensor& flat, TensorList tensors) {
    std::vector<Tensor> outputs;
    outputs.reserve(tensors.size());
    size_t offset = 0;
    for (const auto & tensor : tensors) {
      auto numel = tensor.numel();
      // If unflatten an empty tensor, create a new empty tensor using
      // flat tensor Options.
      // This can avoid the unflattened empty tensor to share the same storage
      // with other unflatten tensors.
      if (numel == 0) {
        outputs.push_back(at::empty({0}, flat.options()));
      } else {
        outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
        offset += numel;
      }
    }
    return outputs;
  }
} // native
} // aten
