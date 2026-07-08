#pragma once
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {
namespace {
#ifndef STRIP_ERROR_MESSAGES
inline std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}
#endif
} // anonymous namespace

inline std::tuple<bool, Tensor> canDispatchToMaskedFill(
    const Tensor& self,
    const torch::List<std::optional<at::Tensor>>& indices,
    const Tensor& value) {
  if (!(value.numel() == 1 && value.device().is_cpu())) {
    return std::make_tuple(false, Tensor());
  }
  return at::indexing::impl::canDispatchToMaskedFill(self, indices);
}

inline AdvancedIndex make_info(Tensor self, IOptTensorListRef orig) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = expandTensors(self, orig, /*ensure_same_device=*/true);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception&) {
    TORCH_CHECK_INDEX(
        false,
        "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ",
        shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  indices.reserve(self.dim());
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  for (auto& indice : indices) {
    if (indice.defined() && indice.dtype() == at::kInt) {
      indice = indice.to(at::kLong);
    }
  }

  return AdvancedIndex(self, indices);
}

} // namespace at::native
