#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace {
static std::string shapes_as_str(TensorList tensors) {
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
} // anonymous namespace

static std::tuple<bool, Tensor> canDispatchToMaskedFill(const Tensor& self, const torch::List<c10::optional<at::Tensor>>& indices,
const Tensor& value){
  if (!(value.numel() ==1 && value.device().is_cpu())){
    return std::make_tuple(false,Tensor());
  }
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const c10::optional<Tensor>& i: indices) {
    if (!i.has_value() || !(*i).defined()){
      num_ind++;
    } else {
      const Tensor &index = *i;
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()){
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(index.size(j) == self.size(srcIdx), "The shape of the mask ", index.sizes(), " at index ", j,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for (const auto i : c10::irange(num_ind, self.ndimension())) {
    (void)i; //Suppress unused variable warning
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}

static AdvancedIndex make_info(Tensor self, IOptTensorListRef orig) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                   " with shapes ", shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (auto & indice : indices) {
    if (indice.defined() && indice.device() != self.device()) {
      indice = indice.to(self.device());
    }
  }
  for (auto & indice : indices) {
    if (indice.defined() && indice.dtype() == at::kInt) {
      indice = indice.to(at::kLong);
    }
  }

  return AdvancedIndex(self, indices);
}

} // at
} // native
