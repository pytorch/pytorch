#ifdef NAMEDTENSOR_ENABLED
#include <ATen/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/utils/memory.h>

namespace at {

bool NamedTensorMeta::has_names() const {
  return !std::all_of(
      names_.begin(), names_.end(), [](const Dimname& n) {
        return n.type() == NameType::WILDCARD;
      });
}

void internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names) {
  if (!names) {
    tensor.unsafeGetTensorImpl()->set_named_tensor_meta(nullptr);
    return;
  }

  auto ndim = tensor.dim();
  TORCH_CHECK(ndim == names->size(),
      "Number of names (", names->size(), ") and "
      "number of dimensions in tensor (", ndim, ") ",
      "do not match.");

  auto* meta = tensor.get_named_tensor_meta();
  if (meta == nullptr) {
    tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
        torch::make_unique<NamedTensorMeta>(*names));
  } else {
    meta->set_names_(*names);
  }
}

}
#endif
