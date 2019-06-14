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

} // namespace at
#endif
