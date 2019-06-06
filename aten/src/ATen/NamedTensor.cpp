#ifdef NAMEDTENSOR_ENABLED
#include <ATen/NamedTensor.h>

namespace at {

bool NamedTensorMeta::has_names() const {
  return !std::all_of(
      names.begin(), names.end(), [](const Dimname& n) {
        return n.type() == NameType::WILDCARD;
      });
}

}
#endif
