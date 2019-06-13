#ifdef NAMEDTENSOR_ENABLED

#include <ATen/NamedTensorUtils.h>

namespace at {
namespace namedinference {

optional<std::vector<Dimname>> reduction_op(optional<DimnameList> self_names, int64_t dim) {
  if (self_names == nullopt) {
    return nullopt;
  }
  auto outnames = self_names->vec();
  outnames.erase(outnames.begin() + dim);
  return outnames;
}

} // namespace namedinference
} // namespace at
#endif
