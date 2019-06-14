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

static DimnameList::const_iterator find_untagged_name(
    DimnameList::const_iterator begin,
    DimnameList::const_iterator end,
    Symbol target) {
  return std::find_if(begin, end,
      [&target](const Dimname& candidate) { return candidate.untagged_name() == target; });
}

static void check_unique_names(DimnameList names) {
  // Strategy: Compare each element with the ones that come after it.
  // Although this is O(N^2), in practice N is small (no more than 25).
  for (auto it = names.begin(); it != names.end(); ++it) {
    if (it->type() == NameType::WILDCARD) {
      continue;
    }
    auto dup = find_untagged_name(it + 1, names.end(), it->untagged_name());
    while (dup != names.end()) {
      TORCH_CHECK(it->full_name() != dup->full_name(),
          "Cannot construct a tensor with duplicate names. Got names: ",
          names, ".");

      // "C.in" and "C" are not allowed, but "C.in" and "C.out" are.
      TORCH_CHECK(it->type() == NameType::TAGGED && dup->type() == NameType::TAGGED,
          "Cannot construct a tensor with duplicate names unless they are tagged ",
          "and have different tags. Got names: ", names, ", offending names: (",
          *it, " and ", *dup, ").");
      dup = find_untagged_name(dup + 1, names.end(), it->untagged_name());
    }
  }
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
  check_unique_names(*names);

  auto* meta = tensor.get_named_tensor_meta();
  if (meta == nullptr) {
    tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
        torch::make_unique<NamedTensorMeta>(*names));
  } else {
    meta->set_names_(*names);
  }
}

} // namespace at
#endif
