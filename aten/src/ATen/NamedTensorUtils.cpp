#ifdef NAMEDTENSOR_ENABLED

#include <ATen/NamedTensorUtils.h>

namespace at {

// Two Dimnames cannot be in the same Tensor if one of them can refer to the other.
// In practice, this constraint means that a Tensor cannot have duplicate names
// unless they are tagged and the tags are different.
static DimnameList::const_iterator find_incompatible_name(
    DimnameList::const_iterator begin,
    DimnameList::const_iterator end,
    const Dimname& target) {
  return std::find_if(begin, end,
      [&target](const Dimname& candidate) {
        return target.can_refer_to(candidate) || candidate.can_refer_to(target);
      });
}

static void check_unique_names(DimnameList names) {
  // Strategy: Compare each element with the ones that come after it.
  // Although this is O(N^2), in practice N is small (no more than 25).
  for (auto it = names.begin(); it != names.end(); ++it) {
    auto dup = find_incompatible_name(it + 1, names.end(), *it);
    while (dup != names.end()) {
      // Simple error message if you're not using tags
      TORCH_CHECK(it->type() == NameType::TAGGED || dup->type() == NameType::TAGGED,
          "Cannot construct a tensor with duplicate names. Got names: ",
          names, ".");

      // Complicated error message if you're using tags
      TORCH_CHECK(false,
          "Cannot construct a tensor with duplicate names unless they are tagged ",
          "and have different tags. Got names: ", names, ", offending names: (",
          *it, " and ", *dup, ").");
      dup = find_incompatible_name(dup + 1, names.end(), *it);
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


namespace namedinference {

optional<std::vector<Dimname>> erase_name(optional<DimnameList> self_names, int64_t dim) {
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
