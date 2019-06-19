#ifdef NAMEDTENSOR_ENABLED

#include <ATen/NamedTensorUtils.h>
#include <sstream>

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

// Returns "Tensor['N', 'C', 'H', 'W']" for a tensor with names ('N', 'C', 'H', 'W').
static std::string toDimnameRepr(const Tensor& tensor) {
  std::ostringstream os;
  os << "Tensor";
  if (tensor.names() == nullopt) {
    os << "[";
    for (auto i = 0; i < tensor.dim(); i++) {
      if (i != 0) os << ", ";
      os << "None";
    }
    os << "]";
  } else {
    os << *tensor.names();
  }
  return os.str();
}

int64_t dimname_to_position(const Tensor& tensor, Dimname dim) {
  TORCH_CHECK(dim.type() != NameType::WILDCARD,
      "Please look up dimensions by name, got: name = None.");
  TORCH_CHECK(tensor.names().has_value(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");
  const auto names = *tensor.names();

  const auto it = std::find_if(
      names.begin(), names.end(),
      [&dim](const Dimname& candidate) { return dim.can_refer_to(candidate); });
  TORCH_CHECK(it != names.end(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");

  // Check that it can't refer to another dimension
  const auto dup = std::find_if(
      it + 1, names.end(),
      [&dim](const Dimname& candidate) { return dim.can_refer_to(candidate); });
  TORCH_CHECK(
      dup == names.end(),
      "Name ", dim, " could refer to multiple dimensions in ",
      toDimnameRepr(tensor), ". Please disambiguate by using a more ",
      "specific name like ", *it, " or ", dup, ".");
  return std::distance(names.begin(), it);
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
