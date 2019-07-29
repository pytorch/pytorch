#ifdef BUILD_NAMEDTENSOR

#include <ATen/NamedTensorUtils.h>
#include <bitset>
#include <sstream>

namespace at {

Tensor& internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names);
  return tensor;
}

Tensor& internal_set_names_inplace(Tensor& tensor, std::vector<Dimname>&& names, bool validate_names) {
#ifdef DEBUG
  validate_names = true;
#endif
  impl::internal_set_names_inplace(
      tensor.unsafeGetTensorImpl(), std::move(names), validate_names);
  return tensor;
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

std::vector<int64_t> dimnames_to_positions(const Tensor& tensor, DimnameList dims) {
  std::vector<int64_t> result;
  result.reserve(dims.size());
  for (const auto& name : dims) {
    result.push_back(dimname_to_position(tensor, name));
  }
  return result;
}

static void report_positional_error(
    const Dimname& name,
    const Dimname& other_name,
    DimnameList names,
    DimnameList other_names) {
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(false,
      "Names ", name, " and ", other_name, " do not match positionally ",
      "from the right in names ", names, " and ", other_names, ".");
}

static void check_for_misalignment(
    const Dimname& name,
    DimnameList names,
    DimnameList other_names) {
  if (name.is_wildcard()) {
    return;
  }
  auto it = std::find_if(other_names.begin(), other_names.end(),
      [&](const Dimname& candidate) { return name.can_refer_to(candidate); });
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(it == other_names.end(),
      "Names ", names, " and ", other_names, " are misaligned: name ", name,
      " appears in a different position from the right.");
}

// Assumption: A DimnameList can have no duplicate full names with
// the exception of wildcards
static std::vector<Dimname> unify_from_right(DimnameList names, DimnameList other_names) {
  const auto wildcard = Dimname::wildcard();
  const auto size = std::max(names.size(), other_names.size());
  auto result = std::vector<Dimname>(size, wildcard);

  auto names_it = names.rbegin();
  auto other_it = other_names.rbegin();
  auto result_it = result.rbegin();
  while (names_it != names.rend() || other_it != other_names.rend()) {
    // TODO(zou3519): Don't support tagged names for now. They're a little weird.
    if (names_it->is_tagged() || other_it->is_tagged()) {
      TORCH_INTERNAL_ASSERT("unify_from_right: NYI: tagged names.");
    }

    const auto& name = names_it == names.rend() ? wildcard : *names_it;
    const auto& other_name = other_it == other_names.rend() ? wildcard : *other_it;

    // Step 1: Check that the names match
    const auto maybeName = unify(name, other_name);
    if (!maybeName) {
      report_positional_error(name, other_name, names, other_names);
    }
    *result_it = *maybeName;

    // Step 2: Check that the names are not misaligned
    if (!names_it->is_normal() || !other_it->is_normal()) {
      // Let: N = max(len(names), len(other_names))
      //      K = # of special names among names and other_names.
      // This search (including the outer loop) is O(N*K) but typically # of dims is small.
      check_for_misalignment(name, names, other_names);
      check_for_misalignment(other_name, other_names, names);
    }

    if (names_it != names.rend()) {
      ++names_it;
    }
    if (other_it != other_names.rend()) {
      ++other_it;
    }
    ++result_it;
  }
  return result;
}

// Assumption: A DimnameList can have no duplicate full names with
// the exception of wildcards
CAFFE2_API optional<std::vector<Dimname>>
unify_from_right(optional<DimnameList> names, optional<DimnameList> other_names) {
  if (!names && !other_names) {
    return nullopt;
  }
  if (!names) {
    return other_names.value().vec();
  }
  if (!other_names) {
    return names.value().vec();
  }
  return unify_from_right(*names, *other_names);
}


namespace namedinference {

static std::bitset<64> compute_included_idxs(IntArrayRef excluded_idxs) {
  std::bitset<64> included_idxs;
  for (auto i : excluded_idxs) {
    TORCH_INTERNAL_ASSERT(
        i < 64,
        "Named tensors must have dimension less than 64.");
    included_idxs.set(i);
  }
  included_idxs.flip();
  return included_idxs;
}

void propagate_names_except(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs) {
  auto src_names = src.names();
  if (!src_names.has_value()) {
    return;
  }

  auto result_dim = result.dim();
  auto src_dim = src_names->size();
  TORCH_INTERNAL_ASSERT(src_dim - excluded_idxs.size() == result_dim);

  // fast path
  if (excluded_idxs.size() == 1) {
    std::vector<Dimname> outnames = src_names->vec();
    outnames.erase(outnames.begin() + excluded_idxs[0]);
    internal_set_names_inplace(result, std::move(outnames), /*validate_names=*/false);
    return;
  }

  std::vector<Dimname> outnames;
  outnames.reserve(result_dim);
  auto included_idxs = compute_included_idxs(excluded_idxs);
  for (size_t dim = 0; dim < src_dim; ++dim) {
    if (included_idxs[dim]) {
      outnames.push_back((*src_names)[dim]);
    }
  }
  internal_set_names_inplace(result, std::move(outnames), /*validate_names=*/false);
}

void propagate_names_for_reduction(Tensor& result, const Tensor& src, IntArrayRef reduced_dims, bool keepdim) {
  if (keepdim) {
    propagate_names(result, src);
    return;
  }
  // This actually means "full reduction"
  if (reduced_dims.size() == 0) {
    return;
  }
  propagate_names_except(result, src, reduced_dims);
}

void propagate_names(Tensor& result, const Tensor& src) {
  at::internal_set_names_inplace(result, src.names());
}

void propagate_names(TensorImpl* result, TensorImpl* src) {
  const auto names = at::impl::internal_get_names(src);
  at::impl::internal_set_names_inplace(result, names);
}

} // namespace namedinference
} // namespace at
#endif
