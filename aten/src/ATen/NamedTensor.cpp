#ifdef BUILD_NAMEDTENSOR
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

Tensor& internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names);
  return tensor;
}

Tensor& internal_set_names_inplace(Tensor& tensor, std::vector<Dimname>&& names, bool validate_names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), std::move(names), validate_names);
  return tensor;
}

std::vector<Dimname> FIXME_default_names(size_t len) {
  return { len, Dimname::wildcard() };
}

namespace impl {

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

static NamedTensorMeta* get_named_tensor_meta(TensorImpl* impl) {
  return static_cast<NamedTensorMeta*>(impl->named_tensor_meta());
}

void check_valid_names(TensorImpl* impl, DimnameList names) {
  auto ndim = impl->dim();
  TORCH_CHECK(ndim == names.size(),
      "Number of names (", names.size(), ") and "
      "number of dimensions in tensor (", ndim, ") ",
      "do not match.");
  check_unique_names(names);
}

void internal_set_names_inplace(TensorImpl* impl, optional<DimnameList> names) {
  if (!names) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  check_valid_names(impl, *names);
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    impl->set_named_tensor_meta(torch::make_unique<NamedTensorMeta>(*names));
  } else {
    meta->set_names_(*names);
  }
}

void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names) {
  if (validate_names) {
    check_valid_names(impl, names);
  }
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    impl->set_named_tensor_meta(torch::make_unique<NamedTensorMeta>(names));
  } else {
    meta->set_names_(names);
  }
}

optional<DimnameList> get_names(TensorImpl* impl) {
  const auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    return nullopt;
  } else {
    return meta->names();
  }
}

bool has_names(TensorImpl* impl) {
  const auto* named_tensor_meta = get_named_tensor_meta(impl);
  return named_tensor_meta != nullptr && named_tensor_meta->has_names();
}

} // namespace impl

} // namespace at
#endif
