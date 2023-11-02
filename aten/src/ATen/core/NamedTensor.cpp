#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/NamedTensor.h>

#include <ATen/core/TensorBase.h>

namespace at {

thread_local bool NamesMode_enabled = true;

bool NamesMode::is_enabled() {
  return NamesMode_enabled;
}

void NamesMode::set_enabled(bool enabled) {
  NamesMode_enabled = enabled;
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::Named, !enabled);
}

const TensorBase& internal_set_names_inplace(const TensorBase& tensor, optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names, /*validate_names=*/true);
  return tensor;
}

const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::vector<Dimname>&& names, bool validate_names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), std::move(names), validate_names);
  return tensor;
}

DimnameList default_names(size_t len) {
  static std::vector<Dimname> all_unnamed(kMaxNamedTensorDim, Dimname::wildcard());
    TORCH_INTERNAL_ASSERT(
        len <= kMaxNamedTensorDim,
        "Only tensors with up to ", kMaxNamedTensorDim, " are supported.");
  return DimnameList(&all_unnamed.front(), len);
}

static void check_unique_names(DimnameList names) {
  // Strategy: Compare each element with the ones that come after it.
  // Although this is O(N^2), in practice N is small (no more than 25).
  for (auto it = names.begin(); it != names.end(); ++it) {
    if (it->isWildcard()) continue;
    auto dup = std::find(it + 1, names.end(), *it);
    while (dup != names.end()) {
      TORCH_CHECK(false,
          "Cannot construct a tensor with duplicate names. Got names: ",
          names, ".");
    }
  }
}

void check_names_valid_for(const TensorBase& tensor, DimnameList names) {
  return impl::check_names_valid_for(tensor.unsafeGetTensorImpl(), names);
}

void check_names_valid_for(size_t tensor_dim, DimnameList names) {
  TORCH_CHECK(
      tensor_dim <= kMaxNamedTensorDim,
      "Named tensors only support up to ", kMaxNamedTensorDim, " dims: "
      "Attempted to create a tensor with dim ", tensor_dim, " with names ", names);
  TORCH_CHECK(tensor_dim == names.size(),
      "Number of names (", names.size(), ") and "
      "number of dimensions in tensor (", tensor_dim, ") ",
      "do not match. Attempted to create a tensor with names ", names);
  check_unique_names(names);
}

namespace impl {

static NamedTensorMeta* get_named_tensor_meta(TensorImpl* impl) {
  if (!NamesMode::is_enabled()) {
    return nullptr;
  }
  return static_cast<NamedTensorMeta*>(impl->named_tensor_meta());
}

static const NamedTensorMeta* get_named_tensor_meta(const TensorImpl* impl) {
  if (!NamesMode::is_enabled()) {
    return nullptr;
  }
  return static_cast<const NamedTensorMeta*>(impl->named_tensor_meta());
}

void check_names_valid_for(TensorImpl* impl, DimnameList names) {
  check_names_valid_for(impl->dim(), names);
}

void internal_set_names_inplace(TensorImpl* impl, optional<DimnameList> names, bool validate_names) {
  if (!names) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  if (validate_names) {
    check_names_valid_for(impl, *names);
  }
  // Do this after validation!
  if (std::all_of(names->begin(), names->end(), [](const Dimname& n) { return n.isWildcard(); })) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    // Constructor is private
    impl->set_named_tensor_meta(std::make_unique<NamedTensorMeta>(NamedTensorMeta::HasNonWildcard, *names));
  } else {
    meta->set_names(NamedTensorMeta::HasNonWildcard, *names);
  }
}

void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names) {
  if (validate_names) {
    check_names_valid_for(impl, names);
  }
  // Do this after validation!
  if (std::all_of(names.begin(), names.end(), [](const Dimname& n) { return n.isWildcard(); })) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    impl->set_named_tensor_meta(std::make_unique<NamedTensorMeta>(NamedTensorMeta::HasNonWildcard, names));
  } else {
    meta->set_names(NamedTensorMeta::HasNonWildcard, names);
  }
}

optional<DimnameList> get_opt_names(const TensorImpl* impl) {
  const auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    return nullopt;
  } else {
    return meta->names();
  }
}

DimnameList get_names(const TensorImpl* impl) {
  auto maybe_names = get_opt_names(impl);
  if (maybe_names) {
    return *maybe_names;
  }
  return default_names(impl->dim());
}

bool has_names(const TensorImpl* impl) {
  return impl->has_named_tensor_meta() && NamesMode::is_enabled();
}

} // namespace impl

} // namespace at
