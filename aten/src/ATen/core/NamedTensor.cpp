#include <ATen/core/NamedTensor.h>
#include <ATen/core/EnableNamedTensor.h>

#ifdef BUILD_NAMEDTENSOR
#include <ATen/core/Tensor.h>
#include <c10/util/C++17.h>

namespace at {

bool NamedTensorMeta::has_names() const {
  return !std::all_of(
      names_.begin(), names_.end(), [](const Dimname& n) {
        return n.type() == NameType::WILDCARD;
      });
}

thread_local bool NamesMode_enabled = true;

bool NamesMode::is_enabled() {
  return NamesMode_enabled;
}

void NamesMode::set_enabled(bool enabled) {
   NamesMode_enabled = enabled;
}

Tensor& internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names);
  return tensor;
}

Tensor& internal_set_names_inplace(Tensor& tensor, std::vector<Dimname>&& names, bool validate_names) {
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

void check_names_valid_for(const Tensor& tensor, DimnameList names) {
  return impl::check_names_valid_for(tensor.unsafeGetTensorImpl(), names);
}

namespace impl {

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
  auto ndim = impl->dim();
  TORCH_CHECK(
      ndim <= kMaxNamedTensorDim,
      "Named tensors only support up to ", kMaxNamedTensorDim, " dims: "
      "Attempted to create a tensor with dim ", ndim, " with names ", names);
  TORCH_CHECK(ndim == names.size(),
      "Number of names (", names.size(), ") and "
      "number of dimensions in tensor (", ndim, ") ",
      "do not match. Attempted to create a tensor with names ", names);
  check_unique_names(names);
}

void internal_set_names_inplace(TensorImpl* impl, optional<DimnameList> names) {
  if (!names) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  check_names_valid_for(impl, *names);
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    impl->set_named_tensor_meta(c10::guts::make_unique<NamedTensorMeta>(*names));
  } else {
    meta->set_names(*names);
  }
}

void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names) {
  if (validate_names) {
    check_names_valid_for(impl, names);
  }
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    impl->set_named_tensor_meta(c10::guts::make_unique<NamedTensorMeta>(names));
  } else {
    meta->set_names(names);
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
  const auto* named_tensor_meta = get_named_tensor_meta(impl);
  return named_tensor_meta != nullptr && named_tensor_meta->has_names();
}

} // namespace impl

} // namespace at
#endif
