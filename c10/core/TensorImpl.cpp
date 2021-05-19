#include <c10/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Optional.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(
    caffe2_keep_on_shrink,
    true,
    "If set, keeps memory when a tensor is shrinking its size.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int64(
    caffe2_max_keep_on_shrink_memory,
    LLONG_MAX,
    "The maximum memory in bytes to keep on shrink, if the difference between "
    "tensor sizes is bigger than this then tensor will be reset.");

namespace c10 {

const char* const TensorImpl::err_msg_tensor_metadata_change_not_allowed =
    "is not allowed on a Tensor created from .data or .detach().\n"
    "If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)\n"
    "without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.\n"
    "For example, change:\n"
    "    x.data.set_(y)\n"
    "to:\n"
    "    with torch.no_grad():\n"
    "        x.set_(y)";

at::Tensor& TensorImpl::mutable_grad() {
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  return autograd_meta_->mutable_grad();
}

const at::Tensor& TensorImpl::grad() const {
  // Yes, I know this looks really weird.  But I don't really have a choice as
  // long as this function returns a const reference to Tensor.  I'm not
  // really sure how I would have designed this API differently, but it
  // is not so easy to fix right now because the mutable counterpart of
  // this function must keep working so that "x.grad() = ..." keeps working
  // (part of public API).
  if (!autograd_meta_)
    return impl::GetAutogradMetaFactory()->undefined_tensor();
  return autograd_meta_->grad();
}

const at::Tensor& TensorImpl::_fw_grad(uint64_t level, const at::Tensor& self)
    const {
  // See TensorImpl::grad() above for explanation about the line below
  if (!autograd_meta_)
    return impl::GetAutogradMetaFactory()->undefined_tensor();
  return autograd_meta_->fw_grad(level, self);
}

void TensorImpl::_set_fw_grad(
    const at::Tensor& new_grad,
    const at::Tensor& self,
    uint64_t level,
    bool is_inplace_op) {
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  autograd_meta_->set_fw_grad(new_grad, self, level, is_inplace_op);
}

TensorImpl::TensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type)
    // Use std::forward to suppress static analyzer false positive.
    : TensorImpl(
          std::forward<Storage>(storage),
          key_set,
          data_type,
          storage.device()) {}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
TensorImpl::TensorImpl(
    ImplType type,
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type)
    : storage_(std::move(storage)),
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(storage_.device()),
      key_set_(key_set) {
  init_bitfields();
  // Inference tensor doesn't have version counter.
  if (!is_inference_tensor()) {
    version_counter_ = VariableVersion(/*version=*/0);
  }
}

TensorImpl::TensorImpl(
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    c10::optional<c10::Device> device_opt)
    // NOLINTNEXTLINE(performance-move-const-arg)
    : TensorImpl({}, key_set, data_type, std::move(device_opt)) {}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
TensorImpl::TensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt) {
  init_bitfields();

  if (!key_set.empty()) {
    TORCH_INTERNAL_ASSERT(
        data_type == ScalarType::Undefined || device_opt_.has_value());
    // UndefinedTensorImpl is a singleton, so we skip logging it
    C10_LOG_API_USAGE_ONCE("tensor.create");
  }

  bool inference_mode = c10::InferenceMode::is_enabled();

  // TODO: be more explicit about the full key set at call sites so we
  // don't have to keep recomputing it here
  DispatchKey k = key_set.highestPriorityBackendTypeId();

  key_set = key_set | getAutocastRelatedKeySetFromBackend(k);

  // Inference tensor doesn't have autograd related keys.
  if (inference_mode) {
    // See Note [Expected TLS state in InferenceMode] for why we exclude
    // Autograd & ADInplaceOrView keys. Normally key_set only contains backend
    // keys but we do the substraction here to make sure.
    key_set_ = key_set - c10::autograd_dispatch_keyset_with_ADInplaceOrView;
  } else {
    // TODO: Ideally we only add AutogradBackend key when the tensor requires
    // grad.
    //       See Note [Dream: skip VariableType kernel when requires_grad=false]
    key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);
  }

  // Inference tensor doesn't have version counter.
  if (!is_inference_tensor()) {
    version_counter_ = VariableVersion(/*version=*/0);
  }

  // we would also like to check that non-cpu devices have an index, but some
  // Caffe2 operators create Storages with default devices.
}

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
IntArrayRef TensorImpl::sizes() const {
  return sizes_and_strides_.sizes_arrayref();
}
#endif

IntArrayRef TensorImpl::strides() const {
  return sizes_and_strides_.strides_arrayref();
}

void TensorImpl::HandleResize() {
  // If needed, we will free the data. the next mutable_data() call
  // will create the data storage.
  bool reset_tensor = false;
  if (reserved_) {
    // If tensor is reserved then don't claim its memeory unless nbytes()
    // is smaller than new size
    reset_tensor =
        storage_.nbytes() < (storage_offset_ + numel_) * data_type_.itemsize();
  } else {
    reset_tensor = storage_.nbytes() <
            (storage_offset_ + numel_) * data_type_.itemsize() ||
        !FLAGS_caffe2_keep_on_shrink ||
        storage_.nbytes() - (storage_offset_ + numel_) * data_type_.itemsize() >
            static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
  }

  if (reset_tensor && storage_initialized()) {
    FreeMemory();
  }
}

bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = dim() - 1; d >= 0; d--) {
    const auto size_d = sizes_and_strides_.size_at_unchecked(d);
    if (size_d != 1) {
      if (sizes_and_strides_.stride_at_unchecked(d) == z) {
        z *= size_d;
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

bool TensorImpl::compute_channels_last_contiguous_2d() const {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes_and_strides_.size()) {
    case 4: {
      int64_t expected = 1;
      for (auto& d : {1, 3, 2, 0}) {
        const auto size_d = sizes_and_strides_.size_at_unchecked(d);
        if (size_d != 1) {
          if (sizes_and_strides_.stride_at_unchecked(d) != expected) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

bool TensorImpl::compute_channels_last_contiguous_3d() const {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes_and_strides_.size()) {
    case 5: {
      int64_t expected = 1;
      for (auto& d : {1, 4, 3, 2, 0}) {
        const auto size_d = sizes_and_strides_.size_at_unchecked(d);
        if (size_d != 1) {
          if (sizes_and_strides_.stride_at_unchecked(d) != expected) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

bool TensorImpl::compute_strides_like_channels_last_2d() const {
  return is_channels_last_strides_2d(
      TensorImpl::sizes(), TensorImpl::strides());
}

bool TensorImpl::compute_strides_like_channels_last_3d() const {
  return is_channels_last_strides_3d(
      TensorImpl::sizes(), TensorImpl::strides());
}

bool TensorImpl::compute_non_overlapping_and_dense() const {
  if (dim() == 1) {
    return sizes_and_strides_.size_at_unchecked(0) < 2 ||
        sizes_and_strides_.stride_at_unchecked(0) == 1;
  }
  SmallVector<int64_t, 5> perm;
  perm.resize(dim());
  for (int64_t i = 0; i < dim(); i++) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes_and_strides_.size_at_unchecked(a) < 2) {
      return false;
    } else if (sizes_and_strides_.size_at_unchecked(b) < 2) {
      return true;
    }
    return sizes_and_strides_.stride_at_unchecked(a) <
        sizes_and_strides_.stride_at_unchecked(b);
  });
  auto require_stride = 1;
  for (int64_t i = 0; i < dim(); i++) {
    const auto size_perm_i = sizes_and_strides_.size_at_unchecked(perm[i]);
    if (size_perm_i < 2) {
      return true;
    }
    if (sizes_and_strides_.stride_at_unchecked(perm[i]) != require_stride) {
      return false;
    }
    require_stride *= size_perm_i;
  }
  return true;
}

void TensorImpl::release_resources() {
  autograd_meta_.reset();
  if (storage_) {
    storage_ = {};
  }
}

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
int64_t TensorImpl::dim() const {
  return sizes_and_strides_.size();
}
#endif

int64_t TensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return sizes_and_strides_.size_at_unchecked(d);
}

int64_t TensorImpl::stride(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return sizes_and_strides_.stride_at_unchecked(d);
}

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
bool TensorImpl::has_storage() const {
  return storage_;
}
#endif

void TensorImpl::throw_storage_access_error() const {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "Cannot access storage of ", tensorimpl_type_name());
}

bool TensorImpl::is_contiguous_nondefault_policy_impl(
    at::MemoryFormat memory_format) const {
  if (has_contiguity_ ==
      static_cast<uint8_t>(HasContiguityPolicy::ContiguityNotSupported)) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "Tensors of type ",
        tensorimpl_type_name(),
        " do not have is_contiguous");
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        has_contiguity_ ==
        static_cast<uint8_t>(HasContiguityPolicy::CustomBehavior));
    return is_contiguous_custom(memory_format);
  }
}

bool TensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  TORCH_INTERNAL_ASSERT(
      false,
      "TensorImpl::is_contiguous_custom should never be called; did you "
      "set_has_contiguity_policy and forget to override is_contiguous_custom?");
}

static void deletePlacementDeleteContext(void* ptr) {
  delete static_cast<PlacementDeleteContext*>(ptr);
}

at::DataPtr PlacementDeleteContext::makeDataPtr(
    at::DataPtr&& data_ptr,
    PlacementDtor placement_dtor,
    size_t size,
    at::Device device) {
  auto* ptr = data_ptr.get();
  return {
      ptr,
      new PlacementDeleteContext(std::move(data_ptr), placement_dtor, size),
      &deletePlacementDeleteContext,
      device};
}

// NOLINTNEXTLINE(modernize-use-equals-default)
AutogradMetaInterface::~AutogradMetaInterface() {}

// Setting requires_grad to true on inference tensor outside InferenceMode
// is forbidden.  Ideally it would also be illegal inside InferenceMode.
// But there's no way that we can directly allocate a tensor to have
// requires_grad = true in C++ constructor so set_requires_grad is widely
// used in C++ frontend. Forbidding it inside InferenceMode will force users
// to delete these setter code in their code which is not ideal.
void TensorImpl::set_requires_grad(bool requires_grad) {
  TORCH_CHECK(
      !(requires_grad && is_inference_tensor() &&
        !c10::InferenceMode::is_enabled()),
      "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
  if (!requires_grad && !autograd_meta_)
    return;
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  // NB: In principle, setting requires_grad to false could result in
  // the AutogradMeta becoming equal to a default constructed state,
  // in which case we could apply the nullptr AutogradMeta optimization
  // (see autograd_meta_ docs).  But we don't do this right now.  Note
  // that it is unsound to unconditionally set AutogradMeta to false
  // when you set requires_grad to False, as there may be nontrivial
  // information content in the other fields; for example, we may
  // have set the string name for a Variable, or there may be hooks
  // registered for it.
  autograd_meta_->set_requires_grad(requires_grad, this);
}

bool TensorImpl::requires_grad() const {
  if (!autograd_meta_)
    return false;
  return autograd_meta_->requires_grad();
}

void TensorImpl::set_autograd_meta(
    std::unique_ptr<c10::AutogradMetaInterface> autograd_meta) {
  // NB: autograd_meta may be null!  That just means it's the default
  // constructor
  autograd_meta_ = std::move(autograd_meta);
}

c10::AutogradMetaInterface* TensorImpl::autograd_meta() const {
  // NB: Might return null!
  return autograd_meta_.get();
}

c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<TensorImpl>(
      // No need to populate Storage; copy_tensor_metadata will do it for us.
      key_set_,
      data_type_,
      device_opt_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<TensorImpl>(
      // No need to populate Storage; copy_tensor_metadata will do it for us.
      key_set_,
      data_type_,
      device_opt_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

void TensorImpl::copy_tensor_metadata_except_version_counter(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    bool allow_tensor_metadata_change) {
  dest_impl->storage_ = src_impl->storage_;
  dest_impl->sizes_and_strides_ = src_impl->sizes_and_strides_;
  dest_impl->storage_offset_ = src_impl->storage_offset_;
  dest_impl->data_type_ = src_impl->data_type_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  dest_impl->key_set_ = src_impl->key_set_;
  dest_impl->is_contiguous_ = src_impl->is_contiguous_;
  dest_impl->has_contiguity_ = src_impl->has_contiguity_;
  dest_impl->is_channels_last_contiguous_ =
      src_impl->is_channels_last_contiguous_;
  dest_impl->is_channels_last_3d_contiguous_ =
      src_impl->is_channels_last_3d_contiguous_;
  dest_impl->is_channels_last_ = src_impl->is_channels_last_;
  dest_impl->is_channels_last_3d_ = src_impl->is_channels_last_3d_;
  dest_impl->is_non_overlapping_and_dense_ =
      src_impl->is_non_overlapping_and_dense_;
  dest_impl->is_wrapped_number_ = src_impl->is_wrapped_number_;
  dest_impl->reserved_ = src_impl->reserved_;
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  dest_impl->storage_access_should_throw_ =
      src_impl->storage_access_should_throw_;
  if (src_impl->named_tensor_meta_ != nullptr) {
    dest_impl->named_tensor_meta_ = src_impl->named_tensor_meta_->clone();
  }
}

void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  copy_tensor_metadata_except_version_counter(
      src_impl, dest_impl, allow_tensor_metadata_change);
  // TODO: In the ideal end state, it's okay to set disabled version_counter
  // on inference tensor since it's a no-op. This requires refactor on call
  // sites.
  if (!dest_impl->is_inference_tensor()) {
    dest_impl->set_version_counter(version_counter);
  }
}

void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) {
  copy_tensor_metadata_except_version_counter(
      src_impl, dest_impl, allow_tensor_metadata_change);
  if (!dest_impl->is_inference_tensor()) {
    dest_impl->set_version_counter(std::move(version_counter));
  }
}

namespace impl {

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AutogradMetaFactory* meta_factory = nullptr;
} // namespace

void SetAutogradMetaFactory(AutogradMetaFactory* factory) {
  meta_factory = factory;
}
AutogradMetaFactory* GetAutogradMetaFactory() {
  TORCH_CHECK(
      meta_factory,
      "Support for autograd has not been loaded; have you linked against libtorch.so?")
  return meta_factory;
}

} // namespace impl

} // namespace c10
