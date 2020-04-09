#include <c10/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Optional.h>

C10_DEFINE_bool(
    caffe2_keep_on_shrink,
    true,
    "If set, keeps memory when a tensor is shrinking its size.");

C10_DEFINE_int64(
    caffe2_max_keep_on_shrink_memory,
    LLONG_MAX,
    "The maximum memory in bytes to keep on shrink, if the difference between "
    "tensor sizes is bigger than this then tensor will be reset.");

namespace c10 {

const char * const TensorImpl::err_msg_tensor_metadata_change_not_allowed =
    "is not allowed on a Tensor created from .data or .detach().\n"
    "If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)\n"
    "without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.\n"
    "For example, change:\n"
    "    x.data.set_(y)\n"
    "to:\n"
    "    with torch.no_grad():\n"
    "        x.set_(y)";

at::Tensor& TensorImpl::grad() {
  if (!autograd_meta_) autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  return autograd_meta_->grad();
}

const at::Tensor& TensorImpl::grad() const {
  // Yes, I know this looks really weird.  But I don't really have a choice as
  // long as this function returns a const reference to Tensor.  I'm not
  // really sure how I would have designed this API differently, but it
  // is not so easy to fix right now because the mutable counterpart of
  // this function must keep working so that "x.grad() = ..." keeps working
  // (part of public API).
  if (!autograd_meta_) return impl::GetAutogradMetaFactory()->undefined_tensor();
  return autograd_meta_->grad();
}

TensorImpl::TensorImpl(Storage&& storage, DispatchKeySet key_set)
    : TensorImpl(std::move(storage), key_set, storage.dtype(), storage.device()) {}

TensorImpl::TensorImpl(DispatchKeySet key_set, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt)
    : TensorImpl({}, key_set, data_type, std::move(device_opt)) {}

TensorImpl::TensorImpl(Storage&& storage, DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                       c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      sizes_{0},
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt),
      key_set_(key_set) {
  if (!key_set.empty()) {
    AT_ASSERT(data_type.id() ==  caffe2::TypeIdentifier::uninitialized() ||
              device_opt_.has_value());
    // UndefinedTensorImpl is a singleton, so we skip logging it
    C10_LOG_API_USAGE_ONCE("tensor.create");
  }
  // we would also like to check that non-cpu devices have an index, but some Caffe2 operators create
  // Storages with default devices.
  strides_.push_back(1);
}

IntArrayRef TensorImpl::sizes() const {
  return sizes_;
}

IntArrayRef TensorImpl::strides() const {
  return strides_;
}

bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = dim() - 1; d >= 0; d--) {
    if (sizes_[d] != 1) {
      if (strides_[d] == z) {
        z *= sizes_[d];
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
  switch (sizes_.size()) {
    case 4:
      {
        int64_t expected = 1;
        for (auto& d : {1, 3, 2, 0}) {
          if (sizes_[d] != 1) {
            if (strides_[d] != expected) {
              return false;
            }
            expected *= sizes_[d];
          }
        }
        return true;
      }
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
  switch (sizes_.size()) {
    case 5:
      {
        int64_t expected = 1;
        for (auto& d : {1, 4, 3, 2, 0}) {
          if (sizes_[d] != 1) {
            if (strides_[d] != expected) {
              return false;
            }
            expected *= sizes_[d];
          }
        }
        return true;
      }
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

bool TensorImpl::compute_strides_like_channels_last_2d() const {
  return is_channels_last_strides_2d(sizes_, strides_);
}

bool TensorImpl::compute_strides_like_channels_last_3d() const {
  return is_channels_last_strides_3d(sizes_, strides_);
}

bool TensorImpl::compute_non_overlapping_and_dense() const {
  if (dim() == 1) {
    return sizes_[0] < 2 || strides_[0] == 1;
  }
  SmallVector<int64_t,5> perm;
  perm.resize(dim());
  for (int64_t i = 0; i < dim(); i ++) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
      if (sizes_[a] < 2) {
        return false;
      } else if (sizes_[b] < 2) {
        return true;
      }
      return strides_[a] < strides_[b];
  });
  auto require_stride = 1;
  for (int64_t i = 0; i < dim(); i ++) {
    if (sizes_[perm[i]] < 2) {
      return true;
    }
    if (strides_[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= sizes_[perm[i]];
  }
  return true;
}

void TensorImpl::release_resources() {
  autograd_meta_.reset();
  if (storage_) {
    storage_ = {};
  }
}

int64_t TensorImpl::dim() const {
  return sizes_.size();
}

int64_t TensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return sizes_[d];
}

int64_t TensorImpl::stride(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return strides_[d];
}

bool TensorImpl::has_storage() const {
  return storage_;
}

bool TensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
#ifdef DEBUG
  AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
  if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_contiguous_;
  }
  else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return is_channels_last_3d_contiguous_;
  }
  return is_contiguous_;
}

const Storage& TensorImpl::storage() const {
  return storage_;
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
  return {ptr,
          new PlacementDeleteContext(std::move(data_ptr), placement_dtor, size),
          &deletePlacementDeleteContext,
          device};
}

AutogradMetaInterface::~AutogradMetaInterface() {}

void TensorImpl::set_requires_grad(bool requires_grad) {
  if (!requires_grad && !autograd_meta_) return;
  if (!autograd_meta_) autograd_meta_ = impl::GetAutogradMetaFactory()->make();
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
  if (!autograd_meta_) return false;
  return autograd_meta_->requires_grad();
}

void TensorImpl::set_autograd_meta(std::unique_ptr<c10::AutogradMetaInterface> autograd_meta) {
  // NB: autograd_meta may be null!  That just means it's the default
  // constructor
  autograd_meta_ = std::move(autograd_meta);
}

c10::AutogradMetaInterface* TensorImpl::autograd_meta() const {
  // NB: Might return null!
  return autograd_meta_.get();
}

void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  dest_impl->storage_ = src_impl->storage_;
  dest_impl->sizes_ = src_impl->sizes_;
  dest_impl->strides_ = src_impl->strides_;
  dest_impl->storage_offset_ = src_impl->storage_offset_;
  dest_impl->data_type_ = src_impl->data_type_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  dest_impl->key_set_ = src_impl->key_set_;
  dest_impl->is_contiguous_ = src_impl->is_contiguous_;
  dest_impl->is_channels_last_contiguous_ = src_impl->is_channels_last_contiguous_;
  dest_impl->is_channels_last_3d_contiguous_ = src_impl->is_channels_last_3d_contiguous_;
  dest_impl->is_channels_last_ = src_impl->is_channels_last_;
  dest_impl->is_channels_last_3d_ = src_impl->is_channels_last_3d_;
  dest_impl->is_non_overlapping_and_dense_ = src_impl->is_non_overlapping_and_dense_;
  dest_impl->is_wrapped_number_ = src_impl->is_wrapped_number_;
  dest_impl->reserved_ = src_impl->reserved_;
  dest_impl->set_version_counter(version_counter);
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  if (src_impl->named_tensor_meta_ != nullptr) {
    dest_impl->named_tensor_meta_ = src_impl->named_tensor_meta_->clone();
  }
}

namespace impl {

namespace {
AutogradMetaFactory* meta_factory = nullptr;
}

void SetAutogradMetaFactory(AutogradMetaFactory* factory) {
  meta_factory = factory;
}
AutogradMetaFactory* GetAutogradMetaFactory() {
  TORCH_CHECK(meta_factory, "Support for autograd has not been loaded; have you linked against libtorch.so?")
  return meta_factory;
}

} // namespace impl

} // namespace c10
