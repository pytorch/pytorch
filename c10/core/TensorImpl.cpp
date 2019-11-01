#include <c10/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/LocalTensorTypeSet.h>
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
  if (autograd_meta()) {
    return autograd_meta()->grad();
  } else {
    AT_ERROR("grad is not implemented for Tensor");
  }
}

const at::Tensor& TensorImpl::grad() const {
  if (autograd_meta()) {
    return autograd_meta()->grad();
  } else {
    AT_ERROR("grad is not implemented for Tensor");
  }
}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeSet type_set)
    : TensorImpl(std::move(storage), type_set, storage.dtype(), storage.device()) {}

TensorImpl::TensorImpl(TensorTypeSet type_set, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt)
    : TensorImpl({}, type_set, data_type, std::move(device_opt)) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeSet type_set, const caffe2::TypeMeta& data_type,
                       c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      sizes_{0},
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt),
      type_set_(type_set.remove(TensorTypeId::VariableTensorId)) {
  if (!type_set.empty()) {
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
    if (size(d) != 1) {
      if (stride(d) == z) {
        z *= size(d);
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

bool TensorImpl::compute_channels_last_contiguous() const {
  if (dim() == 4) {
    int64_t expected = 1;
    for (auto& d : {1, 3, 2, 0}) {
      if (size(d) != 1) {
        if (stride(d) == expected) {
          expected *= size(d);
        } else {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

bool TensorImpl::compute_strides_like_channels_last() const {
  if (dim() == 4) {
    int64_t min = 0;
    for (auto& d : {1, 3, 2, 0}) {
      if (size(d) != 1) {
        if (stride(d) > min) {
          min = stride(d);
        } else {
          return false;
        }
      }
    }
    return true;
  }
  return false;
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

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    resize_dim(0);
  }
  return this;
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

bool NonVariableTypeMode::is_enabled() {
  return !impl::tls_variable_is_enabled();
}

void NonVariableTypeMode::set_enabled(bool enabled) {
  impl::tls_variable_set_enabled(!enabled);
}

} // namespace c10
