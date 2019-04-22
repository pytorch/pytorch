#include <c10/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <c10/core/WrapDimMinimal.h>
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

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id)
    : TensorImpl(std::move(storage), type_id, storage.dtype(), storage.device()) {}

TensorImpl::TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt)
    : TensorImpl({}, type_id, data_type, std::move(device_opt)) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type,
                       c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      sizes_{0},
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt),
      type_id_(type_id) {
  AT_ASSERT(type_id == UndefinedTensorId() || data_type.id() ==  caffe2::TypeIdentifier::uninitialized() ||
            device_opt_.has_value());
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

void TensorImpl::release_resources() {
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

void TensorImpl::set_memory_format_tag(MemoryFormat memory_format) {
  AT_ASSERT(
      !is_variable()); // TODO: remove this when Variable and Tensor are merged
  memory_format_tag_ = memory_format;
}

bool TensorImpl::maybe_as_channels_last() {
  AT_ASSERT(
      !is_variable()); // TODO: remove this when Variable and Tensor are merged
  if (dim() == 4 && is_contiguous_) {
    strides_[1] = 1;
    strides_[3] = sizes_[1]; // size(1);
    strides_[2] = strides_[3] * sizes_[3];
    strides_[0] = strides_[2] * sizes_[2];
    set_memory_format_tag(at::MemoryFormat::ChannelsLast);
    is_contiguous_ = false;
    return true;
  } else if (dim() == 4) {
    auto strides_1 = 1;
    auto strides_3 = sizes_[1]; // size(1);
    auto strides_2 = strides_3 * sizes_[3];
    auto strides_0 = strides_2 * sizes_[2];
    if (strides_0 == strides_[0] && strides_1 == strides_[1] &&
        strides_2 == strides_[2] && strides_3 == strides_[3]) {
      set_memory_format_tag(at::MemoryFormat::ChannelsLast);
      return true;
    }
  }
  return false;
}

// VITALYF Move back to .h file
bool TensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
#ifdef DEBUG
  AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    return memory_format_tag_ == MemoryFormat::ChannelsLast;
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

} // namespace c10
