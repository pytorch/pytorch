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

TensorImpl::TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator *allocator, bool is_variable)
    : TensorImpl({}, type_id, data_type, is_variable) {
  // Variables, UndefinedTensors and SparseTensors don't have storages.
  if (!is_variable && type_id != UndefinedTensorId() && data_type.id() != caffe2::TypeIdentifier::uninitialized()
      && type_id != SparseCPUTensorId() && type_id != SparseCUDATensorId()) {
    storage_ = Storage(data_type, 0, allocator, true);
  }
}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable)
    : TensorImpl(std::move(storage), type_id, storage.dtype(), is_variable) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type, bool is_variable)
    : storage_(std::move(storage)),
      sizes_{0},
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      type_id_(type_id),
      is_variable_(is_variable) {
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
