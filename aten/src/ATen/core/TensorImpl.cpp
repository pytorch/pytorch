#include <ATen/core/TensorImpl.h>

#include <ATen/core/optional.h>
#include <ATen/core/Backend.h>
#include <ATen/core/WrapDimMinimal.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <ATen/core/VariableHooksInterface.h>

namespace at {

Tensor& TensorImpl::grad() {
  AT_ERROR("grad is not implemented for Tensor");
}

const Tensor& TensorImpl::grad() const {
  AT_ERROR("grad is not implemented for Tensor");
}

TensorImpl::TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator *allocator, TensorImplOptions options)
    : TensorImpl({}, type_id, data_type, options) {
  // UndefinedTensors and SparseTensors don't have storages.
  if (type_id != UndefinedTensorId() && data_type.id() != caffe2::TypeIdentifier::uninitialized()
      && type_id != SparseCPUTensorId() && type_id != SparseCUDATensorId()) {
    storage_ = Storage(data_type, 0, allocator, true);
  }
}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, TensorImplOptions options)
    : TensorImpl(std::move(storage), type_id, storage.dtype(), options) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type, TensorImplOptions options)
    : storage_(std::move(storage)),
      sizes_{0},
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      type_id_(type_id),
      options_(options) {
  strides_.reset(new int64_t[1]);
  strides_[0] = 1;
}

IntList TensorImpl::sizes() const {
  return sizes_;
}

IntList TensorImpl::strides() const {
  if (!options_.has_strides_) {
    AT_ERROR(type_id_, " does not have strides");
  }
  AT_ASSERTM(strides_,
             "Caffe2 tensors don't (yet) have meaningful strides and cannot "
             "be used in PyTorch.");
  return IntList{strides_.get(), sizes_.size()};
}

bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  if (!strides_) {
    // Special case for Caffe2 tensors which don't have strides set.
    return true;
  }
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
  if (!options_.has_strides_) {
    AT_ERROR(type_id_, " does not have strides");
  }
  AT_ASSERTM(strides_,
             "Caffe2 tensors don't (yet) have meaningful strides and cannot "
             "be used in PyTorch.");
  d = at::maybe_wrap_dim(d, dim(), false);
  return strides_[d];
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  // We only allow this operation on dense tensors
  if (!(options_.has_storage_ && options_.has_strides_)) {
    AT_CHECK(condition_when_zero_dim == (dim() == 0),
           type_id_, " does not support changing dimensionality via maybe_zero_dim");
  }
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    resize_dim(0);
  }
  return this;
}

const Storage& TensorImpl::storage() const {
  if (!options_.has_storage_) {
    AT_ERROR(type_id_, " does not have storage");
  }
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

} // namespace at
