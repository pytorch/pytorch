#include <ATen/TensorImpl.h>

#include "ATen/Context.h"
#include <ATen/Tensor.h>
#include <ATen/core/optional.h>
#include <ATen/Context.h>

#include <ATen/detail/VariableHooksInterface.h>

#include <TH/THTensor.hpp>

namespace at {

Type& TensorImpl::type() const {
  Type* base_type = &globalContext().getType(backend_, scalar_type_);
  if (is_variable_) {
    return detail::getVariableHooks().getVariableType(*base_type);
  } else {
    return *base_type;
  }
}

Tensor& TensorImpl::grad() {
  AT_ERROR("grad is not implemented for Tensor");
}

const Tensor& TensorImpl::grad() const {
  AT_ERROR("grad is not implemented for Tensor");
}

Tensor TensorImpl::detach() const {
  AT_ERROR("detach is not implemented for Tensor");
}

const char* TensorImpl::toString() const {
  // This matches behavior with VariableImpl
  return type().toString();
}

void TensorImpl::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  AT_ERROR("backward is not implemented for Tensor");
}

void TensorImpl::set_data(Tensor new_data) {
  AT_ERROR("set_type is not implemented for Tensor");
}

void Tensor::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  pImpl->backward(std::move(gradient), keep_graph, create_graph);
}

TensorImpl::TensorImpl(Backend backend, ScalarType scalar_type) {
  backend_ = backend;
  scalar_type_ = scalar_type;
  auto type = &globalContext().getType(backend, scalar_type);
  Storage* storage = type->storage(true).release();
  StorageImpl* storage_impl = storage->pImpl();
  tensor = new THTensor(storage_impl);
}

TensorImpl::~TensorImpl() {
  if (tensor) tensor->release();
}

IntList TensorImpl::sizes() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(THTensor_getSizePtr(tensor), dim());
}

IntList TensorImpl::strides() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(THTensor_getStridePtr(tensor), dim());
}

void TensorImpl::release_resources() {
  if (tensor) {
      tensor->release();
      tensor = nullptr;
  }
}

int64_t TensorImpl::dim() const {
  if(THTensor_isZeroDim(tensor)) {
    return 0;
  }
  return tensor->dim();
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  AT_CHECK(tensor, "TensorImpl without THTensor in maybe_zero_dim");
  THTensor_maybe_zero_dim(tensor, condition_when_zero_dim);
  return this;
}

void * TensorImpl::unsafeGetTH(bool retain) {
  if (retain) {
    tensor->retain();
  }
  return tensor;
}

std::unique_ptr<Storage> TensorImpl::storage() {
  StorageImpl* storage = tensor->storage_;
  storage->retain();
  return std::unique_ptr<Storage>(new Storage(storage));
}

} // namespace at
