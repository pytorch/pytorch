#include <ATen/TensorImpl.h>

#include <ATen/Tensor.h>
#include <ATen/optional.h>
#include <ATen/Context.h>

#include <ATen/detail/VariableHooksInterface.h>

#include <TH/THTensor.hpp>

namespace at {

Type& TensorImpl::type() const {
  Type* base_type = &globalContext().getType(backend_, scalar_type_);
  Type* r = nullptr;
  if (is_variable_) {
    r = &detail::getVariableHooks().getVariableType(*base_type);
  } else {
    r = base_type;
  }
  AT_ASSERT(type_ == r);
  return *r;
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
  bool is_zero_dim = condition_when_zero_dim && tensor->sizes().size() == 1 && tensor->size(0) == 1;
  THTensor_setIsZeroDim(tensor, is_zero_dim);
  return this;
}

void * TensorImpl::unsafeGetTH(bool retain) {
  if (retain) {
    tensor->retain();
  }
  return tensor;
}

} // namespace at
