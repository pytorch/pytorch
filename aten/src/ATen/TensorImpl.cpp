#include <ATen/TensorImpl.h>

#include <ATen/Tensor.h>
#include <ATen/optional.h>

#include <TH/THTensor.hpp>

namespace at {
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
  switch (type().backend()) {
    case Backend::CPU:
      switch (type().scalarType()) {
#define DEFINE_CPU_STRING(_,name,_2) \
        case ScalarType::name: return "CPU" #name "Tensor";
        AT_FORALL_SCALAR_TYPES(DEFINE_CPU_STRING)
#undef DEFINE_CPU_STRING
        case ScalarType::Undefined: AT_ASSERT(false);
        case ScalarType::NumOptions: AT_ASSERT(false);
      }
    case Backend::CUDA:
      switch (type().scalarType()) {
#define DEFINE_CUDA_STRING(_,name,_2) \
        case ScalarType::name: return "CUDA" #name "Tensor";
        AT_FORALL_SCALAR_TYPES(DEFINE_CUDA_STRING)
#undef DEFINE_CUDA_STRING
        case ScalarType::Undefined: AT_ASSERT(false);
        case ScalarType::NumOptions: AT_ASSERT(false);
      }
    case Backend::SparseCPU:
      switch (type().scalarType()) {
#define DEFINE_CPU_STRING(_,name,_2) \
        case ScalarType::name: return "SparseCPU" #name "Tensor";
        AT_FORALL_SCALAR_TYPES(DEFINE_CPU_STRING)
#undef DEFINE_CPU_STRING
        case ScalarType::Undefined: AT_ASSERT(false);
        case ScalarType::NumOptions: AT_ASSERT(false);
      }
    case Backend::SparseCUDA:
      switch (type().scalarType()) {
#define DEFINE_CUDA_STRING(_,name,_2) \
        case ScalarType::name: return "SparseCUDA" #name "Tensor";
        AT_FORALL_SCALAR_TYPES(DEFINE_CUDA_STRING)
#undef DEFINE_CUDA_STRING
        case ScalarType::Undefined: AT_ASSERT(false);
        case ScalarType::NumOptions: AT_ASSERT(false);
      }
    case Backend::Undefined: return "UndefinedTensor";
    case Backend::NumOptions: AT_ASSERT(false);
  }
  AT_ASSERT(false);
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
  if (isScalar()) {
    return 0;
  }
  return tensor->dim();
}

void * TensorImpl::unsafeGetTH(bool retain) {
  if (retain) {
    tensor->retain();
  }
  return tensor;
}

} // namespace at
