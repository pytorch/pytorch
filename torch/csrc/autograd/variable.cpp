#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/assertions.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/utils/auto_unique_ptr.h"

#include <ATen/Scalar.h>
#include <ATen/ScalarType.h>
#include <ATen/Storage.h>
#include <ATen/Tensor.h>
#include <ATen/TensorImpl.h>
#include <ATen/Type.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch { namespace autograd {
namespace {
at::Tensor handle_scalars(at::Tensor& data) {
#ifndef WITH_SCALARS
  if (data.dim() == 0) {
    // Don't expose 0-dim tensors to Variable API.
    return data.as_strided_({1}, {1});
  }
#endif
  return data;
}
} // namespace

Variable::Impl::Impl(at::Tensor data_, bool requires_grad_, Edge gradient_edge)
    : TensorImpl(VariableType::getType(data_)),
      data(std::move(data_)),
      grad_fn(std::move(gradient_edge.function)),
      requires_grad(requires_grad_),
      is_view(false),
      output_nr(gradient_edge.input_nr),
      pyobj(nullptr) {
  TORCH_ASSERTM(
      !grad_fn || !requires_grad,
      "_requires_grad should be false if grad_fn is set");
  if (!data.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

Variable::Impl::~Impl() = default;

Variable Variable::as_view(Variable base, at::Tensor data, Edge gradient_edge) {
  if (data.defined()) {
    data = handle_scalars(data);
    auto impl = new Variable::ViewImpl(
        std::move(base), std::move(data), std::move(gradient_edge));
    return Variable(std::move(impl), false);
  }
  return Variable();
}

Variable::Variable(at::Tensor data, bool requires_grad) {
  if (data.defined()) {
    pImpl = new Variable::Impl(handle_scalars(data), requires_grad);
  }
}

Variable::Variable(at::Tensor data, Edge gradient_edge) {
  if (data.defined()) {
    pImpl = new Variable::Impl(
        handle_scalars(data), false, std::move(gradient_edge));
  }
}

const char* Variable::Impl::toString() const {
  return "Variable";
}

IntList Variable::Impl::sizes() const {
  return data.sizes();
}

IntList Variable::Impl::strides() const {
  return data.strides();
}

int64_t Variable::Impl::dim() const {
  return data.dim();
}

const char* Variable::Impl::typeString() {
  return "VariableType";
}

void* Variable::Impl::unsafeGetTH(bool retain) {
  return data.unsafeGetTH(retain);
}

std::unique_ptr<at::Storage> Variable::Impl::storage() {
  return data.storage();
}

Scalar Variable::Impl::localScalar() {
  return data.pImpl->localScalar();
}

std::shared_ptr<Function> Variable::Impl::get_grad_accumulator() {
  if (grad_fn) {
    throw std::logic_error(
        "get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!requires_grad) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex);

  auto result = grad_accumulator.lock();
  if (result)
    return result;

  result = std::make_shared<AccumulateGrad>(Variable(this, true));
  grad_accumulator = result;
  return result;
}

Variable::ViewImpl::ViewImpl(
    Variable base_,
    at::Tensor data_,
    Edge gradient_edge)
    : Variable::Impl(std::move(data_), false, std::move(gradient_edge)),
      base(std::move(base_)) {
  TORCH_ASSERTM(base.defined(), "base is undefined");
  if (base.is_view()) {
    base = base.base();
  }
  is_view = true;
  version_counter = base.version_counter();
  attr_version = version_counter.current_version();
}

std::shared_ptr<Function>& Variable::ViewImpl::get_grad_fn() {
  std::lock_guard<std::mutex> lock(mutex);
  if (!grad_fn && !base.requires_grad()) {
    return grad_fn;
  }
  auto current_version = version_counter.current_version();
  if (attr_version != current_version) {
    TORCH_ASSERT(output_nr == 0);
    auto fn = std::make_shared<generated::AsStridedBackward>();
    fn->self_geometry = at::TensorGeometry(base);
    fn->size = sizes();
    fn->stride = strides();
    fn->storage_offset = data.storage_offset();
    fn->set_next_functions(get_next_functions(base));
    fn->num_inputs = 1;
    grad_fn = std::move(fn);
    attr_version = current_version;
  }
  return grad_fn;
}

void Variable::ViewImpl::rebase_history(Edge gradient_edge) {
  TORCH_ASSERT(gradient_edge.input_nr == 0);
  TORCH_ASSERT(gradient_edge.function);
  TORCH_ASSERTM(
      gradient_edge.function->num_inputs == 1,
      "Functions which modify views in-place must return a single Variable");
  this->output_nr = gradient_edge.input_nr;
  auto copy_slices = std::make_shared<CopySlices>(
      base, at::TensorGeometry(data), std::move(gradient_edge.function));
  base.set_gradient_edge({std::move(copy_slices), 0});
  get_grad_fn(); // trigger an update to the view's grad_fn
}

void Variable::rebase_history(Edge gradient_edge) {
  TORCH_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    auto& impl = static_cast<Variable::ViewImpl&>(*get());
    impl.rebase_history(std::move(gradient_edge));
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

Variable Variable::detach() const {
  Variable detached(data(), /*requires_grad=*/false);
  detached.set_version(version_counter());
  return detached;
}

void Variable::detach_() {
  if (is_view()) {
    throw std::runtime_error(
        "Can't detach views in-place. Use detach() instead");
  }
  set_requires_grad(false);
  set_gradient_edge(Edge());
}

}} // namespace torch::autograd
