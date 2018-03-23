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

#include <ATen/ATen.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch { namespace autograd {
Variable::Impl::Impl(at::Tensor data_, bool requires_grad_, Edge gradient_edge_)
    : TensorImpl(VariableType::getType(data_)),
      data(std::move(data_)),
      grad_fn(std::move(gradient_edge_.function)),
      requires_grad(requires_grad_),
      is_view(false),
      output_nr(gradient_edge_.input_nr),
      pyobj(nullptr) {
  TORCH_ASSERTM(
      !grad_fn || !requires_grad,
      "_requires_grad should be false if grad_fn is set");
  if (!data.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

Variable::Impl::~Impl() = default;

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
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(Variable(this, true));
  grad_accumulator = result;
  return result;
}

Variable::ViewImpl::ViewImpl(
    Variable base_,
    at::Tensor data_,
    Edge gradient_edge_)
    : Variable::Impl(std::move(data_), false, std::move(gradient_edge_)),
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
    fn->set_next_edges(collect_next_edges(base));
    fn->set_num_inputs(1);
    grad_fn = std::move(fn);
    attr_version = current_version;
  }
  return grad_fn;
}

void Variable::ViewImpl::rebase_history(Edge gradient_edge) {
  TORCH_ASSERT(gradient_edge.input_nr == 0);
  TORCH_ASSERT(gradient_edge.function);
  TORCH_ASSERTM(
      gradient_edge.function->num_inputs() == 1,
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
  auto detached = make_variable(data(), /*requires_grad=*/false);
  detached.set_version_counter(version_counter());
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

void Variable::set_tracing_state(
    jit::tracer::ValueTracingState* new_tracing_state) {
  get()->tracing_state.reset(new_tracing_state);
}

jit::tracer::ValueTracingState& Variable::tracing_state() const noexcept {
  return *get()->tracing_state;
}
}} // namespace torch::autograd
