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
Variable::Impl::Impl(at::Tensor data, bool requires_grad, Edge gradient_edge)
    : TensorImpl(VariableType::getType(data)),
      data_(std::move(data)),
      grad_fn_(std::move(gradient_edge.function)),
      requires_grad_(requires_grad),
      is_view_(false),
      output_nr_(gradient_edge.input_nr),
      pyobj_(nullptr) {
  TORCH_ASSERTM(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
  if (!data_.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

Variable::Impl::~Impl() = default;

const char* Variable::Impl::toString() const {
  // technically this will say Variable[CPUFloatType] rather than
  // Variable[CPUFloatTensor], but this is better than just Variable
  return type().toString();
}

IntList Variable::Impl::sizes() const {
  return data_.sizes();
}

IntList Variable::Impl::strides() const {
  return data_.strides();
}

int64_t Variable::Impl::dim() const {
  return data_.dim();
}

const char* Variable::Impl::typeString() {
  return "VariableType";
}

void* Variable::Impl::unsafeGetTH(bool retain) {
  return data_.unsafeGetTH(retain);
}

std::unique_ptr<at::Storage> Variable::Impl::storage() {
  return data_.storage();
}

Scalar Variable::Impl::localScalar() {
  return data_.pImpl->localScalar();
}

std::shared_ptr<Function> Variable::Impl::get_grad_accumulator() {
  if (grad_fn_) {
    throw std::logic_error(
        "get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!requires_grad_) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto result = grad_accumulator_.lock();
  if (result)
    return result;

  result = std::make_shared<AccumulateGrad>(Variable(this, true));
  grad_accumulator_ = result;
  return result;
}

Tensor Variable::Impl::detach() const {
  auto detached = make_variable(data_, /*requires_grad=*/false);
  detached.set_version_counter(version_counter_);
  return detached;
}

void Variable::Impl::detach_() {
  if (is_view_) {
    throw std::runtime_error(
        "Can't detach views in-place. Use detach() instead");
  }
  set_requires_grad(false);
  grad_fn_.reset();
  output_nr_ = 0;
}

void Variable::Impl::set_data(Tensor new_data) {
  data_ = std::move(new_data);
  if (data_.type() != *type_) {
    type_ = VariableType::getType(data_);
  }
}

Variable::ViewImpl::ViewImpl(Variable base, at::Tensor data, Edge gradient_edge)
    : Variable::Impl(std::move(data), false, std::move(gradient_edge)),
      base_(std::move(base)) {
  TORCH_ASSERTM(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  version_counter_ = base_.version_counter();
  attr_version = version_counter_.current_version();
}

std::shared_ptr<Function>& Variable::ViewImpl::get_grad_fn() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!grad_fn_ && !base_.requires_grad()) {
    return grad_fn_;
  }
  auto current_version = version_counter_.current_version();
  if (attr_version != current_version) {
    TORCH_ASSERT(output_nr_ == 0);
    auto fn = std::make_shared<generated::AsStridedBackward>();
    fn->self_geometry = at::TensorGeometry(base_);
    fn->size = sizes();
    fn->stride = strides();
    fn->storage_offset = data_.storage_offset();
    fn->set_next_edges(collect_next_edges(base_));
    fn->set_num_inputs(1);
    grad_fn_ = std::move(fn);
    attr_version = current_version;
  }
  return grad_fn_;
}

void Variable::ViewImpl::rebase_history(Edge gradient_edge) {
  TORCH_ASSERT(gradient_edge.input_nr == 0);
  TORCH_ASSERT(gradient_edge.function);
  TORCH_ASSERTM(
      gradient_edge.function->num_inputs() == 1,
      "Functions which modify views in-place must return a single Variable");
  this->output_nr_ = gradient_edge.input_nr;
  auto copy_slices = std::make_shared<CopySlices>(
      base_, at::TensorGeometry(data_), std::move(gradient_edge.function));
  base_.set_gradient_edge({std::move(copy_slices), 0});
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

void Variable::set_tracing_state(
    jit::tracer::ValueTracingState* new_tracing_state) {
  get()->tracing_state_.reset(new_tracing_state);
}

jit::tracer::ValueTracingState& Variable::tracing_state() const noexcept {
  return *get()->tracing_state_;
}
}} // namespace torch::autograd
