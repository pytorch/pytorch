#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/variable_version.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace autograd {
Variable::Impl::Impl(at::Tensor data, bool requires_grad, Edge gradient_edge)
    : TensorImpl(data.type_id(), data.dtype(), /*allocator=*/nullptr, /* is variable */ true),
      data_(std::move(data)),
      grad_fn_(std::move(gradient_edge.function)),
      requires_grad_(false),
      is_view_(false),
      output_nr_(gradient_edge.input_nr),
      pyobj_(nullptr) {
  // set_requires_grad also checks error conditions.
  set_requires_grad(requires_grad);
  AT_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
  if (!data_.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

Variable::Impl::~Impl() = default;

int64_t Variable::Impl::numel() const {
  return data_.numel();
}

IntList Variable::Impl::sizes() const {
  return data_.sizes();
}

IntList Variable::Impl::strides() const {
  return data_.strides();
}

bool Variable::Impl::is_contiguous() const {
  return data_.is_contiguous();
}

int64_t Variable::Impl::dim() const {
  return data_.dim();
}

int64_t Variable::Impl::size(int64_t d) const {
  return data_.size(d);
}

int64_t Variable::Impl::stride(int64_t d) const {
  return data_.stride(d);
}

void Variable::Impl::resize_dim(int64_t ndim) {
  AT_ERROR("variable impl does not have resize_dim");
}

void Variable::Impl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("variable impl does not have set_size");
}

void Variable::Impl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("variable impl does not have set_stride");
}

void Variable::Impl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("variable impl does not have set_storage_offset");
}

void* Variable::Impl::slow_data() const {
  return data_.unsafeGetTensorImpl()->slow_data();
}

const at::Storage& Variable::Impl::storage() const {
  return data_.storage();
}

int64_t Variable::Impl::storage_offset() const {
  return data_.storage_offset();
}

int64_t Variable::Impl::get_device_slow() const {
  return data_.get_device();
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

  c10::raw::intrusive_ptr::incref(this);
  auto intrusive_from_this = c10::intrusive_ptr<Variable::Impl>::reclaim(this);
  result = std::make_shared<AccumulateGrad>(Variable(std::move(intrusive_from_this)));
  grad_accumulator_ = result;
  return result;
}

void Variable::Impl::detach_() {
  if (is_view_) {
    AT_ERROR("Can't detach views in-place. Use detach() instead");
  }
  set_requires_grad(false);
  grad_fn_.reset();
  output_nr_ = 0;
}

void Variable::Impl::backward(
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  std::vector<Edge> edges;
  edges.emplace_back(grad_fn_, output_nr_);

  std::vector<Variable> inputs;
  if (!gradient.has_value()) {
    gradient = make_variable(at::ones_like(data_), /*requires_grad=*/false);
  }
  inputs.push_back(std::move(as_variable_ref(*gradient)));
  Engine::get_default_engine().execute(edges, inputs, keep_graph, create_graph);
}

void Variable::Impl::set_data(Tensor new_data) {
  // Resets gradient accumulator if metadata is out of date
  std::lock_guard<std::mutex> lock(mutex_);
  auto prior_accumulator = grad_accumulator_.lock();
  if (prior_accumulator) {
    const auto prior_device = prior_accumulator->input_metadata(0).device();
    const auto new_device = new_data.is_cuda() ? new_data.get_device() : -1;

    if (new_data.type() != data_.type() || prior_device != new_device) {
      grad_accumulator_.reset();
    }
  }

  // Updates metadata
  data_type_ = new_data.type().typeMeta();
  type_id_ = new_data.type().type_id();
  is_variable_ = true;
  data_ = std::move(new_data);
}

void Variable::Impl::release_resources() {
  data_.reset();
  grad_.reset();
  grad_fn_.reset();
  hooks_.clear();
}

Variable::DifferentiableViewImpl::DifferentiableViewImpl(Variable base, at::Tensor data, Edge gradient_edge)
    : Variable::Impl(std::move(data), false, std::move(gradient_edge)),
      base_(std::move(base)) {
  AT_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  version_counter_ = base_.version_counter();
  attr_version = version_counter_.current_version();
}

std::shared_ptr<Function>& Variable::DifferentiableViewImpl::get_grad_fn() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!grad_fn_ && !base_.requires_grad()) {
    return grad_fn_;
  }
  auto current_version = version_counter_.current_version();
  if (attr_version != current_version) {
    AT_ASSERT(output_nr_ == 0);
    auto fn = std::make_shared<generated::AsStridedBackward>();
    fn->self_geometry = at::TensorGeometry(base_);
    fn->size = sizes().vec();
    fn->stride = strides().vec();
    fn->storage_offset = data_.storage_offset();
    fn->set_next_edges(collect_next_edges(base_));
    fn->add_input_metadata(
      base_.type()
    , sizes() // Note: sizes(), not base_.sizes(), is intentional
    , base_.is_cuda() ? base_.get_device() : -1);
    grad_fn_ = std::move(fn);
    attr_version = current_version;
  }
  return grad_fn_;
}

void Variable::DifferentiableViewImpl::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.input_nr == 0);
  AT_ASSERT(gradient_edge.function);
  AT_CHECK(
      gradient_edge.function->num_inputs() == 1,
      "Functions which modify views in-place must return a single Variable");
  this->output_nr_ = gradient_edge.input_nr;
  auto copy_slices = std::make_shared<CopySlices>(
      base_, at::TensorGeometry(data_), std::move(gradient_edge.function));
  base_.set_gradient_edge({std::move(copy_slices), 0});
  get_grad_fn(); // trigger an update to the view's grad_fn
}

void Variable::DifferentiableViewImpl::release_resources() {
  Variable::Impl::release_resources();
  base_.reset();
}

void Variable::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    auto& impl = static_cast<Variable::DifferentiableViewImpl&>(*get());
    impl.rebase_history(std::move(gradient_edge));
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

}} // namespace torch::autograd
