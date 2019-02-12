#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/variable_version.h>

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
Variable::Impl::Impl(at::Tensor data, std::unique_ptr<Variable::AutogradMeta> autograd_meta, bool requires_grad, Edge gradient_edge)
    : TensorImpl(data.type_id(), data.dtype(), /*allocator=*/nullptr, /* is variable */ true),
      data_(std::move(data)) {
  autograd_meta->grad_fn_ = std::move(gradient_edge.function);
  autograd_meta->requires_grad_ = false;
  autograd_meta->is_view_ = false;
  autograd_meta->output_nr_ = gradient_edge.input_nr;
  autograd_meta->pyobj_ = nullptr;

  // set_requires_grad also checks error conditions.
  autograd_meta->set_requires_grad(requires_grad, this);
  AT_CHECK(
      !autograd_meta->grad_fn_ || !autograd_meta->requires_grad_,
      "requires_grad should be false if grad_fn is set");
  if (!data_.defined()) {
    throw std::runtime_error("data is undefined");
  }

  set_autograd_meta(std::move(autograd_meta));
}

Variable::Impl::~Impl() = default;

int64_t Variable::Impl::numel() const {
  return data_.numel();
}

IntArrayRef Variable::Impl::sizes() const {
  return data_.sizes();
}

IntArrayRef Variable::Impl::strides() const {
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

bool Variable::Impl::has_storage() const {
  return data_.has_storage();
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

std::shared_ptr<Function> Variable::grad_accumulator() const {
  auto autograd_meta = get_autograd_meta();
  if (autograd_meta->grad_fn_) {
    throw std::logic_error(
        "grad_accumulator() should be only called on leaf Variables");
  }
  if (!autograd_meta->requires_grad_) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);

  auto result = autograd_meta->grad_accumulator_.lock();
  if (result)
    return result;

  c10::raw::intrusive_ptr::incref(unsafeGetTensorImpl());
  auto intrusive_from_this = c10::intrusive_ptr<Variable::Impl>::reclaim(static_cast<Variable::Impl*>(unsafeGetTensorImpl()));
  result = std::make_shared<AccumulateGrad>(Variable(std::move(intrusive_from_this)));
  autograd_meta->grad_accumulator_ = result;
  return result;
}

void Variable::detach_() {
  if (is_view()) {
    AT_ERROR("Can't detach views in-place. Use detach() instead");
  }
  auto autograd_meta = get_autograd_meta();
  autograd_meta->set_requires_grad(false, unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
}

void Variable::backward(
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) const {
  auto autograd_meta = get_autograd_meta();
  std::vector<Edge> edges;
  edges.emplace_back(autograd_meta->grad_fn_, autograd_meta->output_nr_);

  std::vector<Variable> inputs;
  if (!gradient.has_value()) {
    gradient = make_variable(at::ones_like(data()), /*requires_grad=*/false);
  }
  inputs.push_back(std::move(as_variable_ref(*gradient)));
  Engine::get_default_engine().execute(edges, inputs, keep_graph, create_graph);
}

void Variable::Impl::set_data(Tensor new_data) {
  // Resets gradient accumulator if metadata is out of date
  auto autograd_meta = get_autograd_meta();
  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);
  auto prior_accumulator = autograd_meta->grad_accumulator_.lock();
  if (prior_accumulator) {
    const auto prior_device = prior_accumulator->input_metadata(0).device();
    const auto new_device = new_data.is_cuda() ? new_data.get_device() : -1;

    if (new_data.type() != data_.type() || prior_device != new_device) {
      autograd_meta->grad_accumulator_.reset();
    }
  }

  // Updates metadata
  data_type_ = new_data.type().typeMeta();
  type_id_ = new_data.type().type_id();
  is_variable_ = true;

  auto new_data_copy = at::Tensor(new_data.getIntrusivePtr()->shallow_copy_and_detach());
  data_ = std::move(new_data_copy);
}

void Variable::Impl::release_resources() {
  autograd_meta_.reset();
  data_.reset();
}

Variable::DifferentiableViewImpl::DifferentiableViewImpl(Variable base, at::Tensor data, Edge gradient_edge, std::unique_ptr<Variable::DifferentiableViewMeta> autograd_meta)
    : Variable::Impl(std::move(data), std::move(autograd_meta), false, std::move(gradient_edge)) {
  auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
  diff_view_meta->base_ = std::move(base);
  AT_CHECK(diff_view_meta->base_.defined(), "base is undefined");
  if (diff_view_meta->base_.is_view()) {
    diff_view_meta->base_ = diff_view_meta->base_.base();
  }
  diff_view_meta->is_view_ = true;
  diff_view_meta->version_counter_ = diff_view_meta->base_.version_counter();
  diff_view_meta->attr_version = diff_view_meta->version_counter_.current_version();
}

const std::shared_ptr<Function>& Variable::grad_fn() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    if (!diff_view_meta->grad_fn_ && !diff_view_meta->base_.requires_grad()) {
      return diff_view_meta->grad_fn_;
    }
    auto current_version = diff_view_meta->version_counter_.current_version();
    if (diff_view_meta->attr_version != current_version) {
      AT_ASSERT(diff_view_meta->output_nr_ == 0);
      auto fn = std::make_shared<generated::AsStridedBackward>();
      fn->self_geometry = at::TensorGeometry(diff_view_meta->base_);
      fn->size = sizes().vec();
      fn->stride = strides().vec();
      fn->storage_offset = data().storage_offset();
      fn->set_next_edges(collect_next_edges(diff_view_meta->base_));
      fn->add_input_metadata(
        diff_view_meta->base_.type()
      , sizes() // Note: sizes(), not base_.sizes(), is intentional
      , diff_view_meta->base_.is_cuda() ? diff_view_meta->base_.get_device() : -1);
      diff_view_meta->grad_fn_ = std::move(fn);
      diff_view_meta->attr_version = current_version;
    }
    return diff_view_meta->grad_fn_;
  } else {
    return get_autograd_meta()->grad_fn_;
  }
}

void Variable::DifferentiableViewImpl::release_resources() {
  auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
  diff_view_meta->base_.reset();
  Variable::Impl::release_resources();
}

void Variable::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    AT_ASSERT(gradient_edge.input_nr == 0);
    AT_ASSERT(gradient_edge.function);
    AT_CHECK(
        gradient_edge.function->num_inputs() == 1,
        "Functions which modify views in-place must return a single Variable");
    diff_view_meta->output_nr_ = gradient_edge.input_nr;
    auto copy_slices = std::make_shared<CopySlices>(
        diff_view_meta->base_, at::TensorGeometry(data()), std::move(gradient_edge.function));
    diff_view_meta->base_.set_gradient_edge({std::move(copy_slices), 0});
    grad_fn(); // trigger an update to the view's grad_fn
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

}} // namespace torch::autograd
