#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/generated/VariableType.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <typeinfo>

namespace torch {
namespace autograd {
Variable::AutogradMeta::AutogradMeta(at::TensorImpl* self_impl, bool requires_grad, Edge gradient_edge) {
  grad_fn_ = std::move(gradient_edge.function);
  requires_grad_ = false;
  is_view_ = false;
  output_nr_ = gradient_edge.input_nr;

  // set_requires_grad also checks error conditions.
  set_requires_grad(requires_grad, self_impl);
  TORCH_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
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
  auto intrusive_from_this = c10::intrusive_ptr<at::TensorImpl>::reclaim(unsafeGetTensorImpl());
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
    const Tensor& gradient,
    bool keep_graph,
    bool create_graph) const {
  auto autograd_meta = get_autograd_meta();
  std::vector<Edge> edges;
  edges.emplace_back(autograd_meta->grad_fn_, autograd_meta->output_nr_);

  std::vector<Variable> inputs;
  Tensor gradient_ = gradient;
  if (!gradient.defined()) {
    gradient_ = at::ones_like(*this);
  }
  inputs.push_back(std::move(as_variable_ref(gradient_)));
  Engine::get_default_engine().execute(edges, inputs, keep_graph, create_graph);
}

void Variable::set_data(const at::Tensor &new_data) const {
  // `var.set_data(new_data)` shallow-copies all non-autograd TensorImpl fields
  // from `new_data` to `var`. It requires that `new_data` has the same derived
  // type of TensorImpl as `var`.
  TORCH_CHECK(
    _has_same_tensorimpl_type(*this, new_data),
    "Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have different types of TensorImpl.");

  // Resets gradient accumulator if metadata is out of date
  Variable::AutogradMeta* autograd_meta = get_autograd_meta();
  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);
  auto prior_accumulator = autograd_meta->grad_accumulator_.lock();
  if (prior_accumulator) {
    const auto prior_device = prior_accumulator->input_metadata(0).device();
    const auto new_device = new_data.device();

    if (new_data.type() != type() || prior_device != new_device) {
      autograd_meta->grad_accumulator_.reset();
    }
  }

  // Version counter is not shared when we replace a `Variable`'s tensor data
  // by calling `set_data(...)`. The original version of the `Variable` is always preserved.
  // See NOTE [ Version Counter Sharing ] for details.
  //
  // `var.set_data(new_data)` always ignores `var`'s `allow_tensor_metadata_change_`, because
  // users need this API as an escape hatch for changing a tensor's metadata regardless of its
  // `allow_tensor_metadata_change_` value, and the users are responsible for ensuring this is
  // the behavior they want.
  get()->shallow_copy_from(new_data.getIntrusivePtr());
}

Variable::DifferentiableViewMeta::DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base, Edge gradient_edge)
    : Variable::AutogradMeta(self_impl, false, std::move(gradient_edge)) {
  base_ = std::move(base);
  TORCH_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  self_impl->set_version_counter(base_.version_counter());
  attr_version = self_impl->version_counter().current_version();
}

Variable::DifferentiableViewMeta::~DifferentiableViewMeta() {
  base_.reset();
}

const std::shared_ptr<Function>& Variable::grad_fn() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    if (!diff_view_meta->grad_fn_ && !diff_view_meta->base_.requires_grad()) {
      return diff_view_meta->grad_fn_;
    }
    auto current_version = this->current_version();
    if (diff_view_meta->attr_version != current_version) {
      AT_ASSERT(diff_view_meta->output_nr_ == 0);
      auto fn = std::make_shared<generated::AsStridedBackward>();
      fn->self_geometry = at::TensorGeometry(diff_view_meta->base_);
      fn->size = sizes().vec();
      fn->stride = strides().vec();
      fn->storage_offset = storage_offset();
      fn->set_next_edges(collect_next_edges(diff_view_meta->base_));
      fn->add_input_metadata(
        diff_view_meta->base_.type()
      , sizes() // Note: sizes(), not base_.sizes(), is intentional
      , diff_view_meta->base_.device());
      diff_view_meta->grad_fn_ = std::move(fn);
      diff_view_meta->attr_version = current_version;
    }
    return diff_view_meta->grad_fn_;
  } else {
    return get_autograd_meta()->grad_fn_;
  }
}

void Variable::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    AT_ASSERT(gradient_edge.input_nr == 0);
    AT_ASSERT(gradient_edge.function);
    TORCH_CHECK(
        gradient_edge.function->num_inputs() == 1,
        "Functions which modify views in-place must return a single Variable");
    diff_view_meta->output_nr_ = gradient_edge.input_nr;
    auto copy_slices = std::make_shared<CopySlices>(
        diff_view_meta->base_, at::TensorGeometry(*this), std::move(gradient_edge.function));
    diff_view_meta->base_.set_gradient_edge({std::move(copy_slices), 0});
    grad_fn(); // trigger an update to the view's grad_fn
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

}} // namespace torch::autograd
