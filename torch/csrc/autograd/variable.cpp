#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/generated/Functions.h>

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

std::shared_ptr<Node> Variable::grad_accumulator() const {
  auto autograd_meta = get_autograd_meta();
  if (!autograd_meta) {
    return nullptr;
  }
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
  // I think the choice here is conservative.  In principle, doing
  // an in-place detach should give us the ability to just clear
  // the autograd meta.  But this function ONLY resets requires_grad,
  // grad_fn and output_nr; there's other metadata like debug name
  // and hooks which aren't cleared.  Is this function supposed to
  // clear those too? I'm not too sure, so I'm leaving it be for now.
  auto autograd_meta = materialize_autograd_meta();
  autograd_meta->set_requires_grad(false, unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
}

void Variable::backward(
    const Tensor& gradient,
    bool keep_graph,
    bool create_graph) const {
  torch::autograd::backward({*this}, {gradient}, keep_graph, create_graph);
}

void Variable::set_data(const at::Tensor &new_data) const {
  // `var.set_data(new_data)` shallow-copies all non-autograd TensorImpl fields
  // from `new_data` to `var`. It requires that `new_data` and `var` have compatible
  // tensor type.
  TORCH_CHECK(
    _has_compatible_shallow_copy_type(*this, new_data),
    "Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.");

  // Resets gradient accumulator if metadata is out of date
  AutogradMeta* autograd_meta = get_autograd_meta();
  if (autograd_meta) {
    std::lock_guard<std::mutex> lock(autograd_meta->mutex_);
    auto prior_accumulator = autograd_meta->grad_accumulator_.lock();
    if (prior_accumulator) {
      const auto prior_device = prior_accumulator->input_metadata(0).device();
      const auto new_device = new_data.device();

      if (new_data.type() != type() || prior_device != new_device) {
        autograd_meta->grad_accumulator_.reset();
      }
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

DifferentiableViewMeta::DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base)
    : AutogradMeta(self_impl, false) {
  base_ = std::move(base);
  TORCH_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  self_impl->set_version_counter(base_.version_counter());
  attr_version = self_impl->version_counter().current_version();
}

DifferentiableViewMeta::~DifferentiableViewMeta() {
  base_.reset();
}

namespace {
  std::shared_ptr<Node> singleton_shared_ptr;
}

const std::shared_ptr<Node>& Variable::grad_fn() const {
  if (is_view()) {
    // NB: is_view() ==> get_autograd_meta()
    auto diff_view_meta = static_cast<DifferentiableViewMeta*>(get_autograd_meta());
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
    if (get_autograd_meta()) {
      return get_autograd_meta()->grad_fn_;
    } else {
      return singleton_shared_ptr;
    }
  }
}

void Variable::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    // NB: is_view() ==> get_autograd_meta()
    auto diff_view_meta = static_cast<DifferentiableViewMeta*>(get_autograd_meta());
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

void Variable::create_cpp_hook() {
  auto &list = materialize_autograd_meta()->cpp_hooks_list;
  list.reset(new hooks_list());
  std::unique_ptr<FunctionPreHook> hook_ptr(new CppFunctionPreHook(list, output_nr()));
  clear_hooks();
  add_hook(std::make_shared<CppFunctionPreHook>(list, 0));
  auto fn = grad_fn();
  if (fn) {
    fn->add_pre_hook(std::move(hook_ptr));
  }
}

void Variable::remove_hook(unsigned pos) {
  auto &list = materialize_autograd_meta()->cpp_hooks_list;
  TORCH_CHECK(list && pos < list->size() , "Invalid index, no hook at position ", pos);
  // Hook will be ignored
  (*list)[pos] = nullptr;
}

namespace {

at::Tensor singleton_undefined_tensor;

struct ConcreteAutogradMetaFactory : public c10::impl::AutogradMetaFactory {
  std::unique_ptr<c10::AutogradMetaInterface> make() const override {
    return c10::guts::make_unique<AutogradMeta>();
  }
  const at::Tensor& undefined_tensor() const override {
    return singleton_undefined_tensor;
  }
};

ConcreteAutogradMetaFactory meta_factory;

static c10::impl::AutogradMetaFactoryRegisterer meta_factory_registerer(&meta_factory);

}

AutogradMeta* Variable::materialize_autograd_meta() {
  auto p = unsafeGetTensorImpl();
  if (!p->autograd_meta()) {
    p->set_autograd_meta(c10::guts::make_unique<AutogradMeta>());
  }
  return get_autograd_meta();
}

}} // namespace torch::autograd
