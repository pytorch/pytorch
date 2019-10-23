#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
// TODO: dependency shindig
#include <torch/csrc/autograd/generated/Functions.h>

#include <iostream>

namespace at {

void Tensor::enforce_invariants() {
  if (impl_.get() == nullptr) {
    throw std::runtime_error("TensorImpl with nullptr is not supported");
  }
  // Following line throws if the method is not a POD data type or is not
  // supported by ATen
  scalar_type();
  if (defined()) {
    // If it's a variable - we definitely not in C2 land
    if (!is_variable()) {
      AT_ASSERTM(
          impl_->dtype_initialized(),
          "Partially-initialized tensor not supported by Tensor");
      AT_ASSERTM(
          !impl_->is_sparse(),
          "Sparse Tensors are supported by Tensor, but invariant checking isn't implemented.  Please file a bug.");
      AT_ASSERTM(
          impl_->storage_initialized(),
          "Partially-initialized tensor not supported by Tensor");
    }
  }
}

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

std::string Tensor::toString() const {
  return type().toString();
}

// Define the methods out-of-line when they access fields on AutogradMeta
// (eventually we will make it possible to access AutogradMeta inline, but not
// yet.)

// Variable
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor Tensor::variable_data() const noexcept {
  auto self_impl_copy = impl_->shallow_copy_and_detach(
    /*version_counter=*/0,
    /*allow_tensor_metadata_change=*/false);
  self_impl_copy->set_autograd_meta(c10::guts::make_unique<torch::autograd::AutogradMeta>(self_impl_copy.get(), false));
  return Tensor(self_impl_copy);
}

// Gradient Node and Edges
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch::autograd::Node* Tensor::grad_fn_unsafe() const {
  return get_autograd_meta()->grad_fn_.get();
}

std::shared_ptr<torch::autograd::Node> Tensor::try_get_grad_accumulator() const {
  return get_autograd_meta()->grad_accumulator_.lock();
}

void Tensor::set_gradient_edge(torch::autograd::Edge edge) const noexcept {
  get_autograd_meta()->grad_fn_ = std::move(edge.function);
  get_autograd_meta()->output_nr_ = edge.input_nr;
}

// Hooks
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void Tensor::add_hook(std::shared_ptr<torch::autograd::FunctionPreHook> hook) const {
  get_autograd_meta()->hooks_.push_back(std::move(hook));
}

const std::vector<std::shared_ptr<torch::autograd::FunctionPreHook>>& Tensor::hooks()
    const noexcept {
  return get_autograd_meta()->hooks_;
}

void Tensor::clear_hooks() const {
  get_autograd_meta()->hooks_.clear();
}

// View Tensors
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool Tensor::is_view() const noexcept {
  return get_autograd_meta()->is_view_;
}

const Tensor& Tensor::base() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<torch::autograd::DifferentiableViewMeta*>(get_autograd_meta());
    return diff_view_meta->base_;
  } else {
    throw std::runtime_error("Can't get base of non-view Tensor");
  }
}

// Miscellaneous
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void Tensor::set_name(const std::string& name) const {
  get_autograd_meta()->name_ = name;
}

const std::string& Tensor::name() const noexcept {
  return get_autograd_meta()->name_;
}

torch::autograd::AutogradMeta* Tensor::get_autograd_meta() const noexcept {
  return static_cast<torch::autograd::AutogradMeta*>(impl_->autograd_meta());
}

// Traditionally out-of-line functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

std::shared_ptr<torch::autograd::Node> Tensor::grad_accumulator() const {
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

  result = std::make_shared<torch::autograd::AccumulateGrad>(*this);
  autograd_meta->grad_accumulator_ = result;
  return result;
}

const std::shared_ptr<torch::autograd::Node>& Tensor::grad_fn() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<torch::autograd::DifferentiableViewMeta*>(get_autograd_meta());
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    if (!diff_view_meta->grad_fn_ && !diff_view_meta->base_.requires_grad()) {
      return diff_view_meta->grad_fn_;
    }
    auto current_version = this->current_version();
    if (diff_view_meta->attr_version != current_version) {
      AT_ASSERT(diff_view_meta->output_nr_ == 0);
      auto fn = std::make_shared<torch::autograd::generated::AsStridedBackward>();
      fn->self_geometry = at::TensorGeometry(diff_view_meta->base_);
      fn->size = sizes().vec();
      fn->stride = strides().vec();
      fn->storage_offset = storage_offset();
      fn->set_next_edges(torch::autograd::collect_next_edges(diff_view_meta->base_));
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

void Tensor::rebase_history(torch::autograd::Edge gradient_edge) const {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    auto diff_view_meta = static_cast<torch::autograd::DifferentiableViewMeta*>(get_autograd_meta());
    AT_ASSERT(gradient_edge.input_nr == 0);
    AT_ASSERT(gradient_edge.function);
    TORCH_CHECK(
        gradient_edge.function->num_inputs() == 1,
        "Functions which modify views in-place must return a single Variable");
    diff_view_meta->output_nr_ = gradient_edge.input_nr;
    auto copy_slices = std::make_shared<torch::autograd::CopySlices>(
        diff_view_meta->base_, at::TensorGeometry(*this), std::move(gradient_edge.function));
    diff_view_meta->base_.set_gradient_edge({std::move(copy_slices), 0});
    grad_fn(); // trigger an update to the view's grad_fn
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

void Tensor::remove_hook(unsigned pos) const {
  auto &list = get_autograd_meta()->cpp_hooks_list;
  TORCH_CHECK(list && pos < list->size() , "Invalid index, no hook at position ", pos);
  // Hook will be ignored
  (*list)[pos] = nullptr;
}


} // namespace at
