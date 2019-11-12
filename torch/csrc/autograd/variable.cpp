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
AutogradMeta::AutogradMeta(at::TensorImpl* self_impl, bool requires_grad, Edge gradient_edge) {
  grad_fn_ = std::move(gradient_edge.function);
  requires_grad_ = false;
  is_view_ = false;
  output_nr_ = gradient_edge.input_nr;

  // set_requires_grad also checks error conditions.
  if (requires_grad) {
    TORCH_INTERNAL_ASSERT(self_impl);
    set_requires_grad(requires_grad, self_impl);
  }
  TORCH_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
}


DifferentiableViewMeta::DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base)
    : AutogradMeta(self_impl, false) {
  base_ = std::move(base);
  TORCH_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  self_impl->set_version_counter(impl::version_counter(base_));
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
    auto diff_view_meta = static_cast<DifferentiableViewMeta*>(impl::get_autograd_meta(*this));
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    if (!diff_view_meta->grad_fn_ && !diff_view_meta->base_.requires_grad()) {
      return diff_view_meta->grad_fn_;
    }
    auto current_version = this->_version();
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
    if (impl::get_autograd_meta(*this)) {
      return impl::get_autograd_meta(*this)->grad_fn_;
    } else {
      return singleton_shared_ptr;
    }
  }
}

void Variable::remove_hook(unsigned pos) {
  auto &list = impl::materialize_autograd_meta(*this)->cpp_hooks_list;
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

namespace impl {

  AutogradMeta* materialize_autograd_meta(const Variable& self) {
    auto p = self.unsafeGetTensorImpl();
    if (!p->autograd_meta()) {
      p->set_autograd_meta(c10::guts::make_unique<AutogradMeta>());
    }
    return get_autograd_meta(self);
  }

  void rebase_history(const Variable& self, Edge gradient_edge) {
    AT_ASSERT(gradient_edge.function != nullptr);
    if (self.is_view()) {
      // NB: is_view() ==> get_autograd_meta()
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(get_autograd_meta(self));
      AT_ASSERT(gradient_edge.input_nr == 0);
      AT_ASSERT(gradient_edge.function);
      TORCH_CHECK(
          gradient_edge.function->num_inputs() == 1,
          "Functions which modify views in-place must return a single Variable");
      diff_view_meta->output_nr_ = gradient_edge.input_nr;
      auto copy_slices = std::make_shared<CopySlices>(
          diff_view_meta->base_, at::TensorGeometry(self), std::move(gradient_edge.function));
      set_gradient_edge(diff_view_meta->base_, {std::move(copy_slices), 0});
      self.grad_fn(); // trigger an update to the view's grad_fn
    } else {
      set_gradient_edge(self, std::move(gradient_edge));
    }
  }

  void create_cpp_hook(const Variable& self) {
    auto &list = materialize_autograd_meta(self)->cpp_hooks_list;
    list.reset(new hooks_list());
    std::unique_ptr<FunctionPreHook> hook_ptr(new CppFunctionPreHook(list, self.output_nr()));
    clear_hooks(self);
    add_hook(self, std::make_shared<CppFunctionPreHook>(list, 0));
    auto fn = self.grad_fn();
    if (fn) {
      fn->add_pre_hook(std::move(hook_ptr));
    }
  }

  void set_grad_accumulator(const Variable& self,
      std::weak_ptr<Node> grad_accumulator) {
    materialize_autograd_meta(self)->grad_accumulator_ = std::move(grad_accumulator);
  }

  std::shared_ptr<Node> try_get_grad_accumulator(const Variable& self) {
    if (get_autograd_meta(self)) {
      return get_autograd_meta(self)->grad_accumulator_.lock();
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<Node> grad_accumulator(const Variable& self) {
    auto autograd_meta = get_autograd_meta(self);
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

    c10::raw::intrusive_ptr::incref(self.unsafeGetTensorImpl());
    auto intrusive_from_this = c10::intrusive_ptr<at::TensorImpl>::reclaim(self.unsafeGetTensorImpl());
    result = std::make_shared<AccumulateGrad>(Variable(std::move(intrusive_from_this)));
    autograd_meta->grad_accumulator_ = result;
    return result;
  }

  Edge gradient_edge(const Variable& self) {
    // If grad_fn is null (as is the case for a leaf node), we instead
    // interpret the gradient function to be a gradient accumulator, which will
    // accumulate its inputs into the grad property of the variable. These
    // nodes get suppressed in some situations, see "suppress gradient
    // accumulation" below. Note that only variables which have `requires_grad =
    // True` can have gradient accumulators.
    if (const auto& gradient = self.grad_fn()) {
      return Edge(gradient, self.output_nr());
    } else {
      return Edge(grad_accumulator(self), 0);
    }
  }

  void set_gradient_edge(const Variable& self, Edge edge) noexcept {
    auto* meta = materialize_autograd_meta(self);
    meta->grad_fn_ = std::move(edge.function);
    meta->output_nr_ = edge.input_nr;
  }

  Node* grad_fn_unsafe(const Variable& self) {
    if (get_autograd_meta(self)) {
      return get_autograd_meta(self)->grad_fn_.get();
    } else {
      return nullptr;
    }
  }

  // Versions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void set_version_counter(
      const Variable& self,
      const c10::VariableVersion& version_counter) noexcept {
    self.unsafeGetTensorImpl()->set_version_counter(version_counter);
  }

  void bump_version(const Variable& self) noexcept {
    self.unsafeGetTensorImpl()->bump_version();
  }

  const c10::VariableVersion& version_counter(const Variable& self) noexcept {
    return self.unsafeGetTensorImpl()->version_counter();
  }

  // Hooks
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void add_hook(const Variable& self, std::shared_ptr<FunctionPreHook> hook) {
    materialize_autograd_meta(self)->hooks_.push_back(std::move(hook));
  }

  namespace {
    std::vector<std::shared_ptr<FunctionPreHook>> empty_singleton;
  }

  // TODO: Return an ArrayRef instead (and delete the singleton while you're at
  // it
  const std::vector<std::shared_ptr<FunctionPreHook>>& hooks(const Variable& self)
      noexcept {
    if (get_autograd_meta(self)) {
      return get_autograd_meta(self)->hooks_;
    } else {
      return empty_singleton;
    }
  }

  void clear_hooks(const Variable& self) {
    // This is a little goofy, but usually this should be a no oop
    materialize_autograd_meta(self)->hooks_.clear();
  }

  void set_name(const Variable& self, const std::string& name) {
    materialize_autograd_meta(self)->name_ = name;
  }

  // Miscellaneous
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void set_pyobj(const Variable& self, PyObject* pyobj) noexcept {
    self.unsafeGetTensorImpl()->set_pyobj(pyobj);
  }

  PyObject* pyobj(const Variable& self) noexcept {
    return self.unsafeGetTensorImpl()->pyobj();
  }

  AutogradMeta* get_autograd_meta(const Variable& self) noexcept {
    // NB: could return null
    return static_cast<AutogradMeta*>(self.unsafeGetTensorImpl()->autograd_meta());
  }

} // namespace impl

namespace {
  std::string singleton_string;
}

const std::string& Variable::name() const noexcept {
  if (impl::get_autograd_meta(*this)) {
    return impl::get_autograd_meta(*this)->name_;
  } else {
    return singleton_string;
  }
}

}} // namespace torch::autograd
