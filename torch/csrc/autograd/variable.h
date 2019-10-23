#pragma once

#include <torch/csrc/utils/python_stub.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/cpp_hook.h>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

using variable_list = std::vector<at::Tensor>;

struct Node;

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///                                Variable
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// A `Variable` is a `Tensor` augmented with the ability to interact in our
/// autograd machinery. Conceptually, `Variable`s travel along `Edge`s between
/// `Node`s in the autograd graph. A `Variable` can either be a leaf, like a
/// weight in a neural network, or an interior variable, when it is the result
/// of an operation between variables. Every `Variable` also stores another
/// `Variable` called its `grad` (gradient). If the variable is a leaf, its
/// gradient will be accumulated into this variable.
///
/// Variables are just another type of Tensor; a Tensor you have in your hand
/// could very well be a Variable.  Some tensors are NOT variables, although
/// we hope to eliminate this case in the near future.
///
///                              Gradient Edges
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Furthermore, `Variable`s have the notion of a `gradient_edge`, which is the
/// edge in the autograd graph that connects the variable to a particular input
/// of the gradient function that will be invoked with the variable during the
/// backward pass. More precisely, this gradient function can be one of two
/// things:
/// 1. A `grad_fn`, if the variable is in the interior of the graph. This is the
///    gradient of the function that produced the variable.
/// 2. A `grad_accumulator`, if the variable is a leaf, which accumulates a
///    scalar gradient value into its `grad` variable.
///
///                               Versioning
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Another major feature of `Variable`s are *versions*. Versions are
/// incremented when an in-place mutation of a variable occurs. Versions are
/// useful when constructing `SavedVariable`s, which take a snapshot of a
/// `Variable` at a certain version. You can retrieve a `Variable`'s version
/// through its `current_version()` method.
///
///                                 Views
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// It is possible for a  `Variable` to be a *view* of another `Variable`, in
/// which case it tracks that `Variable`'s data and autograd history. Beyond
/// construction, the interface of a view is identical to that of a regular
/// `Variable`. You can determine whether `Variable` is in fact a view by
/// probing its `is_view()` method. Note that the *view* semantics are only
/// meaningful for `Variable` relations that are relevant to autograd. For
/// example, if you hide your code from autograd using `.no_grad()`, the
/// `Variable`s will not be registered as having view relations, even if they
/// share storage.
/// See NOTE [ Autograd View Variables ] for more details.


// NB: Historically Variable had a separate static type from Tensor, providing
// an extended interface that isn't valid on all Tensors (because Variables
// "wrapped" Tensors).  When we merged Variable and Tensor, this distinction
// became less meaningful; however, we still define Variable as an alias for
// Tensor for backwards compatibility.
using Variable = at::Tensor;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            AutogradMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Each `Variable` has one unique `AutogradMeta` struct, which stores autograd
/// metadata fields that are necessary for tracking the Variable's autograd history.

struct TORCH_API AutogradMeta {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;

  std::vector<std::shared_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list;

  // Only meaningful on leaf variables (must be false otherwise)
  bool requires_grad_;

  bool is_view_;

  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  uint32_t output_nr_;

  // Mutex to ensure that concurrent read operations that modify internal
  // state are still thread-safe. Used by grad_fn() and
  // grad_accumulator().
  std::mutex mutex_;

  /// Sets the `requires_grad` property of `Variable`. This should be true for
  /// leaf variables that want to accumulate gradients, and false for all other
  /// variables.
  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) {
    TORCH_CHECK(
      !requires_grad || at::isFloatingType(at::typeMetaToScalarType(self_impl->dtype())),
      "Only Tensors of floating point dtype can require gradients");
    requires_grad_ = requires_grad;
  }

  // TODO: devirtualize this guy
  virtual bool requires_grad() const {
    return requires_grad_ || grad_fn_;
  }

  /// Accesses the gradient `Variable` of this `Variable`.
  Variable& grad() {
    return grad_;
  }

  const Variable& grad() const {
    return grad_;
  }

  AutogradMeta(
    at::TensorImpl* self_impl,
    bool requires_grad = false,
    Edge gradient_edge = Edge());
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     DifferentiableViewMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// NOTE [ Autograd View Variables ]
///
/// Many operations return Variable that shares storage with an input Variable.
/// The returned Variable is called a **view** Variable on the input **base**
/// Variable.
///
/// In PyTorch, we have two types of views: differentiable views, and
/// non-differentiable views. In either type, to support proper version
/// checking, the base and view Variables must always share the same
/// version_counter.
///
///
/// Differentiable Views
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Differentiable views are the view variables where you want gradients to flow
/// back to the base variables. Out-of-place operations on views are quite
/// straightforward, but in-place ones are very tricky. Even if the base
/// variable may not require grad when we create the view, we still need to
/// track the view relation because future in-place ops may require back-proping
/// through it. For example, we need to support
///
///   (1) in-place operation on view, e.g.,
///
///     # Have:
///     #   base.requires_grad = False
///     #   var.requires_grad = True
///     base[1] = var  # i.e., base[1].copy_(var)
///     torch.autograd.grad(base.sum(), var)  <- should return an all ones tensor
///
///   (2) in-place operation on base after view is created, e.g.,
///
///     # Have:
///     #   base.requires_grad = False
///     #   var.requires_grad = True
///     view = base[1]
///     base.copy_(var)
///     torch.autograd.grad(view.sum(), var)  <- should return a tensor with
///                                              var[1] filled with all ones and
///                                              zeros everywhere else
///
/// DifferentiableViewMeta is created to support gradient tracking of
/// such **in-place** operations. In particular,
///   + if an in-place op is done on base, the grad_fn field of the view may
///     become stale. So accesses should always go through grad_fn(), which
///     reconstructs an updated grad_fn if the version_counter has incremented.
///     All other fields are always valid.
///   + if an in-place op is done on view, in rebase_history() of view, which is
///     called after every in-place op in VariableType.cpp, the grad_fn of base
///     is updated.
///
///
/// Non-Differentiable Views
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// In certain cases, although function outputs share storage with inputs, they
/// will **never** require gradient history tracking. Instead of registering the
/// view relation via DifferentiableViewMeta in autograd, the views will be
/// using usual AutogradMeta and just share the version counters with the base
/// Variables.
/// Such views include:
///   1. Views created from .detach()
///   2. Views that are non-differentiable by its nature.
///      E.g., `sparse_tensor.indices()` is a integral view on a (possibly)
///      floating point tensor.
///      See top of `derivatives.yaml` on how to specify that outputs of a
///      function are non-differentiable.
/// These are called non-differentiable views as the gradients do not flow
/// through the view relation.
/// Relevant logic for non-differentiable views is implemented in
/// make_variable_view below, and wrap_output of gen_variable_type.py.
struct TORCH_API DifferentiableViewMeta : public AutogradMeta {
  /// The base `Variable` (never a view).
  Variable base_;

  /// The value of the version_counter at the time grad_fn was created. The
  /// grad_fn field is stale if attr_version !=
  /// version_counter.current_version().
  uint32_t attr_version;

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_ || (is_view_ && base_.requires_grad());
  }

  DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base, Edge gradient_edge);
  ~DifferentiableViewMeta();
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        Variable Implementation
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Factory Functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a `Variable` that is a *view* of another (*base*) variable.
/// The `gradient_edge` is an optional (gradient_function, input_number) pair.
/// `is_differentiable` is a bool that specifies whether this view is
/// differentiable, i.e., whether the relation should be tracked by autograd.
/// See NOTE [ Autograd View Variables ] for details.

/// NOTE: `allow_tensor_metadata_change` is set to true by default, because there
/// are a lot of call sites to these factory functions that need to change the
/// variable's size or storage afterwards, and they don't expect the original
/// tensor (where the variable is created from) to be updated. Setting
/// `allow_tensor_metadata_change_` to false by default would unnecessarily
/// prevent those changes from happening and is undesirable.

// See NOTE [ Autograd View Variables ] for details.
inline Variable make_variable_view(
    Variable base,
    at::Tensor data,
    bool is_differentiable = true,
    bool allow_tensor_metadata_change = true,
    Edge gradient_edge = Edge()) {
  if (data.defined()) {
    if (is_differentiable) {
      /// Differentiable view. Track history with DifferentiableViewMeta.
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      data_impl_copy->set_autograd_meta(c10::guts::make_unique<DifferentiableViewMeta>(
        data_impl_copy.get(), std::move(base), std::move(gradient_edge)));
      return Variable(data_impl_copy);
    } else {
      /// Non-differentiable view. Just share version counter.
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/base.version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      data_impl_copy->set_autograd_meta(c10::guts::make_unique<AutogradMeta>(
        data_impl_copy.get(), false, std::move(gradient_edge)));
      return Variable(data_impl_copy);
    }
  }
  return Variable();
}

/// Creates a `Variable` from the given `Tensor`, copying its underlying `TensorImpl`.
/// `requires_grad` should be
/// set only for leaves, and determines whether the `Variable` will accumulate
/// gradients. NOTE: `data` must *not* be a `Variable` already. Its dynamic
/// type *must* be `Tensor`.
inline Variable make_variable(
    at::Tensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  TORCH_CHECK(
      !data.is_variable(),
      "Must not create a new variable from a variable, use its .tensor_data()");
  if (data.defined()) {
    if (data.getIntrusivePtr().use_count() == 1 && data.getIntrusivePtr()->unique_version()) {
      auto data_impl = data.getIntrusivePtr();
      data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      data_impl->set_autograd_meta(c10::guts::make_unique<AutogradMeta>(data_impl.get(), requires_grad));
      return Variable(std::move(data_impl));
    } else {
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      data_impl_copy->set_autograd_meta(c10::guts::make_unique<AutogradMeta>(
        data_impl_copy.get(), requires_grad));
      return Variable(data_impl_copy);
    }
  }
  return Variable();
}

/// Creates a `Variable` from the given `Tensor`, copying its underlying `TensorImpl`.
/// `gradient_edge` should be a (function, input_nr) pair specifying the function
/// in the autograd graph, and what particular input of that function, this
/// variable is connected to.
inline Variable make_variable(
    at::Tensor data,
    Edge gradient_edge,
    bool allow_tensor_metadata_change = true) {
  TORCH_CHECK(
      !data.is_variable(),
      "Must not create a new variable from a variable, use its .tensor_data()");
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    data_impl_copy->set_autograd_meta(c10::guts::make_unique<AutogradMeta>(
      data_impl_copy.get(), false, std::move(gradient_edge)));
    return Variable(data_impl_copy);
  }
  return Variable();
}

// Tensor Conversion
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Tests if a `Tensor` reference actually is a `Variable`, and returns the ref
/// if so.  (Historically, this also did a static cast, but now that Tensor
/// and Variable are the same type, this is a no-op.)
///
/// TODO: Remove this when the dynamic distinction between Variable and Tensor
/// is removed.
inline Variable& as_variable_ref(at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_variable(),
      "Attempted to cast a Tensor to a Variable, but "
      "the dynamic type of the value is not Variable.");
  return tensor;
}

inline const Variable& as_variable_ref(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_variable(),
      "Attempted to cast a Tensor to a Variable, but "
      "the dynamic type of the value is not Variable.");
  return tensor;
}

TORCH_API void _create_cpp_hook(const at::Tensor& self);

}} // namespace torch::autograd

// NB: These template definitions shouldn't live here, but they require access
// to the structure of AutogradMeta, so for now they live here.  Once
// AutogradMeta is available inline, move these back to Tensor.h
namespace at {

template <typename T>
auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_void_t<T> {
  TORCH_CHECK(requires_grad(), "cannot register a hook on a variable that "
                           "doesn't require gradient");
  auto &list = get_autograd_meta()->cpp_hooks_list;
  if(!list) {
    torch::autograd::_create_cpp_hook(*this);
  }
  unsigned idx = list->size();
  // Return the grad argument in case of a hook with void return type to have an
  // std::function with Variable return type
  std::function<void(Tensor)> fn(hook);
  list->emplace_back([fn](Tensor grad){
   fn(grad);
    return Tensor();});
  return idx;
}

template <typename T>
auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_var_t<T> {
  TORCH_CHECK(requires_grad(), "cannot register a hook on a variable that "
                           "doesn't require gradient");
  auto &list = get_autograd_meta()->cpp_hooks_list;
  if(!list) {
    torch::autograd::_create_cpp_hook(*this);
  }
  unsigned idx = list->size();
  list->push_back(hook);
  return idx;
}

} // namespace at
