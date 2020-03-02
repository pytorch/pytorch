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
#include <cstdint>

namespace torch { namespace autograd {

struct Node;

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///                                Variable
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// A `Variable` augments a `Tensor` with the ability to interact in our
/// autograd machinery. Conceptually, `Variable`s travel along `Edge`s between
/// `Node`s in the autograd graph. A `Variable` can either be a leaf, like a
/// weight in a neural network, or an interior variable, when it is the result
/// of an operation between variables. Every `Variable` also stores another
/// `Variable` called its `grad` (gradient). If the variable is a leaf, its
/// gradient will be accumulated into this variable.
///
/// Every Tensor is a Variable, but sometimes we colloquially refer to Variables
/// that don't require gradients as Tensors (since none of the autograd
/// machinery for Variables applies).  Historically, Variables and Tensors
/// were separate concepts, but we've merged them together.
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
/// meaningful for `Variable` relations that are relevant to autograd.
/// See NOTE [ Autograd View Variables ] for more details.
///
///
///                               Interface
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// `Variable` inherits from `Tensor` and thus its API is a superset of that of
/// `Tensor`. This means you can perform all the usual mathematical and other
/// operations you can perform on `Tensor`s also on `Variable`s. Furthermore,
/// `Variable` and `Tensor` actually convert implicitly between each other. You
/// can thus call functions defined on `Tensor`s also with `Variable`s. For
/// this, the `Variable` class allows implicit construction from `Tensor`.
///
/// Our intention is to eliminate the Variable class in the near future, or make
/// it so that only internal code uses it to do internal operations.
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct AutogradMeta;
struct DifferentiableViewMeta;

using Variable = at::Tensor;

// Private-ish functions for manipulating variables; we don't want to put them
// on Tensor proper
namespace impl {

  // WARNING: This may return a nullptr.  If you require AutogradMeta to return
  // a materialized structure, use materialize_autograd_meta instead.
  TORCH_API AutogradMeta* get_autograd_meta(const Variable&);

  // Returns the current autograd meta, materializing it if it was previously
  // none.  This counts as a *mutating* operation, so do not call it on
  // "read-only" operators; in particular, this is NOT thread safe
  TORCH_API AutogradMeta* materialize_autograd_meta(const Variable&);

  /// Set the gradient accumulator of the `Variable`. This is only applicable to
  /// leaf variables. Interior variables should call `set_gradient_edge()`.
  TORCH_API void set_grad_accumulator(const Variable&, std::weak_ptr<Node> grad_accumulator);

  /// Attempts to get a pointer to the gradient accumulator of the `Variable`,
  /// if it still exists. If the gradient accumulator function has been
  /// destroyed, returns a `nullptr`.
  TORCH_API std::shared_ptr<Node> try_get_grad_accumulator(const Variable&);

  /// Gets the gradient accumulator of the `Variable` if it has one, or else
  /// create one on the fly and return it.
  TORCH_API std::shared_ptr<Node> grad_accumulator(const Variable&);

  /// Returns the "canonical" gradient edge of this `Variable`, i.e. either the
  /// gradient function if this is an interior `Variable`, or the gradient
  /// accumulator otherwise. If the `Variable` is interior, the returned `Edge`
  /// will store the input index of the `Node` to which this variable is
  /// connected in its `input_nr` field. For leaves, the `input_nr` is always
  /// zero. Note that `set_gradient_edge` and `gradient_edge` are not
  /// symmetric. You must use `set_gradient_edge` to set the `grad_fn` and
  /// `set_grad_accumulator` to set the accumulator.
  TORCH_API Edge gradient_edge(const Variable&);

  /// Set the gradient edge -- i.e. `grad_fn` and `input_nr` -- of the
  /// `Variable`.
  /// NOTE: This will always set the `grad_fn`, even if this is a leaf variable,
  /// and never the `grad_accumulator`. For the latter, use
  /// `set_grad_accumulator`. This allows late construction of an interior
  /// `Variable`.
  TORCH_API void set_gradient_edge(const Variable&, Edge edge);

  // Autograd Graph Interaction
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Update the `grad_fn` of an existing Variable. Called after in-place
  /// modifications.
  ///
  /// For View Variables:
  /// Called after in-place modifications. Modifies the grad_fn of the base
  /// Variable.
  TORCH_API void rebase_history(const Variable&, Edge gradient_edge);

  /// Gets the raw gradient function pointer, whatever it currently is.
  TORCH_API Node* grad_fn_unsafe(const Variable&);

  /// Increments the version count of this `Variable`.
  TORCH_API void bump_version(const Variable&);
  TORCH_API void set_version_counter(const Variable&, const c10::VariableVersion& version_counter);

  /// Retrieves this `Variable`s version counter.
  TORCH_API const c10::VariableVersion& version_counter(const Variable&);

  TORCH_API PyObject* pyobj(const Variable&);
  TORCH_API void set_pyobj(const Variable&, PyObject* pyobj);

  TORCH_API void set_name(const Variable&, const std::string& name);

  TORCH_API void add_hook(const Variable&, std::shared_ptr<FunctionPreHook> hook);
  TORCH_API const std::vector<std::shared_ptr<FunctionPreHook>>& hooks(const Variable&);
  TORCH_API void clear_hooks(const Variable&);

  TORCH_API void create_cpp_hook(const Variable&);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            AutogradMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Each `Variable` has one unique `AutogradMeta` struct, which stores autograd
/// metadata fields that are necessary for tracking the Variable's autograd history.
/// As an optimization, a Variable may store a nullptr, in lieu of a default
/// constructed AutogradMeta.

struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;

  std::vector<std::shared_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list;

  // Only meaningful on leaf variables (must be false otherwise)
  bool requires_grad_;

  // Only meaningful on non-leaf variables (must be false otherwise)
  bool retains_grad_;

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
  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) override {
    TORCH_CHECK(
      !requires_grad || at::isFloatingType(at::typeMetaToScalarType(self_impl->dtype())),
      "Only Tensors of floating point dtype can require gradients");
    requires_grad_ = requires_grad;
  }

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_;
  }

  /// Accesses the gradient `Variable` of this `Variable`.
  Variable& grad() override {
    return grad_;
  }

  const Variable& grad() const override {
    return grad_;
  }

  AutogradMeta(at::TensorImpl* self_impl = nullptr, bool requires_grad = false, Edge gradient_edge = Edge() ) {
    grad_fn_ = std::move(gradient_edge.function);
    requires_grad_ = false;
    retains_grad_ = false;
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
///   + if a single autograd Node returns multiple differentiable views, if any
///     output is modified by an inplace operation, the autograd engine will make
///     an equivalent graph (corresponding to the view operations) without using
///     equivalent graph, where each output is treated as if it were produced by a
///     distinct view operation. This discards the original (e.g., user provided)
///     grad_fn. If the provided grad_fn does more than the backward of the view,
///     then the DifferentiableViewMeta must be created with creation_meta=
///     CreationMeta::MULTI_OUTPUT_NODE to prevent the engine from ignoring the
///     provided grad_fn.
///
/// Interaction with GradMode:
/// The particular case that we consider here is:
///
///     # Have:
///     #   base.requires_grad = True or False
///     with torch.no_grad():
///         view = base[1]
///     base.requires_grad_()
///     view.copy_(var)
///     torch.autograd.grad(base.sum(), var)  <- what should it return?
///
/// Given that this particular code example is ambiguous and can easily be replace by
/// either moving both inside the no_grad block or both outside, we explicitly forbid
/// it. For now, it is deprecated by a warning. This is achieved by setting
/// creation_meta=CreationMeta::NO_GRAD_MODE for all differentiable views created
/// in no_grad mode.
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
///
/// Relevant logic for both differentiable and non-differentiable views is implemented in
/// make_variable_(non_)differentiable_view below, and wrap_output of gen_variable_type.py.


/// NOTE [ View + Inplace detection ]
///
/// We want to detect views followed by inplace as they are often forbidden to ensure
/// correctness of the computed gradients. But since we want to only notify the user
/// when both happen, we tag the DifferentiableViewMeta when the view is created
/// via the `make_variable_*_view()` functions. This tag is then checked by the
/// `check_inplace()` function from `VariableTypeUtils.h` that should be called before
/// every inplace operation and to detect cases where other views are modified and this
/// one is rebased by side effect, we also check in the `VariableHooks::grad_fn()`.

/// Flag that gives more information about when this view was created:
/// - IN_CUSTOM_FUNCTION should be set when the view is created inside a custom
///   autograd Function is returned.
/// - NO_GRAD_MODE should be set when a view in created when GradMode is disabled
/// - MULTI_OUTPUT_NODE should be set when a Node created by codegen code returns
///   multiple differentiable views
/// - DEFAULT is for all other cases
enum class CreationMeta: uint8_t { DEFAULT, IN_CUSTOM_FUNCTION, MULTI_OUTPUT_NODE,
                                   NO_GRAD_MODE };

/// Unified function to handle error checking when rebase happens
/// indirect=true means that the caller is not doing the inplace, but the inplace happened
/// somewhere else.
TORCH_API void handle_view_on_rebase(DifferentiableViewMeta* diff_view_meta, bool indirect=false);

struct TORCH_API DifferentiableViewMeta : public AutogradMeta {
  /// The base `Variable` (never a view).
  Variable base_;

  /// The value of the version_counter at the time grad_fn was created. The
  /// grad_fn field is stale if attr_version !=
  /// version_counter.current_version().
  uint32_t attr_version;

  CreationMeta creation_meta;

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_ || (is_view_ && base_.requires_grad());
  }

  DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base,
                         CreationMeta creation_meta=CreationMeta::DEFAULT);
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
// Differentiable view. Track history with DifferentiableViewMeta.
inline Variable make_variable_differentiable_view(
    Variable base,
    at::Tensor data,
    CreationMeta creation_meta) {
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/true);
    data_impl_copy->set_autograd_meta(std::make_unique<DifferentiableViewMeta>(
      data_impl_copy.get(), std::move(base),
      creation_meta));
    return Variable(data_impl_copy);
    }
  return Variable();
}

// See NOTE [ Autograd View Variables ] for details.
// Non-differentiable view. Just share version counter.
inline Variable make_variable_non_differentiable_view(
    Variable base,
    at::Tensor data,
    bool allow_tensor_metadata_change = true) {
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
      /*version_counter=*/impl::version_counter(base),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    data_impl_copy->set_autograd_meta(nullptr);
    return Variable(data_impl_copy);
  }
  return Variable();
}

/// Creates a `Variable` from the given `Tensor`, copying its underlying `TensorImpl`.
/// `requires_grad` should be
/// set only for leaves, and determines whether the `Variable` will accumulate
/// gradients. NOTE: `data` must *not* be a `Variable` already. Its dynamic
/// type *must* be `Tensor`.
///
/// TODO: Eliminate this function as much as possible, as it can be expressed
/// more clearly as detach() or a no-op in most call sites (especially when
/// there is only one use of the variable).
inline Variable make_variable(
    at::Tensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  if (data.defined()) {
    if (data.getIntrusivePtr().use_count() == 1 && data.getIntrusivePtr()->unique_version()) {
      auto data_impl = data.getIntrusivePtr();
      data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      if (requires_grad) {
        data_impl->set_autograd_meta(std::make_unique<AutogradMeta>(data_impl.get(), requires_grad));
      } else {
        data_impl->set_autograd_meta(nullptr);
      }
      return Variable(std::move(data_impl));
    } else {
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      if (requires_grad) {
        data_impl_copy->set_autograd_meta(std::make_unique<AutogradMeta>(
          data_impl_copy.get(), requires_grad));
      } else {
        data_impl_copy->set_autograd_meta(nullptr);
      }
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
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    data_impl_copy->set_autograd_meta(std::make_unique<AutogradMeta>(
      data_impl_copy.get(), false, std::move(gradient_edge)));
    return Variable(data_impl_copy);
  }
  return Variable();
}

// Tensor Conversion
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// In the old days, these casts were checked, but now that every Tensor
// is a Variable this cast is always valid
inline Variable& as_variable_ref(at::Tensor& tensor) {
  return static_cast<Variable&>(tensor);
}

inline const Variable& as_variable_ref(const at::Tensor& tensor) {
  return static_cast<const Variable&>(tensor);
}

}} // namespace torch::autograd
