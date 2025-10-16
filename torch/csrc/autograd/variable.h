#pragma once

#include <torch/csrc/utils/python_stub.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/cpp_hook.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/forward_grad.h>
#include <torch/csrc/autograd/function_hook.h>

#include <ATen/NamedTensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/VariableHooksInterface.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace torch::autograd {

/// `Variable` is exactly the same as `Tensor` (i.e. we have `using Variable =
/// at::Tensor`). This means you can perform all the usual mathematical and
/// other operations you can perform on `Tensor`s also on `Variable`s.
///
/// The only reason we are keeping the `Variable` class is backward
/// compatibility with external user's legacy C++ frontend code. Our intention
/// is to eliminate the `Variable` class in the near future.
using Variable = at::Tensor;

} // namespace torch::autograd

// The following are all internal APIs and should not be shown in libtorch docs.
// Therefore, we wrap the following code with `#ifndef DOXYGEN_SHOULD_SKIP_THIS
// ... #endif`

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace torch::autograd {

/// Check if this type is supported by the autograd engine.
/// If you change this, update the doc at the top of the
/// torch/autograd/__init__.py file and
/// "test_set_requires_grad_only_for_continuous_types" in test/test_autograd.py
inline bool isDifferentiableType(at::ScalarType t) {
  return isFloatingType(t) || isComplexType(t);
}

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
/// were separate concepts, but now they are exactly the same (i.e. we have
/// `using Variable = at::Tensor`).
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
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct AutogradMeta;
struct DifferentiableViewMeta;

// Private-ish functions for manipulating variables; we don't want to put them
// on Tensor proper
namespace impl {

// WARNING: This may return a nullptr.  If you require AutogradMeta to return
// a materialized structure, use materialize_autograd_meta instead.
TORCH_API AutogradMeta* get_autograd_meta(const at::TensorBase& /*self*/);

// WARNING: This will return a nullptr if the Tensor is not a view.
TORCH_API DifferentiableViewMeta* get_view_autograd_meta(
    const at::TensorBase& /*self*/);

// Returns the current autograd meta, materializing it if it was previously
// none.  This counts as a *mutating* operation, so do not call it on
// "read-only" operators; in particular, this is NOT thread safe
TORCH_API AutogradMeta* materialize_autograd_meta(
    const at::TensorBase& /*self*/);

/// Set the gradient accumulator of the `Variable`. This is only applicable to
/// leaf variables. Interior variables should call `set_gradient_edge()`.
TORCH_API void set_grad_accumulator(
    const Variable& /*self*/,
    std::weak_ptr<Node> grad_accumulator);

/// Attempts to get a pointer to the gradient accumulator of the `Variable`,
/// if it still exists. If the gradient accumulator function has been
/// destroyed, returns a `nullptr`.
TORCH_API std::shared_ptr<Node> try_get_grad_accumulator(
    const Variable& /*self*/);
TORCH_API std::shared_ptr<Node> try_get_grad_accumulator(
    const at::TensorBase& /*self*/);

/// Gets the gradient accumulator of the `Variable` if it has one, or else
/// create one on the fly and return it.
TORCH_API std::shared_ptr<Node> grad_accumulator(const Variable& /*self*/);

/// Returns the "canonical" gradient edge of this `Variable`, i.e. either the
/// gradient function if this is an interior `Variable`, or the gradient
/// accumulator otherwise. If the `Variable` is interior, the returned `Edge`
/// will store the input index of the `Node` to which this variable is
/// connected in its `input_nr` field. For leaves, the `input_nr` is always
/// zero. Note that `set_gradient_edge` and `gradient_edge` are not
/// symmetric. You must use `set_gradient_edge` to set the `grad_fn` and
/// `set_grad_accumulator` to set the accumulator.
TORCH_API Edge gradient_edge(const Variable& /*self*/);

/// Set the gradient edge -- i.e. `grad_fn` and `input_nr` -- of the
/// `Variable`.
/// NOTE: This will always set the `grad_fn`, even if this is a leaf variable,
/// and never the `grad_accumulator`. For the latter, use
/// `set_grad_accumulator`. This allows late construction of an interior
/// `Variable`.
TORCH_API void set_gradient_edge(const Variable& /*self*/, Edge edge);

// Autograd Graph Interaction
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Update the `grad_fn` of an existing Variable. Called after in-place
/// modifications.
///
/// For View Variables:
/// Called after in-place modifications. Modifies the grad_fn of the base
/// Variable.
TORCH_API void rebase_history(const Variable& /*self*/, Edge gradient_edge);

/// Gets the raw gradient function pointer, whatever it currently is.
TORCH_API Node* grad_fn_unsafe(const Variable& /*self*/);

/// Increments the version count of this `Variable`.
TORCH_API void bump_version(const Variable& /*self*/);
TORCH_API void set_version_counter(
    const Variable& /*self*/,
    const c10::VariableVersion& version_counter);

/// Retrieves this `Variable`s version counter.
TORCH_API const c10::VariableVersion& version_counter(const Variable& /*self*/);

TORCH_API void set_name(const Variable& /*self*/, const std::string& name);

TORCH_API void add_hook(
    const at::TensorBase& /*self*/,
    std::unique_ptr<FunctionPreHook> hook);
TORCH_API std::vector<std::unique_ptr<FunctionPreHook>>& hooks(
    const Variable& /*self*/);
TORCH_API void clear_hooks(const at::TensorBase& /*self*/);

TORCH_API void set_post_acc_grad_hooks(
    const at::TensorBase& /*self*/,
    std::unique_ptr<PostAccumulateGradHook> dict);
TORCH_API std::unique_ptr<PostAccumulateGradHook>& post_acc_grad_hooks(
    const Variable& /*self*/);

TORCH_API void create_cpp_hook(
    const at::TensorBase& /*self*/,
    bool is_retains_grad_hooks = false);
} // namespace impl

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            AutogradMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Each `Variable` has one unique `AutogradMeta` struct, which stores autograd
/// metadata fields that are necessary for tracking the Variable's autograd
/// history. As an optimization, a Variable may store a nullptr, in lieu of a
/// default constructed AutogradMeta.

struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;

  // This field is used to store all the forward AD gradients
  // associated with this AutogradMeta (and the Tensor it corresponds to)
  // There is a semantic 1:1 correspondence between AutogradMeta and
  // ForwardGrad but:
  //   - This field is lazily populated.
  //   - This field is a shared_ptr but it must never be
  //     shared by multiple Tensors. See Note [ Using ForwardGrad ]
  // Any transition from not_initialized to initialized
  // must be protected by mutex_
  mutable std::shared_ptr<ForwardGrad> fw_grad_;

  // The hooks_ field is actually reused by both python and cpp logic
  // For both cases, we have a data structure, cpp_hooks_list_ (cpp)
  // or dict (python) which is the canonical copy.
  // Then, for both cases, we always register a single hook to
  // hooks_ which wraps all the hooks in the list/dict.
  // And, again in both cases, if the grad_fn exists on that tensor
  // we will additionally register a single hook to the grad_fn.
  //
  // Note that the cpp and python use cases aren't actually aware of
  // each other, so using both is not defined behavior.
  std::vector<std::unique_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;

  // The post_acc_grad_hooks_ field stores only Python hooks
  // (PyFunctionTensorPostAccGradHooks) that are called after the
  // .grad field has been accumulated into. This is less complicated
  // than the hooks_ field, which encapsulates a lot more.
  std::unique_ptr<PostAccumulateGradHook> post_acc_grad_hooks_ = nullptr;

  // Only meaningful on leaf variables (must be false otherwise)
  bool requires_grad_{false};

  // Only meaningful on non-leaf variables (must be false otherwise)
  bool retains_grad_{false};

  bool is_view_{false};

  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  uint32_t output_nr_;

  // The dtype of the grad field; when nullopt, defaults to tensor's dtype.
  std::optional<at::ScalarType> grad_dtype_;

  // When true, allows gradient dtype to be different from tensor dtype,
  // bypassing dtype casting and validation in the autograd engine.
  bool allow_grad_dtype_mismatch_{false};

  // Mutex to ensure that concurrent read operations that modify internal
  // state are still thread-safe. Used by grad_fn(), grad_accumulator(),
  // fw_grad() and set_fw_grad()
  // This is mutable because we need to be able to acquire this from const
  // version of this class for the functions above
  mutable std::mutex mutex_;

  /// Sets the `requires_grad` property of `Variable`. This should be true for
  /// leaf variables that want to accumulate gradients, and false for all other
  /// variables.
  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) final {
    TORCH_CHECK(
        !requires_grad ||
            isDifferentiableType(at::typeMetaToScalarType(self_impl->dtype())),
        "Only Tensors of floating point and complex dtype can require gradients");
    requires_grad_ = requires_grad;
  }

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_;
  }

  /// Accesses the gradient `Variable` of this `Variable`.
  Variable& mutable_grad() override {
    return grad_;
  }

  const Variable& grad() const override {
    return grad_;
  }

  const Variable& fw_grad(uint64_t level, const at::TensorBase& self)
      const override;

  void set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op) override;

  std::optional<at::ScalarType> grad_dtype(const at::TensorBase& self) const;

  void set_grad_dtype(
      const std::optional<at::ScalarType>& grad_dtype,
      const at::TensorBase& self);

  AutogradMeta(
      at::TensorImpl* self_impl = nullptr,
      bool requires_grad = false,
      Edge gradient_edge = Edge())
      : grad_fn_(std::move(gradient_edge.function)),

        output_nr_(gradient_edge.input_nr) {
    // set_requires_grad also checks error conditions.
    if (requires_grad) {
      TORCH_INTERNAL_ASSERT(self_impl);
      set_requires_grad(requires_grad, self_impl);
    }
    TORCH_CHECK(
        !grad_fn_ || !requires_grad_,
        "requires_grad should be false if grad_fn is set");
  }

  ~AutogradMeta() override {
    // If AutogradMeta is being destroyed, it means that there is no other
    // reference to its corresponding Tensor. It implies that no other thread
    // can be using this object and so there is no need to lock mutex_ here to
    // guard the check if fw_grad_ is populated.
    if (fw_grad_) {
      // See note [ Using ForwardGrad ]
      fw_grad_->clear();
    }
  }
};

/// Base class for view functions, providing reapplication of a view on a new
/// base. Each view op should get a codegenerated subclass of this class
/// containing any state needed to reconstruct the view. The class also provides
/// convenience accessors for saved SymInts / tensor state. This is useful for
/// e.g. fake-ification, where we want to use symbolic values or fake tensors
/// instead.
struct TORCH_API ViewFunc {
  virtual ~ViewFunc() = default;
  /// Returns any SymInts in the saved state.
  virtual std::vector<c10::SymInt> get_symints() const {
    return {};
  }
  /// Returns the number of SymInts in the saved state.
  virtual size_t num_symints() const {
    return 0;
  }
  /// Returns any tensors in the saved state.
  virtual std::vector<at::Tensor> get_tensors() const {
    return {};
  }
  /// Returns the number of tensors in the saved state.
  virtual size_t num_tensors() const {
    return 0;
  }
  /// Reapplies the view on the given base using the saved state.
  virtual at::Tensor operator()(const at::Tensor&) const = 0;
  /// Returns a clone of this ViewFunc, optionally with the specified saved
  /// state.
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = std::nullopt,
      std::optional<std::vector<at::Tensor>> = std::nullopt) const = 0;

 protected:
  /// Sets the values of any SymInts in the saved state. The input vector size
  /// must match the number of SymInts in the saved state (i.e. the size of the
  /// list returned by get_symints()).
  /// NOLINTNEXTLINE(performance-unnecessary-value-param)
  virtual void set_symints(std::vector<c10::SymInt> /*unused*/) {}
  /// Sets the values of any Tensors in the saved state. The input vector size
  /// must match the number of Tensors in the saved state (i.e. the size of the
  /// list returned by get_tensors()).
  /// NOLINTNEXTLINE(performance-unnecessary-value-param)
  virtual void set_tensors(std::vector<at::Tensor> /*unused*/) {}
};

/// ViewFunc that represents a chain of two ViewFuncs.
struct ChainedViewFunc : public ViewFunc {
  ChainedViewFunc(
      std::unique_ptr<ViewFunc> first,
      std::unique_ptr<ViewFunc> second)
      : first(std::move(first)), second(std::move(second)) {}
  ~ChainedViewFunc() override = default;
  std::vector<c10::SymInt> get_symints() const override;
  size_t num_symints() const override {
    return first->num_symints() + second->num_symints();
  }
  std::vector<at::Tensor> get_tensors() const override;
  size_t num_tensors() const override {
    return first->num_tensors() + second->num_tensors();
  }
  at::Tensor operator()(
      const at::Tensor& /*input_base*/ /*unused*/) const override;
  std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> /*symints*/ /*unused*/ =
          std::nullopt,
      std::optional<std::vector<at::Tensor>> /*tensors*/ /*unused*/ =
          std::nullopt) const override;

 private:
  std::unique_ptr<ViewFunc> first;
  std::unique_ptr<ViewFunc> second;
};

/// ViewFunc that errors with a specified error message when called.
struct ErroringViewFunc : public ViewFunc {
  ErroringViewFunc(std::string error_msg) : error_msg(std::move(error_msg)) {}
  ~ErroringViewFunc() override = default;
  at::Tensor operator()(const at::Tensor& /*unused*/) const override {
    TORCH_CHECK(false, error_msg);
  }
  std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> /*unused*/ = std::nullopt,
      std::optional<std::vector<at::Tensor>> /*unused*/ =
          std::nullopt) const override {
    return std::make_unique<ErroringViewFunc>(error_msg);
  }

 private:
  std::string error_msg;
};

struct TORCH_API ViewInfo {
  /// The base `Variable`
  /// If this ViewInfo represents a forward (respectively backward) AD gradient,
  /// then this Tensor cannot be a forward (respectively backward) view.
  Variable base_;

  /// By default we use as_strided to recover views which is more efficient.
  /// view_fn is only saved when as_strided is not supported.
  /// If view_fn has value, we use it to recover views in backward.
  std::unique_ptr<ViewFunc> view_fn_;

  /// Analogue of view_fn but in reverse: given a view -> produce the base by
  /// applying the inverse view.
  std::function<Variable(const Variable&)> rev_view_fn_;

  /// Accessors for the view function
  bool has_view_fn() const {
    // assume either BOTH or NEITHER of view_fn_ and rev_view_fn_ exist
    return view_fn_ != nullptr;
  }

  const ViewFunc& view_fn() const {
    TORCH_CHECK(
        has_view_fn(), "Can only access the view function if it exists.");
    return *view_fn_;
  }

  std::function<Variable(const Variable&)> rev_view_fn() const {
    TORCH_CHECK(
        has_view_fn(),
        "Can only access the reverse view function if it exists.");
    return rev_view_fn_;
  }

  /// The chain function can be used to build a new ViewInfo for a
  /// differentiable view function. It will return a new view info that
  /// accurately represents how "tensor" is a view of this instance's "base_".
  /// The "base" and "tensor" are respectively the input and output of the
  /// differentiable view function that happened. They are required to properly
  /// set the optional view_fn_ when it is not provided. The "view_func", if
  /// provided, should be a function that allows to re-do the view between
  /// "base" and "tensor".
  ViewInfo chain(
      const Variable& base,
      const Variable& tensor,
      std::unique_ptr<ViewFunc> view_func = nullptr,
      std::function<Variable(const Variable&)> rev_view_func = nullptr) const;

  ViewInfo(
      Variable base,
      std::unique_ptr<ViewFunc> view_fn,
      std::function<Variable(const Variable&)> rev_view_fn)
      : base_(std::move(base)),
        view_fn_(std::move(view_fn)),
        rev_view_fn_(std::move(rev_view_fn)) {
    TORCH_CHECK(base_.defined(), "base is undefined");
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
/// This class allows to track both forward and backward AD differentiable
/// views. These views can have different base as non-differentiable view for
/// forward and backward mode AD are not the same.
///
/// Most function are either both forward and backward differentiable views (for
/// example: view, select, narrow, transpose, etc) or both not forward and not
/// backward differentiable views (for example: indices, values, eq, lt, etc).
/// But there are also functions that are forward but not backward
/// differentiable views (only detach for now) or functions that are backward
/// but not forward differentiable view (only make_dual and unpack dual for
/// now).
///
/// A concrete example of two views with different bases is as follow:
///
///     # Have:
///     #   dual is a dual Tensor that is neither a forward or backward view
///     detached_dual = dual.detach()
///     view = detached_dual.view_as(dual)
///     # The forward base of view is dual
///     # The backward base of view is detached_dual
///
/// - Backward Mode View
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
///     torch.autograd.grad(base.sum(), var)  <- should return an all ones
///     tensor
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
/// - Forward Mode View
/// Forward differentiable views follow the same semantic as backward ones but
/// show up differently as they are computed along with the forward evaluation.
/// The hard examples above are thus very similar
///
///   (1) in-place operation on view, e.g.,
///
///     # Have:
///     #   base is a regular Tensor
///     #   var is a dual Tensor whose tangent is all ones
///     base[1] = var  # i.e., base[1].copy_(var)
///     # Now, base is a dual Tensor
///     _, fw_grad = fwAD.unpack_dual(base) <- fw_grad should be a tensor with
///                                              fw_grad[1] filled with all ones
///                                              and zeros everywhere else
///
///   (2) in-place operation on base after view is created, e.g.,
///
///     # Have:
///     #   base is a regular Tensor
///     #   var is a dual Tensor whose tangent is all ones
///     view = base[1]
///     base.copy_(var)
///     _, fw_grad = fwAD.unpack_dual(view) <- fw_grad should be an all ones
///     tensor
///
/// See Note [Forward Grad View/inplace] for more details on how we handle these
/// hard cases.
///
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
///     output is modified by an inplace operation, the autograd engine will
///     make an equivalent graph (corresponding to the view operations) without
///     using equivalent graph, where each output is treated as if it were
///     produced by a distinct view operation. This discards the original (e.g.,
///     user provided) grad_fn. If the provided grad_fn does more than the
///     backward of the view, then the DifferentiableViewMeta must be created
///     with creation_meta= CreationMeta::MULTI_OUTPUT_NODE to prevent the
///     engine from ignoring the provided grad_fn.
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
/// Given that this particular code example is ambiguous and can easily be
/// replace by either moving both inside the no_grad block or both outside, we
/// explicitly forbid it. For now, it is deprecated by a warning. This is
/// achieved by setting creation_meta=CreationMeta::NO_GRAD_MODE for all
/// differentiable views created in no_grad mode.
///
/// See Note [View + Inplace update for base tensor]
/// and Note [View + Inplace update for view tensor] for the details how
/// autograd handles inplace update with view ops.
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
/// Relevant logic for both differentiable and non-differentiable views is
/// implemented in make_variable_(non_)differentiable_view below, and
/// wrap_output of gen_variable_type.py.

/// NOTE [ View + Inplace detection ]
///
/// We want to detect views followed by inplace as they are often forbidden to
/// ensure correctness of the computed gradients. But since we want to only
/// notify the user when both happen, we tag the DifferentiableViewMeta when the
/// view is created via the `make_variable_*_view()` functions. This tag is then
/// checked by the `check_inplace()` function from `VariableTypeUtils.h` that
/// should be called before every inplace operation and to detect cases where
/// other views are modified and this one is rebased by side effect, we also
/// check in the `VariableHooks::grad_fn()`.

/// Flag that gives more information about when this view was created:
/// - IN_CUSTOM_FUNCTION should be set when the view is created inside a custom
///   autograd Function is returned.
/// - NO_GRAD_MODE should be set when a view in created when GradMode is
/// disabled
/// - MULTI_OUTPUT_NODE should be set when a Node created by codegen code
/// returns
///   multiple differentiable views
/// - Inference_MODE should be set when a view of normal tensor is created in
/// InferenceMode.
/// - DEFAULT is for all other cases
enum class CreationMeta : uint8_t {
  DEFAULT,
  IN_CUSTOM_FUNCTION,
  MULTI_OUTPUT_NODE,
  NO_GRAD_MODE,
  INFERENCE_MODE
};

/// Handles correctly propagating CreationMeta when a new view is created from a
/// previous view. In general, we don't want the new view to be _less_
/// restrictive than the previous view (it's okay to be _more_ restrictive). A
/// CreationMeta value of DEFAULT is currently the least restrictive, as the
/// behavior for all other CreationMeta values is to error out for in-place ops.
/// A CreationMeta value of INFERENCE_MODE is currently the most restrictive, so
/// it takes precedence in propagation. If this changes, the logic here will
/// need to be updated to properly handle the new semantics.
inline CreationMeta propagate_creation_meta(
    CreationMeta prev_view_creation_meta,
    CreationMeta new_view_creation_meta) {
  return (new_view_creation_meta == CreationMeta::DEFAULT)
      ? prev_view_creation_meta
      : (prev_view_creation_meta == CreationMeta::INFERENCE_MODE
             ? prev_view_creation_meta
             : new_view_creation_meta);
}

/// Unified function to handle error checking when rebase happens
/// indirect=true means that the caller is not doing the inplace, but the
/// inplace happened somewhere else.
TORCH_API void handle_view_on_rebase(
    DifferentiableViewMeta* diff_view_meta,
    bool indirect = false);

struct TORCH_API DifferentiableViewMeta : public AutogradMeta {
 private:
  /// Information about the views
  std::optional<ViewInfo> backward_info_;
  std::optional<ViewInfo> forward_info_;

  // Optimization to reduce the number of ViewInfo we create.
  // In the (very common) case where backward_info_ == forward_info_, we only
  // populate backward_info_ (that should be used as both the forward and
  // backward view information) and set shared_view_info_ = true. Invariants:
  //   - If shared_view_info_ is false, there is no special constraints on
  //     backward_info_ and forward_info_
  //   - If shared_view_info_ is true, we must have:
  //      - backward_info_.has_value() == true
  //      - forward_info_.has_value() == false
  bool shared_view_info_;

  /// The two following fields are extra information that we track to ensure
  /// that any operation on this backward view is valid.

  /// The value of the version_counter at the time grad_fn was created. The
  /// grad_fn field is stale if attr_version_ !=
  /// version_counter.current_version().
  uint32_t attr_version_;
  CreationMeta creation_meta_;

 public:
  /// requires_grad is a backward AD field so we only use the view specific
  /// logic for backward differentiable views
  bool requires_grad() const override {
    return requires_grad_ || grad_fn_ ||
        (has_bw_view() && get_backward_view().base_.requires_grad());
  }

  bool shared_view_info() const {
    return shared_view_info_;
  }

  bool has_bw_view() const {
    return backward_info_.has_value();
  }

  const ViewInfo& get_backward_view() const {
    TORCH_CHECK(
        has_bw_view(), "backward view info can only exist for backward views.");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return backward_info_.value();
  }

  uint32_t get_attr_version() const {
    TORCH_CHECK(
        has_bw_view(), "attr_version can only exist for backward views.");
    return attr_version_;
  }

  void set_attr_version(uint32_t new_attr_version) {
    TORCH_CHECK(
        has_bw_view(), "attr_version can only exist for backward views.");
    attr_version_ = new_attr_version;
  }

  CreationMeta get_creation_meta() const {
    TORCH_CHECK(
        has_bw_view(), "creation_meta can only exist for backward views.");
    return creation_meta_;
  }

  void set_creation_meta(CreationMeta new_creation_meta) {
    TORCH_CHECK(
        has_bw_view(), "creation_meta can only exist for backward views.");
    creation_meta_ = new_creation_meta;
  }

  bool has_fw_view() const {
    return shared_view_info_ || forward_info_.has_value();
  }

  const ViewInfo& get_forward_view() const {
    TORCH_CHECK(
        has_fw_view(), "forward view info can only exist for forward views.");
    TORCH_CHECK(
        !shared_view_info_ || has_bw_view(),
        "forward view info can only exist for forward views.");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return shared_view_info_ ? backward_info_.value() : forward_info_.value();
  }

  DifferentiableViewMeta(
      at::TensorImpl* self_impl,
      std::optional<ViewInfo> backward_info,
      std::optional<ViewInfo> forward_info,
      bool shared_view_info,
      CreationMeta creation_meta = CreationMeta::DEFAULT);
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

/// NOTE: `allow_tensor_metadata_change` is set to true by default, because
/// there are a lot of call sites to these factory functions that need to change
/// the variable's size or storage afterwards, and they don't expect the
/// original tensor (where the variable is created from) to be updated. Setting
/// `allow_tensor_metadata_change_` to false by default would unnecessarily
/// prevent those changes from happening and is undesirable.

// See NOTE [ Autograd View Variables ] for details.
// Differentiable view. Track history with DifferentiableViewMeta.
inline Variable make_variable_differentiable_view(
    const at::Tensor& data,
    std::optional<ViewInfo> backward_info,
    std::optional<ViewInfo> forward_info,
    bool shared_view_info,
    CreationMeta creation_meta,
    bool allow_tensor_metadata_change = true) {
  if (data.defined()) {
    TORCH_CHECK(
        data.getIntrusivePtr()->autograd_meta() == nullptr,
        "Attempted to make a tensor into a differentiable view, but the "
        "tensor already had autograd metadata associated with it.  If you are "
        "using a __torch_dispatch__ mode, the most common cause for this "
        "problem is that you used torch.overrides.enable_reentrant_dispatch() "
        "improperly; tensors created within the extent of reentrant dispatch "
        "MUST NOT be directly returned from __torch_dispatch__; instead, they "
        "must be wrapped into fresh tensors that serve as the output.  If you "
        "are not using wrappers, you probably don't need reentrant dispatch.  "
        "If this doesn't seem applicable, please file a bug to PyTorch.");
    at::TensorImpl* data_impl = data.unsafeGetTensorImpl();
    data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    data_impl->set_autograd_meta(std::make_unique<DifferentiableViewMeta>(
        data_impl,
        std::move(backward_info),
        std::move(forward_info),
        shared_view_info,
        creation_meta));
    return data;
  }
  return Variable();
}

// See NOTE [ Autograd View Variables ] for details.
// Non-differentiable view. Just share version counter.
inline Variable make_variable_non_differentiable_view(
    const Variable& base,
    const at::Tensor& data,
    bool allow_tensor_metadata_change = true,
    bool is_fresh_tensor = false) {
  if (data.defined()) {
    // If we already allocated a new tensor, no need to
    // shallow_copy_and_detach here. (See #163671 history; we tried to
    // fan out to _indices and _values and ran into a SparseTensorImpl
    // can of worms.)
    if (is_fresh_tensor) {
      auto* data_impl = data.unsafeGetTensorImpl();
      data_impl->set_version_counter(impl::version_counter(base));
      data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      data_impl->set_autograd_meta(nullptr);
      return data;
    }
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/impl::version_counter(base),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    data_impl_copy->set_autograd_meta(nullptr);
    return Variable(data_impl_copy);
  }
  return Variable();
}

/// Creates a `Variable` from the given `Tensor`, copying its underlying
/// `TensorImpl`. `requires_grad` should be set only for leaves, and determines
/// whether the `Variable` will accumulate gradients. NOTE: `data` must *not* be
/// a `Variable` already. Its dynamic type *must* be `Tensor`.
///
/// TODO: Eliminate this function as much as possible, as it can be expressed
/// more clearly as detach() or a no-op in most call sites (especially when
/// there is only one use of the variable).
inline Variable make_variable(
    at::Tensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  if (data.defined()) {
    if (data.getIntrusivePtr().use_count() == 1 &&
        data.getIntrusivePtr()->unique_version()) {
      auto data_impl = data.unsafeReleaseIntrusivePtr();
      data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      if (requires_grad) {
        data_impl->set_autograd_meta(
            std::make_unique<AutogradMeta>(data_impl.get(), requires_grad));
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
      return Variable(std::move(data_impl_copy));
    }
  }
  return Variable();
}

/// Creates a `Variable` from the given `Tensor`, copying its underlying
/// `TensorImpl`. `gradient_edge` should be a (function, input_nr) pair
/// specifying the function in the autograd graph, and what particular input of
/// that function, this variable is connected to.
inline Variable make_variable(
    const at::Tensor& data,
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

struct VariableHooks final : at::impl::VariableHooksInterface {
  at::TensorBase tensor_data(
      const at::TensorBase& /*self*/ /*unused*/) const override;
  at::TensorBase variable_data(
      const at::TensorBase& /*self*/ /*unused*/) const override;
  const std::shared_ptr<torch::autograd::Node>& grad_fn(
      const at::TensorBase& /*self*/ /*unused*/) const override;
  unsigned _register_hook(
      const at::TensorBase& /*self*/ /*unused*/,
      std::function<at::TensorBase(const at::TensorBase&)> hook) const override;
  void remove_hook(const at::TensorBase& /*self*/ /*unused*/, unsigned pos)
      const override;
  bool is_view(const at::TensorBase& /*self*/ /*unused*/) const override;
  const at::TensorBase& base(
      const at::TensorBase& /*self*/ /*unused*/) const override;
  const std::string& name(
      const at::TensorBase& /*self*/ /*unused*/) const override;
  bool is_leaf(const at::TensorBase& /*self*/ /*unused*/) const override;
  int64_t output_nr(const at::TensorBase& /*self*/ /*unused*/) const override;
  void set_data(const at::TensorBase& self, const at::TensorBase& new_data)
      const override;
  at::TensorBase data(const at::TensorBase& self) const override;
  int64_t _version(const at::TensorBase& self) const override;
  void retain_grad(const at::TensorBase& self) const override;
  bool retains_grad(const at::TensorBase& self) const override;
  void _backward(
      const at::Tensor& self,
      at::TensorList inputs,
      const std::optional<at::Tensor>& gradient,
      std::optional<bool> keep_graph,
      bool create_graph) const override;
  void requires_grad_(const at::TensorBase& self, bool _requires_grad)
      const override;
  void basic_autograd_not_implemented_fallback(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet dispatch_keys,
      torch::jit::Stack* stack) const override;
  std::optional<c10::ScalarType> grad_dtype(
      const at::TensorBase& /*self*/ /*unused*/) const override;
  void set_grad_dtype(
      const at::TensorBase& /*self*/ /*unused*/,
      const std::optional<c10::ScalarType>& /*grad_dtype*/ /*unused*/)
      const override;
};

namespace utils {

TORCH_API bool has_same_meta(const Variable& base, const Variable& other);

} // namespace utils
} // namespace torch::autograd

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
