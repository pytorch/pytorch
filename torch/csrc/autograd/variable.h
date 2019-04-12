#pragma once

#include <torch/csrc/utils/python_stub.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function_hook.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

struct Function;

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///                                Variable
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// A `Variable` augments a `Tensor` with the ability to interact in our
/// autograd machinery. Conceptually, `Variable`s travel along `Edge`s between
/// `Function`s in the autograd graph. A `Variable` can either be a leaf, like a
/// weight in a neural network, or an interior variable, when it is the result
/// of an operation between variables. Every `Variable` also stores another
/// `Variable` called its `grad` (gradient). If the variable is a leaf, its
/// gradient will be accumulated into this variable.
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
///
///
///                               Interface
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// `Variable` inherits from `Tensor` and thus its API is a superset of that of
/// `Tensor`. This means you can perform all the usual mathematical and other
/// operations you can perform on `Tensor`s also on `Variable`s. Furthermore,
/// `Variable` and `Tensor` actually convert implicitly between each other. You
/// can thus call functions defined on `Tensor`s also with `Variable`s. For
/// this, the `Variable` class allows implicit construction from `Tensor`. It is
/// the responsibility of calling code to ensure that this constructor is
/// invoked only when the `Tensor`'s dynamic type is actually `Variable`. Most
/// notably, it is *not* correct to construct a brand new `Variable` from a
/// `Tensor` using this constructor. To do so, you must use the `make_variable`
/// free function instead. To create a view variable, use `make_variable_view`.
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct TORCH_API Variable : public at::Tensor {
  /// Default constructor.
  Variable() = default;

  // Factory Functions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // NOTE: These factory functions have to be friends to access the
  // `Variable::Impl`. As a side effect, it allows us to keep them in the class.

  /// Creates a `Variable` that is a *view* of another (*base*) variable.
  /// The `gradient_edge` is an optional (gradient_function, input_number) pair.
  /// `is_differentiable` is a bool that specifies whether this view is
  /// differentiable, i.e., whether the relation should be tracked by autograd.
  /// See NOTE [ Autograd View Variables ] for details.
  friend Variable make_variable_view(
      Variable base,
      at::Tensor data,
      bool is_differentiable,
      bool allow_tensor_metadata_change,
      Edge gradient_edge);

  /// Creates a `Variable` from the given `Tensor`, copying its underlying `TensorImpl`.
  /// `requires_grad` should be
  /// set only for leaves, and determines whether the `Variable` will accumulate
  /// gradients. NOTE: `data` must *not* be a `Variable` already. Its dynamic
  /// type *must* be `Tensor`.
  friend Variable make_variable(
      at::Tensor data,
      bool requires_grad,
      bool allow_tensor_metadata_change);

  /// Creates a `Variable` from the given `Tensor`, consuming its underlying `TensorImpl`.
  /// This is intended to be used from functions that immediately create a `Tensor`,
  /// convert it to a `Variable`, and then free it; it has been found to
  /// decrease the overhead of those operations, in some situations.
  /// The comments about `requires_grad` and `data` on the above version also apply to this one.
  friend Variable make_variable_consuming(
      at::Tensor data,
      bool requires_grad,
      bool allow_tensor_metadata_change);

  /// Creates a `Variable` from the given `Tensor` and specify a
  /// `gradient_edge`, i.e. a (function, input_nr) pair specifying the function
  /// in the autograd graph, and what particular input of that function, this
  /// variable is connected to.
  friend Variable make_variable(
      at::Tensor data,
      Edge gradient_edge,
      bool allow_tensor_metadata_change);

  // Tensor Conversions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // "Downcasts" a `Tensor` into a `Variable`. Only call this on tensors you
  // know are Variables.
  /*implicit*/ Variable(at::Tensor const& rhs) : at::Tensor(rhs) {
    AT_CHECK(
        is_variable() || !defined(),
        "Tensor that was converted to Variable was not actually a Variable");
  }

  /*implicit*/ Variable(at::Tensor&& rhs)
      : at::Tensor(std::move(rhs)) {
    AT_CHECK(
        is_variable() || !defined(),
        "Tensor that was converted to Variable was not actually a Variable");
  }

  // NOTE: Assignment operators to Tensor come for free from the constructors.

  const at::Tensor& data() const noexcept;
  at::Tensor& data() noexcept;

  // Gradient Function and Edges
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Gets the gradient function of the `Variable`. If this is a leaf variable,
  /// the pointer returned will be null.
  ///
  /// For View Variables:
  /// Gets the up-to-date grad_fn. If the shared data or base was modified, we
  /// re-create the grad_fn to express the up-to-date view relationship between
  /// this and the base Variable.
  const std::shared_ptr<Function>& grad_fn() const;

  /// Gets the raw gradient function pointer, whatever it currently is.
  Function* grad_fn_unsafe() const;

  /// Set the gradient accumulator of the `Variable`. This is only applicable to
  /// leaf variables. Interior variables should call `set_gradient_edge()`.
  void set_grad_accumulator(std::weak_ptr<Function> grad_accumulator);

  /// Attempts to get a pointer to the gradient accumulator of the `Variable`,
  /// if it still exists. If the gradient accumulator function has been
  /// destroyed, returns a `nullptr`.
  std::shared_ptr<Function> try_get_grad_accumulator() const;

  /// Gets the gradient accumulator of the `Variable` if it has one, or else
  /// create one on the fly and return it.
  std::shared_ptr<Function> grad_accumulator() const;

  /// Returns the "canonical" gradient edge of this `Variable`, i.e. either the
  /// gradient function if this is an interior `Variable`, or the gradient
  /// accumulator otherwise. If the `Variable` is interior, the returned `Edge`
  /// will store the input index of the `Function` to which this variable is
  /// connected in its `input_nr` field. For leaves, the `input_nr` is always
  /// zero. Note that `set_gradient_edge` and `gradient_edge` are not
  /// symmetric. You must use `set_gradient_edge` to set the `grad_fn` and
  /// `set_grad_accumulator` to set the accumulator.
  Edge gradient_edge() const {
    // If grad_fn is null (as is the case for a leaf node), we instead
    // interpret the gradient function to be a gradient accumulator, which will
    // accumulate its inputs into the grad property of the variable. These
    // nodes get suppressed in some situations, see "suppress gradient
    // accumulation" below. Note that only variables which have `requires_grad =
    // True` can have gradient accumulators.
    if (const auto& gradient = grad_fn()) {
      return Edge(gradient, output_nr());
    } else {
      return Edge(grad_accumulator(), 0);
    }
  }

  /// Returns a copy of this `Variable` that is detached from its autograd graph
  /// and has a blank version. This method is OK to call if the `Variable` is a
  /// view.
  /// NOTE: Previously, if we change the tensor metadata (e.g. sizes / strides /
  /// storage / storage_offset) of a tensor created from `detach()`, those metadata
  /// in the original tensor will also be updated. However, the new behavior is that
  /// those metadata changes to the detached tensor will not update the original tensor
  /// anymore, and in the `detach()` function we need to set `allow_tensor_metadata_change_`
  /// to false to make such changes explicitly illegal, in order to prevent users from
  /// changing metadata of the detached tensor and expecting the original tensor to also
  /// be updated.
  Variable detach() const;

  /// Like `detach()`, but removes this `Variable` in-place. This method may
  /// only be called on non-view `Variable`s. You can use `is_view()` to check
  /// this. If this `Variable` is a view, throws an `std::runtime_error()`.
  void detach_();

  /// Computes the gradient of current tensor w.r.t. graph leaves.
  void backward(
      c10::optional<Tensor> gradient,
      bool keep_graph,
      bool create_graph) const;

  /// Sets the `Tensor` held by this `Variable` to the one supplied.
  /// It is rarely necessary to call this; it's used, for example, when
  /// a non-sparse gradient gets added to a sparse gradient, requiring
  /// the type of the gradient `Variable` to become non-sparse.
  void set_data(const at::Tensor &new_data);

  /// Set the gradient edge -- i.e. `grad_fn` and `input_nr` -- of the
  /// `Variable`.
  /// NOTE: This will always set the `grad_fn`, even if this is a leaf variable,
  /// and never the `grad_accumulator`. For the latter, use
  /// `set_grad_accumulator`. This allows late construction of an interior
  /// `Variable`.
  void set_gradient_edge(Edge edge) noexcept;

  /// Returns the input index of the gradient `Function` to which this
  /// `Variable` is connected.  Note: input indexes of the gradient `Function`
  /// correspond to output indexes of the corresponding forward `Function`.
  uint32_t output_nr() const noexcept;

  /// True if this `Variable` is a leaf and thus does not have a `grad_fn`.
  bool is_leaf() const noexcept;

  // Versions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Increments the version count of this `Variable`.
  void bump_version() noexcept;
  void set_version_counter(const c10::VariableVersion& version_counter) noexcept;

  /// Retrieves this `Variable`s version counter.
  const c10::VariableVersion& version_counter() const noexcept;

  /// Retrieves the current value of the `Variable`'s version counter.
  /// Equivalent to calling `version_counter().current_version()`.
  uint32_t current_version() const noexcept;

  // Autograd Graph Interaction
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Update the `grad_fn` of an existing Variable. Called after in-place
  /// modifications.
  ///
  /// For View Variables:
  /// Called after in-place modifications. Modifies the grad_fn of the base
  /// Variable.
  void rebase_history(Edge gradient_edge);

  // Hooks
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void add_hook(std::shared_ptr<FunctionPreHook> hook);
  const std::vector<std::shared_ptr<FunctionPreHook>>& hooks() const noexcept;
  void clear_hooks();

  // View Variables
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Returns true if this `Variable` is a view of another `Variable`.
  bool is_view() const noexcept;

  /// Returns the `Variable` that this `Variable` is a view of. If this
  /// `Variable` is not a view, throw a `std::runtime_error`.
  const Variable& base() const;

  // Miscellaneous
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void set_name(const std::string& name);
  const std::string& name() const noexcept;

  PyObject* pyobj() const noexcept;
  void set_pyobj(PyObject* pyobj) noexcept;

  struct AutogradMeta;
  Variable::AutogradMeta* get_autograd_meta() const noexcept;

 private:
  /// Private implementation struct of the `Variable`. This struct declaration
  /// and the `get()` method which exposes it shall forever remain private and
  /// never be exposed to the public interface of this class.
  struct Impl;
  struct DifferentiableViewImpl;
  struct DifferentiableViewMeta;

  // Private Methods
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Variable(c10::intrusive_ptr<Variable::Impl> self);
  Impl* get() const;
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Variable::AutogradMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Each `Variable` has one unique `AutogradMeta` struct, which stores autograd
/// metadata fields that are necessary for tracking the Variable's autograd history.

struct TORCH_API Variable::AutogradMeta : public c10::AutogradMetaInterface {
  std::string name;

  Variable grad_;
  std::shared_ptr<Function> grad_fn_;
  std::weak_ptr<Function> grad_accumulator_;

  std::vector<std::shared_ptr<FunctionPreHook>> hooks_;

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
  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) override {
    AT_CHECK(
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
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        Variable::DifferentiableViewMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct TORCH_API Variable::DifferentiableViewMeta : public Variable::AutogradMeta {
  /// The base `Variable` (never a view).
  Variable base_;

  /// The value of the version_counter at the time grad_fn was created. The
  /// grad_fn field is stale if attr_version !=
  /// version_counter.current_version().
  uint32_t attr_version;

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_ || (is_view_ && base_.requires_grad());
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Variable::Impl
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct TORCH_API Variable::Impl : public at::TensorImpl {
  explicit Impl(
      at::Tensor data,
      std::unique_ptr<Variable::AutogradMeta> autograd_meta,
      bool requires_grad = false,
      Edge gradient_edge = Edge());

  ~Impl() override;

  int64_t numel() const override;
  at::IntArrayRef sizes() const override;
  at::IntArrayRef strides() const override;
  bool is_contiguous() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  void resize_dim(int64_t ndim) override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;

  int64_t dim() const override;
  bool has_storage() const override;
  const at::Storage& storage() const override;
  void* slow_data() const override;

  void set_data(const at::Tensor &new_data);

  /// Reset all expensive fields to free up resources
  void release_resources() override;

  Variable::AutogradMeta* get_autograd_meta() const {
    return static_cast<Variable::AutogradMeta*>(autograd_meta());
  }

  int64_t storage_offset() const override;

  /// The underlying data tensor for this Variable.
  /// This field will be removed once VariableImpl and TensorImpl are merged.
  at::Tensor data_;
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     Variable::DifferentiableViewImpl
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
/// Variable::DifferentiableViewImpl is created to support gradient tracking of
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
/// view relation via DifferentiableViewImpl in autograd, the views will be
/// using usual Variable::Impl and just share the version counters with the base
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
struct TORCH_API Variable::DifferentiableViewImpl : public Variable::Impl {
  DifferentiableViewImpl(
    Variable base,
    at::Tensor data,
    Edge gradient_edge,
    std::unique_ptr<Variable::DifferentiableViewMeta> autograd_meta);

  /// Reset all expensive fields to free up resources
  void release_resources() override;
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        Variable Implementation
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Factory Functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// NOTE: `allow_tensor_metadata_change` is set to true by default, because there
/// are a lot of call sites to these factory functions that need to change the
/// variable's size or storage afterwards, and they don't expect the original
/// tensor (where the variable is created from) to be updated. Setting
/// `allow_tensor_metadata_change_` to false by default would unnecessarily
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
      /// Differentiable view. Track history with DifferentiableViewImpl.
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach();
      data_impl_copy->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      auto data_copy = at::Tensor(data_impl_copy);
      auto diff_view_meta = c10::guts::make_unique<Variable::DifferentiableViewMeta>();
      return Variable(c10::make_intrusive<Variable::DifferentiableViewImpl>(
              std::move(base), std::move(data_copy), std::move(gradient_edge), std::move(diff_view_meta)));
    } else {
      /// Non-differentiable view. Just share version counter.
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach();
      data_impl_copy->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      auto data_copy = at::Tensor(data_impl_copy);
      auto autograd_meta = c10::guts::make_unique<Variable::AutogradMeta>();
      auto var = Variable(c10::make_intrusive<Variable::Impl>(
              std::move(data_copy), std::move(autograd_meta), false, std::move(gradient_edge)));
      var.set_version_counter(base.version_counter());
      return var;
    }
  }
  return Variable();
}

inline Variable make_variable(
    at::Tensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  AT_CHECK(
      !data.is_variable(),
      "Must not create a new variable from a variable, use its .data()");
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach();
    data_impl_copy->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    auto data_copy = at::Tensor(data_impl_copy);
    auto autograd_meta = c10::guts::make_unique<Variable::AutogradMeta>();
    return Variable(c10::make_intrusive<Variable::Impl>(data_copy, std::move(autograd_meta), requires_grad));
  }
  return Variable();
}

inline Variable make_variable_consuming(
    at::Tensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  AT_CHECK(
      !data.is_variable(),
      "Must not create a new variable from a variable, use its .data()");
  if (data.defined()) {
    AT_ASSERT(data.getIntrusivePtr().use_count() == 1);
    data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    auto autograd_meta = c10::guts::make_unique<Variable::AutogradMeta>();
    return Variable(c10::make_intrusive<Variable::Impl>(std::move(data), std::move(autograd_meta), requires_grad));
  }
  return Variable();
}

inline Variable make_variable(
    at::Tensor data,
    Edge gradient_edge,
    bool allow_tensor_metadata_change = true) {
  AT_CHECK(
      !data.is_variable(),
      "Must not create a new variable from a variable, use its .data()");
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach();
    data_impl_copy->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    auto data_copy = at::Tensor(data_impl_copy);
    auto autograd_meta = c10::guts::make_unique<Variable::AutogradMeta>();
    return Variable(c10::make_intrusive<Variable::Impl>(data_copy, std::move(autograd_meta), false, std::move(gradient_edge)));
  }
  return Variable();
}

// Tensor Conversion
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Downcasts the `Tensor` reference to a `Variable` reference. If compiling
/// in DEBUG mode and the tensor's dynamic type is not in fact `Variable`,
/// throws a `std::invalid_argument` exception.
inline Variable& as_variable_ref(at::Tensor& tensor) {
  AT_CHECK(
      tensor.is_variable(),
      "Attempted to cast a Tensor to a Variable, but "
      "the dynamic type of the value is not Variable.");
  return static_cast<Variable&>(tensor);
}

inline const Variable& as_variable_ref(const at::Tensor& tensor) {
  AT_CHECK(
      tensor.is_variable(),
      "Attempted to cast a Tensor to a Variable, but "
      "the dynamic type of the value is not Variable.");
  return static_cast<const Variable&>(tensor);
}

inline const at::Tensor& Variable::data() const noexcept {
  return get()->data_;
}

inline at::Tensor& Variable::data() noexcept {
  return get()->data_;
}

// Gradient Function and Edges
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline Function* Variable::grad_fn_unsafe() const {
  return get_autograd_meta()->grad_fn_.get();
}

inline void Variable::set_grad_accumulator(
    std::weak_ptr<Function> grad_accumulator) {
  get_autograd_meta()->grad_accumulator_ = std::move(grad_accumulator);
}

inline std::shared_ptr<Function> Variable::try_get_grad_accumulator() const {
  return get_autograd_meta()->grad_accumulator_.lock();
}

inline Variable Variable::detach() const {
  auto var = make_variable_view(*this, get()->data_, /*is_differentiable=*/false, /*allow_tensor_metadata_change=*/false, Edge());
  return var;
}

inline void Variable::set_data(const at::Tensor &new_data) {
  get()->set_data(new_data);
}

inline void Variable::set_gradient_edge(Edge edge) noexcept {
  get_autograd_meta()->grad_fn_ = std::move(edge.function);
  get_autograd_meta()->output_nr_ = edge.input_nr;
}

inline uint32_t Variable::output_nr() const noexcept {
  return get_autograd_meta()->output_nr_;
}

inline bool Variable::is_leaf() const noexcept {
  return get_autograd_meta()->grad_fn_ == nullptr;
}

// Versions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void Variable::set_version_counter(
    const c10::VariableVersion& version_counter) noexcept {
  data().unsafeGetTensorImpl()->set_version_counter(version_counter);
}

inline void Variable::bump_version() noexcept {
  data().unsafeGetTensorImpl()->bump_version();
}

inline uint32_t Variable::current_version() const noexcept {
  return data().unsafeGetTensorImpl()->version_counter().current_version();
}

inline const c10::VariableVersion& Variable::version_counter() const noexcept {
  return data().unsafeGetTensorImpl()->version_counter();
}

// Hooks
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void Variable::add_hook(std::shared_ptr<FunctionPreHook> hook) {
  get_autograd_meta()->hooks_.push_back(std::move(hook));
}

inline const std::vector<std::shared_ptr<FunctionPreHook>>& Variable::hooks()
    const noexcept {
  return get_autograd_meta()->hooks_;
}

inline void Variable::clear_hooks() {
  get_autograd_meta()->hooks_.clear();
}

// View Variables
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline bool Variable::is_view() const noexcept {
  return get_autograd_meta()->is_view_;
}

inline const Variable& Variable::base() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    return diff_view_meta->base_;
  } else {
    throw std::runtime_error("Can't get base of non-view Variable");
  }
}

// Miscellaneous
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void Variable::set_name(const std::string& name) {
  get_autograd_meta()->name = name;
}

inline const std::string& Variable::name() const noexcept {
  return get_autograd_meta()->name;
}

inline void Variable::set_pyobj(PyObject* pyobj) noexcept {
  get()->set_pyobj(pyobj);
}

inline PyObject* Variable::pyobj() const noexcept {
  return get()->pyobj();
}

inline Variable::AutogradMeta* Variable::get_autograd_meta() const noexcept {
  return get()->get_autograd_meta();
}

// Private Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline Variable::Variable(c10::intrusive_ptr<Variable::Impl> self)
    : at::Tensor(std::move(self)) {}

inline Variable::Impl* Variable::get() const {
  AT_CHECK(defined(), "Called Variable::get() on an undefined Variable");
  return static_cast<Variable::Impl*>(impl_.get());
}
}} // namespace torch::autograd
