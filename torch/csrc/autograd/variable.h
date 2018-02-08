#pragma once

#include <Python.h>

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/utils/auto_unique_ptr.h"

#include <ATen/ATen.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch { namespace autograd {

struct Function;

//===----------------------------------------------------------------------===//
//                                Variable
//===----------------------------------------------------------------------===//

/// A `Variable` augments a `Tensor` with the ability to interact in our
/// autograd machinery. `Variable` inherits from `Tensor` and may be converted
/// to and from `Tensor` implicitly.
class Variable : public at::Tensor {
 public:
  /// Creates a Variable that is a *view* of another (*base*) variable.
  /// The `gradient_edge` is an optional (gradient_function, input_number) pair.
  static Variable
  as_view(Variable base, at::Tensor data, Edge gradient_edge = Edge());

  Variable() = default;
  Variable(at::Tensor data, bool requires_grad);
  Variable(at::Tensor data, Edge gradient_edge);

  // "Downcasts" a `Tensor` into a `Variable`. Only call this on tensors you
  // know are Variables.
  /*implicit*/ Variable(at::Tensor const& rhs) : at::Tensor(rhs) {}
  /*implicit*/ Variable(at::Tensor&& rhs) noexcept
      : at::Tensor(std::move(rhs)) {}

  // NOTE: Assignment operators to Tensor come for free from the constructors.

  /// Compare this `Variable` to another `Variable` (or `Tensor`) via
  /// pointer-equality.
  bool is_same(const Variable& other) const noexcept {
    return this->pImpl == other.pImpl;
  }

  void set_name(const std::string& name) noexcept;
  const std::string& name() const noexcept;

  /// Get the gradient function of the `Variable`. If this is a leaf variable,
  /// the pointer returned will be null.
  const std::shared_ptr<Function>& grad_fn() const;

  /// Get the raw gradient function pointer, whatever it currently is.
  Function* grad_fn_ptr() const;

  /// Set the gradient accumulator of the `Variable`. This is only applicable
  /// to leaf variables. Interior variables should call `set_gradient_edge()`.
  void set_grad_accumulator(std::weak_ptr<Function> grad_accumulator);

  /// Attempt to get a pointer to the gradient accumulator of the `Variable`,
  /// if it still exists. If the gradient accumulator function has been
  /// destroyed, returns a `nullptr`.
  std::shared_ptr<Function> try_get_grad_accumulator() const;

  /// Get the gradient accumulator of the `Variable` if it has one, or else
  /// create one on the fly and return it.
  std::shared_ptr<Function> grad_accumulator() const;

  /// Set the gradient edge -- i.e. `grad_fn` and `input_nr` -- of the
  /// `Variable`.
  /// NOTE: This will always set the `grad_fn`, even if this is a leaf
  /// variable, and never the `grad_accumulator`. For the latter, use
  /// `set_grad_accumulator`. This allows late construction of an interior
  /// `Variable`.
  void set_gradient_edge(Edge&& edge) noexcept;

  /// Return the "canonical" gradient edge of this `Variable`, i.e. either the
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

  /// Return the input index of the gradient `Function` to which this `Variable`
  /// is connected.
  int output_nr() const noexcept;

  void set_requires_grad(bool requires_grad) noexcept;
  bool requires_grad() const noexcept;

  PyObject* pyobj() const noexcept;
  void set_pyobj(PyObject* pyobj) noexcept;

  /// Set the type of the underlying `Tensor`.
  void set_type(at::Type*) noexcept;

  const at::Tensor& data() const noexcept;
  at::Tensor& data() noexcept;

  /// Access the gradient `Variable` of this `Variable`.
  const Variable& grad() const noexcept;
  Variable& grad() noexcept;
  void reset_grad() noexcept;

  /// True if this `Variable` is a leaf and thus does not have a `grad_fn`.
  bool is_leaf() const noexcept;

  /// Update the grad_fn of an existing Variable. Called after in-place
  /// modifications.
  void rebase_history(Edge gradient_edge);

  /// Return a copy of this `Variable` that is detached from its autograd graph
  /// and has a blank version. This method is OK to call if the `Variable` is a
  /// view.
  Variable detach() const;

  /// Like `detach()`, but removes this `Variable` in-place. This method may
  /// only be called on non-view `Variable`s. You can use `is_view()` to check
  /// this. If this `Variable` is a view, throws an `std::runtime_error()`.
  void detach_();

  /// Increment the version count of this `Variable`.
  void bump_version() noexcept;
  void set_version(const VariableVersion& version) noexcept;

  /// Return true if this `Variable` is a view of another `Variable`.
  bool is_view() const noexcept;

  /// Return the `Variable` that this `Variable` is a view of. If this
  /// `Variable` is not a view, throw a `std::runtime_error`.
  const Variable& base() const;

  /// Retrieve this `Variable`s version counter.
  const VariableVersion& version_counter() const noexcept;

  /// Retrieve the current value of the `Variable`'s version counter. Equivalent
  /// to calling `version_counter().current_version()`.
  uint32_t current_version() const noexcept;

  void add_hook(std::shared_ptr<FunctionPreHook> hook);
  const std::vector<std::shared_ptr<FunctionPreHook>>& hooks() const noexcept;
  void clear_hooks();

  void set_tracing_state(jit::tracer::ValueTracingState* new_tracing_state);
  jit::tracer::ValueTracingState& tracing_state() const noexcept;
  bool has_tracing_state() const noexcept;

 private:
  /// Private implementation struct of the `Variable`. This struct declaration
  /// and the `get()` method which exposes it shall forever remain private and
  /// never be exposed to the public interface of this class.
  struct Impl;
  struct ViewImpl;
  Variable(Variable::Impl* self, bool retain);
  Impl* get() const noexcept;
};

//===----------------------------------------------------------------------===//
//                            Variable::Impl
//===----------------------------------------------------------------------===//

struct Variable::Impl : public at::TensorImpl {
  explicit Impl(
      at::Tensor data_,
      bool requires_grad_ = false,
      Edge edge = Edge());

  virtual ~Impl();

  const char* toString() const override;
  at::IntList sizes() const override;
  at::IntList strides() const override;
  int64_t dim() const override;
  at::Scalar localScalar() override;
  void* unsafeGetTH(bool retain) override;
  std::unique_ptr<at::Storage> storage() override;
  static const char* typeString();

  std::shared_ptr<Function> get_grad_accumulator();
  virtual std::shared_ptr<Function>& get_grad_fn() {
    return grad_fn;
  }

  std::string name;
  at::Tensor data;

  Variable grad;
  std::shared_ptr<Function> grad_fn;
  std::weak_ptr<Function> grad_accumulator;

  VariableVersion version_counter;
  std::vector<std::shared_ptr<FunctionPreHook>> hooks;

  bool requires_grad; // only meaningful on leaf variables (must be false
                      // otherwise)
  bool is_view;
  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  int output_nr;
  PyObject* pyobj; // weak reference

  // Mutex to ensure that concurrent read operations that modify internal
  // state are still thread-safe. Used by get_grad_fn and
  // get_grad_accumulator.
  std::mutex mutex;

  // For use in torch::jit::tracer
  auto_unique_ptr<jit::tracer::ValueTracingState> tracing_state;
};

//===----------------------------------------------------------------------===//
//                          Variable::ViewImpl
//===----------------------------------------------------------------------===//

// A Variable that is a view on another Variable. The base and view share the
// same version_counter. The grad_fn field of the Variable may become stale
// due to in-place modifications of the shared data. Accesses should go
// through get_grad_fn(). All other fields are always valid.
struct Variable::ViewImpl : public Variable::Impl {
  ViewImpl(Variable base_, at::Tensor data_, Edge gradient_edge);

  // Gets the up-to-date grad_fn. If the shared data or base was modified, we
  // re-create the grad_fn to express the up-to-date view relationship between
  // this and the base Variable.
  virtual std::shared_ptr<Function>& get_grad_fn() override;

  // Called after in-place modifications. Modifies the grad_fn of the base
  // Variable.
  void rebase_history(Edge gradient_edge);

  // The base Variable (never a view)
  Variable base;

  // The value of the version_counter at the time grad_fn was created. The
  // grad_fn field is stale if attr_version !=
  // version_counter.current_version()
  uint32_t attr_version;
};

//===----------------------------------------------------------------------===//
//                        Variable Implementation
//===----------------------------------------------------------------------===//

inline Variable::Variable(Variable::Impl* self, bool retain)
    : at::Tensor(self, retain) {}

inline const std::shared_ptr<Function>& Variable::grad_fn() const {
  return get()->get_grad_fn();
}

inline Function* Variable::grad_fn_ptr() const {
  return get()->grad_fn.get();
}

inline void Variable::set_grad_accumulator(
    std::weak_ptr<Function> grad_accumulator) {
  get()->grad_accumulator = std::move(grad_accumulator);
}

inline std::shared_ptr<Function> Variable::try_get_grad_accumulator() const {
  return get()->grad_accumulator.lock();
}

inline std::shared_ptr<Function> Variable::grad_accumulator() const {
  return get()->get_grad_accumulator();
}

inline void Variable::set_gradient_edge(Edge&& edge) noexcept {
  get()->grad_fn = std::move(edge.function);
  get()->output_nr = edge.input_nr;
}

inline int Variable::output_nr() const noexcept {
  return get()->output_nr;
}

inline void Variable::set_requires_grad(bool requires_grad) noexcept {
  get()->requires_grad = requires_grad;
}

inline bool Variable::requires_grad() const noexcept {
  return get()->requires_grad || get()->grad_fn ||
      (is_view() && base().requires_grad());
}

inline void Variable::set_pyobj(PyObject* pyobj) noexcept {
  get()->pyobj = pyobj;
}

inline PyObject* Variable::pyobj() const noexcept {
  return get()->pyobj;
}

inline void Variable::set_type(at::Type* new_type) noexcept {
  pImpl->setType(new_type);
}

inline void Variable::reset_grad() noexcept {
  get()->grad.reset();
}

inline const at::Tensor& Variable::data() const noexcept {
  return get()->data;
}

inline at::Tensor& Variable::data() noexcept {
  return get()->data;
}

inline const Variable& Variable::grad() const noexcept {
  return get()->grad;
}

inline Variable& Variable::grad() noexcept {
  return get()->grad;
}

inline bool Variable::is_leaf() const noexcept {
  return get()->grad_fn == nullptr;
}

inline void Variable::add_hook(std::shared_ptr<FunctionPreHook> hook) {
  get()->hooks.push_back(std::move(hook));
}

inline const std::vector<std::shared_ptr<FunctionPreHook>>& Variable::hooks()
    const noexcept {
  return get()->hooks;
}

inline void Variable::clear_hooks() {
  get()->hooks.clear();
}

inline void Variable::set_tracing_state(
    jit::tracer::ValueTracingState* new_tracing_state) {
  get()->tracing_state.reset(new_tracing_state);
}

inline jit::tracer::ValueTracingState& Variable::tracing_state() const
    noexcept {
  return *get()->tracing_state;
}

inline bool Variable::has_tracing_state() const noexcept {
  return get()->tracing_state != nullptr;
}

inline void Variable::set_version(const VariableVersion& version) noexcept {
  get()->version_counter = version;
}

inline void Variable::bump_version() noexcept {
  get()->version_counter.bump();
}

inline uint32_t Variable::current_version() const noexcept {
  return get()->version_counter.current_version();
}

inline const VariableVersion& Variable::version_counter() const noexcept {
  return get()->version_counter;
}

inline bool Variable::is_view() const noexcept {
  return get()->is_view;
}

inline const Variable& Variable::base() const {
  if (is_view()) {
    return static_cast<Variable::ViewImpl*>(get())->base;
  }
  throw std::runtime_error("Can't get base of non-view");
}

inline void Variable::set_name(const std::string& name) noexcept {
  get()->name = name;
}

inline const std::string& Variable::name() const noexcept {
  return get()->name;
}

inline Variable::Impl* Variable::get() const noexcept {
  return static_cast<Variable::Impl*>(pImpl);
}
}} // namespace torch::autograd
