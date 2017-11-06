#pragma once

// A wrapper around at::Tensor to represent autograd Variables. Variables
// can be implicitly converted to an at::Tensor.

#include <mutex>
#include <memory>
#include <vector>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/utils/auto_unique_ptr.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

using at::Tensor;
struct VariableImpl;

// TODO: fix name conflict with jit VariableFlags
struct VarFlags {
  constexpr VarFlags(bool requires_grad, bool is_volatile)
    : requires_grad(requires_grad), is_volatile(is_volatile) {}
  VarFlags() : VarFlags(false, false) {}
  bool requires_grad;
  bool is_volatile;
};

constexpr VarFlags DEFAULT_FLAGS = {false, false};

struct Variable : public at::Tensor {
  inline Variable(VariableImpl * self, bool retain);
  Variable() : Tensor() {}
  Variable(const Variable & rhs) : Tensor(rhs) {}
  Variable(Variable && rhs) noexcept : Tensor(std::move(rhs)) {}

  // Implicitly casts a Tensor to a Variable. This should only be called on
  // Tensors which you know are actually Variables.
  /*implicit*/ Variable(Tensor const & rhs) : Tensor(rhs) {}
  /*implicit*/ Variable(Tensor && rhs) noexcept : Tensor(std::move(rhs)) {}

  inline VariableImpl* get() const;

  inline const Tensor & data() const;
  inline       Tensor & data();

  inline Tensor opt_data() const;

  inline const Variable & grad() const;
  inline       Variable & grad();

  inline bool is_leaf() const;

  inline const std::shared_ptr<Function>& grad_fn() const;

  // Updates the flags and grad_fn of an existing Variable. Called after in-place modifications.
  inline void rebase_history(VarFlags flags, int output_nr, std::shared_ptr<Function> grad_fn);

  std::shared_ptr<Function> grad_accumulator() const;

  inline const std::vector<std::shared_ptr<FunctionPreHook>>& hooks() const;
  inline       std::vector<std::shared_ptr<FunctionPreHook>>& hooks();

  inline auto_unique_ptr<jit::tracer::ValueTracingState>& tracing_state() const;

  inline int current_version() const;

  inline VariableVersion& version_counter() const;

  inline const int& output_nr() const;
  inline       int& output_nr();

  inline const bool& requires_grad() const;
  inline       bool& requires_grad();

  inline const bool& is_volatile() const;
  inline       bool& is_volatile();

  inline bool is_view() const;
  inline Variable& base() const;

  inline Variable & operator=(Variable && rhs) &;
  inline Variable & operator=(const Variable & rhs) &;
  inline Variable & operator=(Tensor && rhs) &;
  inline Variable & operator=(const Tensor & rhs) &;
};

struct VariableImpl : public at::TensorImpl {
public:
  VariableImpl(at::Tensor data, VarFlags flags=DEFAULT_FLAGS, int output_nr=0,
               std::shared_ptr<Function> grad_fn=nullptr);
  virtual ~VariableImpl();
  virtual const char * toString() const override;
  virtual at::IntList sizes() const override;
  virtual at::IntList strides() const override;
  virtual int64_t dim() const override;
  virtual at::Scalar localScalar() override;
  virtual void assign_(at::Scalar s) override;
  virtual void * unsafeGetTH(bool retain) override;
  static const char * typeString();

  // Get the VariableType for a base Tensor type
  static at::Type* getType(const at::Type& baseType);
  static at::Type* getType(const at::Tensor& tensor);

public:
  std::shared_ptr<Function> get_grad_accumulator();
  virtual std::shared_ptr<Function>& get_grad_fn() { return _grad_fn; }
  virtual void rebase_grad_fn(std::shared_ptr<Function> grad_fn) {
    _grad_fn = std::move(grad_fn);
  }

  at::Tensor data;
  Variable grad;
  std::shared_ptr<Function> _grad_fn;
  VariableVersion version_counter;
  std::vector<std::shared_ptr<FunctionPreHook>> hooks;
  std::weak_ptr<Function> grad_accumulator;
  // Mutex to ensure that concurrent read operations that modify internal state
  // are still thread-safe. Used by get_grad_fn and get_grad_accumulator.
  std::mutex mutex;
  bool requires_grad;
  bool is_volatile;
  bool is_view;
  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  int output_nr;
  PyObject *pyobj;  // weak reference

  // For use in torch::jit::tracer
  auto_unique_ptr<jit::tracer::ValueTracingState> tracing_state;
  friend struct VariableType;
};

// A Variable that is a view on another Variable. The base and view share the
// same version_counter. The _grad_fn field of the Variable may become stale
// due to in-place modifications of the shared data. Accesses should go through
// get_grad_fn(). All other fields are always valid.
struct VariableViewImpl : public VariableImpl {
  VariableViewImpl(Variable base, at::Tensor data, VarFlags flags, int output_nr,
                   std::shared_ptr<Function> grad_fn);

  // Gets the up-to-date grad_fn. If the shared data or base was modified, we
  // re-create the grad_fn to express the up-to-date view relationship between
  // this and the base Variable.
  virtual std::shared_ptr<Function>& get_grad_fn() override;

  // Called after in-place modifications. Modifies the grad_fn of the base
  // Variable.
  virtual void rebase_grad_fn(std::shared_ptr<Function> grad_fn) override;

  // The base Variable (never a view)
  Variable base;

  // The value of the version_counter at the time grad_fn was created. The
  // _grad_fn field is stale if attr_version != version_counter.current_version()
  int attr_version;
};

inline Variable make_variable(at::Tensor data, VarFlags flags=DEFAULT_FLAGS,
                              int output_nr=0, std::shared_ptr<Function> grad_fn=nullptr) {
  return Variable(new VariableImpl(std::move(data), flags, output_nr, std::move(grad_fn)), false);
}

Variable make_variable(at::Tensor data, std::shared_ptr<Function> grad_fn);

inline Variable make_variable(at::Tensor data, bool requires_grad, bool is_volatile=false) {
  return make_variable(std::move(data), VarFlags(requires_grad, is_volatile));
}

inline Variable make_variable_view(Variable base, at::Tensor data, VarFlags flags=DEFAULT_FLAGS,
                                   int output_nr=0, std::shared_ptr<Function> grad_fn=nullptr) {
  return Variable(new VariableViewImpl(std::move(base), std::move(data), flags, output_nr, std::move(grad_fn)), false);
}


inline Variable::Variable(VariableImpl * self, bool retain) : Tensor(self, retain) {
}

inline VariableImpl* Variable::get() const {
  return static_cast<VariableImpl*>(pImpl);
}

inline const Tensor & Variable::data() const {
  return get()->data;
}
inline Tensor & Variable::data() {
  return get()->data;
}

inline Tensor Variable::opt_data() const {
  if (!defined()) {
    return Tensor();
  }
  return data();
}

inline const Variable & Variable::grad() const {
  return get()->grad;
}
inline Variable & Variable::grad() {
  return get()->grad;
}

inline bool Variable::is_leaf() const {
  return get()->_grad_fn == nullptr;
}

inline const std::shared_ptr<Function>& Variable::grad_fn() const {
  return get()->get_grad_fn();
};
inline void Variable::rebase_history(VarFlags flags, int output_nr, std::shared_ptr<Function> grad_fn) {
  get()->requires_grad = flags.requires_grad;
  get()->is_volatile = flags.is_volatile;
  get()->output_nr = output_nr;
  get()->rebase_grad_fn(std::move(grad_fn));
}
inline std::shared_ptr<Function> Variable::grad_accumulator() const {
  return get()->get_grad_accumulator();
};

inline const std::vector<std::shared_ptr<FunctionPreHook>>& Variable::hooks() const {
  return get()->hooks;
};
inline std::vector<std::shared_ptr<FunctionPreHook>>& Variable::hooks() {
  return get()->hooks;
};

inline auto_unique_ptr<jit::tracer::ValueTracingState>& Variable::tracing_state() const {
  return get()->tracing_state;
};

inline int Variable::current_version() const {
  return get()->version_counter.current_version();
}

inline VariableVersion& Variable::version_counter() const {
  return get()->version_counter;
}

inline const int& Variable::output_nr() const {
  return get()->output_nr;
}

inline int& Variable::output_nr() {
  return get()->output_nr;
}

inline const bool& Variable::requires_grad() const {
  return get()->requires_grad;
}
inline bool& Variable::requires_grad() {
  return get()->requires_grad;
}

inline const bool& Variable::is_volatile() const {
  return get()->is_volatile;
}
inline bool& Variable::is_volatile() {
  return get()->is_volatile;
}

inline bool Variable::is_view()const {
  return get()->is_view;
}
inline Variable& Variable::base() const {
  if (is_view()) {
    return static_cast<VariableViewImpl&>(*get()).base;
  }
  throw std::runtime_error("Can't get base of non-view");
}

inline Variable & Variable::operator=(Variable && rhs) & {
  rhs.swap(*this);
  return *this;
}
inline Variable & Variable::operator=(const Variable & rhs) & {
  Variable(rhs).swap(*this);
  return *this;
}
inline Variable & Variable::operator=(Tensor && rhs) & {
  rhs.swap(*this);
  return *this;
}
inline Variable & Variable::operator=(const Tensor & rhs) & {
  Variable(rhs).swap(*this);
  return *this;
}

}} // namespace torch::autograd
