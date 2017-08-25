#pragma once

// A wrapper around at::Tensor to represent autograd Variables. Variables
// can be implicitly converted to an at::Tensor.

#include <Python.h>
#include <mutex>
#include <memory>
#include <vector>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/utils/auto_unique_ptr.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

using at::Tensor;
struct VariableTensor;

struct Variable : public at::Tensor {
  inline Variable(VariableTensor * self, bool retain);
  Variable() : Tensor() {}
  Variable(const Variable & rhs) : Tensor(rhs) {}
  Variable(Variable && rhs) noexcept : Tensor(std::move(rhs)) {}

  explicit Variable(Tensor const & rhs) : Tensor(rhs) {}
  explicit Variable(Tensor && rhs) noexcept : Tensor(std::move(rhs)) {}

  inline VariableTensor* get() const;

  inline const Tensor & data() const;
  inline       Tensor & data();

  inline Tensor opt_data() const;

  inline const Variable & grad() const;
  inline       Variable & grad();

  inline const std::shared_ptr<Function>& grad_fn() const;
  inline       std::shared_ptr<Function>& grad_fn();

  std::shared_ptr<Function> grad_accumulator() const;

  inline const std::vector<std::shared_ptr<FunctionPreHook>>& hooks() const;
  inline       std::vector<std::shared_ptr<FunctionPreHook>>& hooks();

  inline auto_unique_ptr<jit::tracer::ValueTracingState>& tracing_state() const;

  inline int current_version() const;
  inline int output_nr() const;

  inline bool requires_grad() const;
  inline void set_requires_grad(bool requires_grad);

  inline bool is_volatile() const;
  inline void set_volatile(bool is_volatile);

  inline Variable & operator=(Variable && rhs) &;
  inline Variable & operator=(const Variable & rhs) &;

  // implicit conversion to Tensor
  operator Tensor() const { return Tensor(pImpl, true); }
};

struct VariableTensor : public at::TensorImpl {
public:
  explicit VariableTensor(at::Tensor data);
  VariableTensor(at::Tensor data, std::shared_ptr<Function> grad_fn);
  VariableTensor(at::Tensor data, bool requires_grad, bool is_volatile=false);
  virtual ~VariableTensor();
  virtual const char * toString() const override;
  virtual at::IntList sizes() override;
  virtual at::IntList strides() override;
  virtual int64_t dim() override;
  virtual at::Scalar localScalar() override;
  virtual void assign_(at::Scalar s) override;
  virtual void * unsafeGetTH(bool retain) override;
  static const char * typeString();

  // Get the VariableType for a base Tensor type
  static at::Type* getType(const at::Type& baseType);
  static at::Type* getType(const at::Tensor& tensor);

public:
  std::shared_ptr<Function> get_grad_accumulator();

  at::Tensor data;
  Variable grad;
  std::shared_ptr<Function> grad_fn;
  std::unique_ptr<VariableVersion> version_counter;
  std::vector<std::shared_ptr<FunctionPreHook>> hooks;
  std::weak_ptr<Function> grad_accumulator;
  std::mutex grad_accumulator_lock;
  bool requires_grad;
  bool is_volatile;
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

inline Variable::Variable(VariableTensor * self, bool retain) : Tensor(self, retain) {
}

inline VariableTensor* Variable::get() const {
  return static_cast<VariableTensor*>(pImpl);
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

inline const std::shared_ptr<Function>& Variable::grad_fn() const {
  return get()->grad_fn;
};
inline std::shared_ptr<Function>& Variable::grad_fn() {
  return get()->grad_fn;
};
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
  return **get()->version_counter;
}

inline int Variable::output_nr() const {
  return get()->output_nr;
}

inline bool Variable::requires_grad() const {
  return get()->requires_grad;
}
inline void Variable::set_requires_grad(bool requires_grad) {
  get()->requires_grad = requires_grad;
}

inline bool Variable::is_volatile() const {
  return get()->is_volatile;
}
inline void Variable::set_volatile(bool is_volatile) {
  get()->is_volatile = is_volatile;
}

inline Variable & Variable::operator=(Variable && rhs) & {
  rhs.swap(*this);
  return *this;
}
inline Variable & Variable::operator=(const Variable & rhs) & {
  Variable(rhs).swap(*this);
  return *this;
}

}} // namespace torch::autograd
