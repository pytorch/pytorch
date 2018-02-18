#pragma once

// Function is an abstract class that represents a single operation from one or
// more variables to one more or variables.
//
// Subclasses may represent "forward" or "backward" operations (i.e functions
// and their derivatives). Some functions may be used as both.

#include "torch/csrc/assertions.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/utils/auto_unique_ptr.h"
#include "torch/csrc/utils/python_stub.h"
#include "torch/csrc/utils/variadic.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

struct Function;
struct Variable;
struct Edge;

using tensor_list = std::vector<at::Tensor>;
using variable_list = std::vector<Variable>;
using function_list = std::vector<Edge>;
using saved_variable_list = std::vector<SavedVariable>;
using IndexRange = std::pair<size_t, size_t>;

namespace detail {
struct MakeNextFunctionList : IterArgs<MakeNextFunctionList> {
  function_list next_functions;
  using IterArgs<MakeNextFunctionList>::operator();
  void operator()(const Variable& variable) {
    if (variable.defined()) {
      next_functions.push_back(variable.gradient_edge());
    } else {
      next_functions.emplace_back();
    }
  }
};
} // namespace detail

// Returns true if any of the variables in the list require a gradient.
inline bool any_variable_requires_grad(const variable_list& variables) {
  return std::any_of(
      variables.begin(), variables.end(), [](const Variable& variable) {
        return variable.requires_grad();
      });
}

template <typename... Variables>
function_list get_next_functions(Variables&&... variables) {
  if (!GradMode::is_enabled()) return {};
  detail::MakeNextFunctionList make;
  make.apply(std::forward<Variables>(variables)...);
  return std::move(make.next_functions);
}

struct Function : std::enable_shared_from_this<Function> {
  static thread_local uint64_t function_counter;

  Function() : time(function_counter++) {}
  Function(function_list&& next_functions_) : Function() {
    next_functions = std::move(next_functions_);
  }

  Function(const Function& other) = delete;
  Function(Function&& other) = delete;
  virtual ~Function() {}

  // Implements the operation
  // NOTE: Don't call this function directly. Use operator() instead.
  virtual variable_list apply(const variable_list& inputs) = 0;
  variable_list tracedApply(variable_list inputs);

  variable_list operator()(const variable_list& inputs) {
    profiler::RecordFunction rec(this);
    if (jit::tracer::isTracingVar(inputs)) {
      return tracedApply(inputs);
    }
    return apply(inputs);
  }

  // PyFunctions are not managed by shared_ptrs by default, but are bound to the
  // lifetime of their Python object instead.
  virtual std::shared_ptr<Function> getSharedPtr() {
    return shared_from_this();
  };

  // Releases saved variables if the operation won't be reused
  virtual inline void releaseVariables() {}
  // called before a an apply if will release variables is going to be called
  // allows larger ops like InterpreterAutogradFunction
  // to incrementally release variables as they run
  virtual inline void willReleaseVariables() {}
  // Function name for debugging
  virtual std::string name();

  bool should_compute_output(size_t index) const {
    TORCH_ASSERTM(index < next_functions.size(), "Index out of range");
    return next_functions[index].is_valid();
  }

  bool should_compute_any_outputs() const {
    for (size_t i = 0; i < next_functions.size(); ++i) {
      if (should_compute_output(i)) {
        return true;
      }
    }
    return false;
  }

  bool should_compute_output(std::initializer_list<size_t> idxs) const {
    return std::any_of(idxs.begin(), idxs.end(), [this](size_t i) {
      return should_compute_output(i);
    });
  }

  bool should_compute_output(std::initializer_list<IndexRange> idxs) const {
    return std::any_of(idxs.begin(), idxs.end(), [this](IndexRange range) {
      for (size_t i = range.first; i < range.second; i++) {
        if (should_compute_output(i)) return true;
      }
      return false;
    });
  }

  void set_next_functions(function_list&& next_functions) {
    this->next_functions = std::move(next_functions);
  }

  // An op is traceable if all operations happening within apply() are performed
  // on autograd Variables (i.e. apply mostly instantiates and applies other functions).
  virtual inline bool is_traceable() { return false; };

  // An op is said to pass state transparently to backward, if the state consists
  // only of (Saved)Variables and only non-variable objects that parametrize the
  // operation in some way that defines the graph structure AND the backward function
  // is traceable. In particular, parametrization MUST NOT depend on the data
  // of any Variable.
  // TODO: it might be possible to handle cases where backward is non-traceable
  // but state passing could be considered transparent. This will probably depend
  // on saved_variable_list being mutable.
  // NOTE: this value matters only if is_traceable() returns false.
  virtual inline bool passes_state_transparently() { return false; };

  // Let's the JIT find inputs to apply that are not present explicitly in arguments.
  // Required only for functions that are not traceable, don't pass state to
  // backward transparently, and are not backwards closures of functions that don't
  // pass the state transparently. Which means that hopefully they will hardly ever
  // need to be implemented :)
  virtual inline std::unique_ptr<saved_variable_list> saved_variables() { return nullptr; }

  static void setUpContextEdge(jit::Node* this_node,
                               const variable_list& inputs, const variable_list& outputs);

  int num_inputs = 0;
  uint64_t time;
  function_list next_functions;
  std::vector<std::shared_ptr<FunctionPreHook>> pre_hooks;
  std::vector<std::shared_ptr<FunctionPostHook>> post_hooks;

  PyObject* pyobj = nullptr; // weak reference

  auto_unique_ptr<jit::tracer::FunctionTracingState> tracing_state;
};

// See Function::is_traceable() for definition.
struct TraceableFunction : public Function {
  using Function::Function;

  virtual inline bool is_traceable() final { return true; };
};
}} // namespace torch::autograd
