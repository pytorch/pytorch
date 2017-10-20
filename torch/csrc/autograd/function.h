#pragma once

// Function is an abstract class that represents a single operation from one or
// more variables to one more or variables.
//
// Subclasses may represent "forward" or "backward" operations (i.e functions
// and their derivatives). Some functions may be used as both.

#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/utils/auto_unique_ptr.h"
#include "torch/csrc/utils/python_stub.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/tracer.h"

#include <ATen/ATen.h>

#include <memory>
#include <vector>

namespace torch { namespace autograd {

struct Function;
struct Variable;

using tensor_list = std::vector<at::Tensor>;
using variable_list = std::vector<Variable>;
using edge_type = std::pair<std::shared_ptr<Function>, int>;
using function_list = std::vector<edge_type>;
using saved_variable_list = std::vector<SavedVariable>;

struct edge_hasher {
  std::size_t operator()(const edge_type& edge) const {
#define HASH_IDX(idx) std::hash<std::tuple_element<idx, edge_type>::type>()(std::get<idx>(edge))
    // TODO: that's probably a bad hash function, but whatever
    return HASH_IDX(0) ^ HASH_IDX(1);
  }
};

// State used to create "backward" functions
struct FunctionFlags {
  // Roughly speaking, is_executable corresponds to requires_grad.
  // See http://pytorch.org/docs/notes/autograd.html for more details:
  // both is_executable and is_volatile specify whether or not backwards
  // gradient computation will be performed for a function, but they differ in
  // their precedence.
  bool is_executable = false;
  bool is_volatile = false;
  // What functions take the output of this function as input.
  // There is one function per output of this function.
  function_list next_functions;
};

struct Function : std::enable_shared_from_this<Function> {
  Function()
    : num_inputs(0)
    , next_functions()
    , is_executable(false)
    , is_stochastic(false)
    , pre_hooks()
    , post_hooks()
    , pyobj(nullptr)
    {}

  Function(FunctionFlags&& flags)
    : num_inputs(0)
    , next_functions(std::move(flags.next_functions))
    , is_executable(flags.is_executable)
    , is_stochastic(false)
    , pre_hooks()
    , post_hooks()
    , pyobj(nullptr)
    {}

  Function(const Function& other) = delete;
  Function(Function&& other) = delete;
  virtual ~Function() {}

  // Implements the operation
  // NOTE: Don't call this function directly. Use apply_fn or operator() instead.
  virtual variable_list apply(const variable_list& inputs) = 0;
  variable_list tracedApply(variable_list inputs);

  variable_list operator()(const variable_list& inputs) {
    profiler::RecordFunction rec(this);
    if (jit::tracer::isTracing(inputs)) {
      return tracedApply(inputs);
    }
    return apply(inputs);
  }

  // PyFunctions are not managed by shared_ptrs by default, but are bound to the
  // lifetime of their Python object instead.
  virtual std::shared_ptr<Function> getSharedPtr() {
    return shared_from_this();
  };

  // Computes is_executable, is_volatile, and next_functions from a list
  // of input variables
  static FunctionFlags flags(const variable_list& inputs);
  static FunctionFlags flags(const std::initializer_list<Variable>& inputs);
  static FunctionFlags flags(at::TensorList inputs);

  // Releases saved variables if the operation won't be reused
  virtual inline void releaseVariables() {}

  // Function name for debugging
  virtual std::string name();

  inline bool should_compute_output(int i) const {
    auto& fn = next_functions[i].first;
    return fn && fn->is_executable;
  }

  inline bool should_compute_any_outputs() const {
    for (size_t i = 0; i < next_functions.size(); ++i) {
      if (should_compute_output((int)i)) {
        return true;
      }
    }
    return false;
  }

  inline bool should_compute_output(std::initializer_list<int> idxs) const {
    return std::any_of(idxs.begin(), idxs.end(), [this](int i) {
      return should_compute_output(i);
    });
  }

  inline void set_flags(FunctionFlags&& flags) {
    is_executable = flags.is_executable;
    next_functions = std::move(flags.next_functions);
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

  static void setUpContextEdge(jit::Node* this_node, int ctx_output_nr,
                               const variable_list& inputs, const variable_list& outputs);

  int num_inputs;
  function_list next_functions;
  bool is_executable;
  bool is_stochastic;
  std::vector<std::shared_ptr<FunctionPreHook>> pre_hooks;
  std::vector<std::shared_ptr<FunctionPostHook>> post_hooks;

  PyObject *pyobj;  // weak reference

  auto_unique_ptr<jit::tracer::FunctionTracingState> tracing_state;
};

// Actually what is a ForwardFunction here applies to all functions that are
// applied only in forward OR are backward closures that don't save any Variables.
// I chose this name, because the second situation is quite rare.
template<bool transparent_state = false>
struct ForwardFunction : public Function {
  using Function::Function;

  virtual inline std::unique_ptr<saved_variable_list> saved_variables() final {
    return std::unique_ptr<saved_variable_list>(new saved_variable_list());
  }

  virtual inline bool is_traceable() final { return false; };

  virtual inline bool passes_state_transparently() final { return transparent_state; };
};

// See Function::is_traceable() for definition.
struct TraceableFunction : public Function {
  using Function::Function;

  virtual inline bool is_traceable() final { return true; };
};

template<typename T>
struct apply_fn {
  template<typename... Args>
  apply_fn(Args&& ...args)
    : fn_(std::make_shared<T>(std::forward<Args>(args)...)) {}

  Variable operator()(const variable_list& inputs) {
    return (*fn_)(inputs)[0];
  }

  template<typename... Args>
  Variable operator()(Args&& ...inputs) {
    return (*fn_)(variable_list{inputs...})[0];
  }

  std::shared_ptr<T> fn_;
};

}} // namespace torch::autograd
