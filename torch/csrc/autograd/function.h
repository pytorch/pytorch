#pragma once

// Function is an abstract class that represents a single operation from one or
// more variables to one more or variables.
//
// Subclasses may represent "forward" or "backward" operations (i.e functions
// and their derivatives). Some functions may be used as both.

#include <Python.h>
#include "torch/csrc/autograd/function_hook.h"

#include <ATen/ATen.h>

#include <memory>
#include <vector>

namespace torch { namespace autograd {

struct Function;
struct Variable;

using tensor_list = std::vector<at::Tensor>;
using variable_list = std::vector<std::shared_ptr<Variable>>;
using function_list = std::vector<std::pair<std::shared_ptr<Function>, int>>;

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

struct Function {
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
  virtual variable_list apply(const variable_list& inputs) = 0;

  // Computes is_executable, is_volatile, and next_functions from a list
  // of input variables
  static FunctionFlags flags(const variable_list& inputs);

  // Releases saved variables if the operation won't be reused
  virtual inline void releaseVariables() {}

  // Function name for debugging
  virtual std::string name();

  inline bool should_compute_output(int i) const {
    auto& fn = next_functions[i].first;
    return fn && fn->is_executable;
  }

  inline void set_flags(FunctionFlags&& flags) {
    is_executable = flags.is_executable;
    next_functions = std::move(flags.next_functions);
  }

  int num_inputs;
  function_list next_functions;
  bool is_executable;
  bool is_stochastic;
  std::vector<std::shared_ptr<FunctionPreHook>> pre_hooks;
  std::vector<std::shared_ptr<FunctionPostHook>> post_hooks;

  PyObject *pyobj;  // weak reference
};


}} // namespace torch::autograd
