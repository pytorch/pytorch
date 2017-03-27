#pragma once

// Function is an abstract class that represents a single operation from one or
// more variables to one more or varaibles.
//
// Subclasses may represent "forward" or "backward" operations (i.e functions
// and their derivatives). Some functions may be used as both.

#include <memory>
#include <THPP/THPP.h>
#include <vector>

#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/function_hook.h"

namespace torch { namespace autograd {

struct Function;
struct Variable;

using tensor_list = std::vector<std::unique_ptr<thpp::Tensor>>;
using variable_list = std::vector<std::shared_ptr<Variable>>;
using function_list = std::vector<std::pair<std::shared_ptr<Function>, int>>;

// State used to create "backward" functions
struct FunctionFlags {
  bool requires_grad = false;
  bool is_volatile = false;
  function_list previous_functions;
};

struct Function {
  Function()
    : num_outputs(0)
    , previous_functions()
    , requires_grad(false)
    , is_volatile(false)
    , is_stochastic(false)
    , pre_hooks()
    , post_hooks()
    {}

  Function(FunctionFlags&& flags)
    : num_outputs(0)
    , previous_functions(std::move(flags.previous_functions))
    , requires_grad(flags.requires_grad)
    , is_volatile(flags.is_volatile)
    , is_stochastic(false)
    , pre_hooks()
    , post_hooks()
    {}

  Function(const Function& other) = delete;
  Function(Function&& other) = delete;
  virtual ~Function() {}

  // Implements the operation
  virtual variable_list apply(const variable_list& inputs) = 0;

  // Computes requires_grad, is_volatile, and previous_functions from a list
  // of input variables
  static FunctionFlags flags(const variable_list& inputs);

  // Releases saved variables if the operation won't be reused
  virtual inline void releaseVariables() {}

  // Function name for debugging
  virtual std::string name();

  inline bool needs_input_grad(int i) const {
    auto& fn = previous_functions[i].first;
    return fn && fn->requires_grad;
  }

  // These variables are usually only meaningful for "backward" functions.
  // num_outputs is the number of outputs of corresponding "forward" function;
  // it's actually the number of inputs of this function.
  int num_outputs;
  function_list previous_functions;
  bool requires_grad;
  bool is_volatile;
  bool is_stochastic;
  std::vector<std::shared_ptr<FunctionPreHook>> pre_hooks;
  std::vector<std::shared_ptr<FunctionPostHook>> post_hooks;
};


}} // namespace torch::autograd
