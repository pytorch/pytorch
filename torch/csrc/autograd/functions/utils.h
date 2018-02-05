#pragma once

#include <Python.h>
#include <functional>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

using function_constructor = std::function<std::shared_ptr<Function>(function_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr);

/**
 * Checks that inputs contains exactly `args` items and that the first `required_args`
 * items are not NULL. If not specified, `required_args` defaults to `args`.
 */
void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args=-1);
}}
