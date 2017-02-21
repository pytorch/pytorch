#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/grad_buffer.h"

namespace torch { namespace autograd {

struct Engine {
  using ready_queue_type = std::deque<std::pair<std::shared_ptr<Function>, GradBuffer>>;
  using function_queue = std::vector<Function*>;
  using dependencies_type = std::unordered_map<Function*, int>;

  // Given a list of output variables and their gradients, computes the
  // gradients of "root" variables by backpropagation.
  static void backward(
      const variable_list& variables,
      tensor_list& grad_variables,
      bool retain_variables);

private:
  static dependencies_type compute_dependencies(
      function_queue queue,
      ready_queue_type& ready);
};

}} // namespace torch::autograd
