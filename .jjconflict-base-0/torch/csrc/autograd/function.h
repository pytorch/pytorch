#pragma once

#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/node.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <c10/util/intrusive_ptr.h>

namespace torch::autograd {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                       Associated Free Functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
// Implementation of `collect_next_edges` (see below).
struct MakeNextFunctionList : IterArgs<MakeNextFunctionList> {
  edge_list next_edges;
  using IterArgs<MakeNextFunctionList>::operator();
  void operator()(const Variable& variable) {
    if (variable.defined()) {
      next_edges.emplace_back(impl::gradient_edge(variable));
    } else {
      next_edges.emplace_back();
    }
  }
  void operator()(const Variable* variable) {
    operator()(*variable);
  }
  void operator()(const std::optional<Variable>& variable) {
    if (variable.has_value()) {
      operator()(*variable);
    } else {
      next_edges.emplace_back();
    }
  }
};
} // namespace detail

/// Create an `Edge` between the given `variable` and the `function`, which is
/// assumed to be the gradient function of this variable (i.e. the function
/// through which this variable is backpropagated during the backward pass).
/// This sets the `grad_fn` property of the `variable`. This function assumes
/// that the `Variable` is a new input to the gradient function and its
/// `input_nr` thus equal to `function->num_inputs()`. Additionally, it
/// increments the `Node`'s number of inputs by one. Approximately
/// equivalent to `variable.set_gradient_edge(function,
/// function->add_input_metadata(variable.dispatch_type(), variable.sizes()))`.
/// If you don't want the `Node`'s `num_inputs` to be incremented, use
/// `set_gradient_edge` directly.
inline void create_gradient_edge(
    Variable& variable,
    c10::intrusive_ptr<Node> function) {
  // Copy before move.
  const auto input_nr = function->add_input_metadata(variable);
  impl::set_gradient_edge(variable, {std::move(function), input_nr});
}

/// Return true if any of the variables in the list require a gradient.
inline bool any_variable_requires_grad(const variable_list& variables) {
  return std::any_of(
      variables.begin(), variables.end(), [](const Variable& variable) {
        return variable.defined() && variable.requires_grad();
      });
}

/// Return the next edges of all the given variables, or tuples of variables.
template <typename... Variables>
edge_list collect_next_edges(Variables&&... variables) {
  detail::MakeNextFunctionList make;
  make.apply(std::forward<Variables>(variables)...);
  return std::move(make.next_edges);
}

struct TypeAndSize {
  TypeAndSize() = default;
  /* implicit */
  TypeAndSize(const at::Tensor& t)
      : sym_sizes(t.sym_sizes().vec()), options(t.options()) {}

  at::Tensor zeros();

  std::vector<c10::SymInt> sym_sizes;
  at::TensorOptions options;
};

} // namespace torch::autograd
