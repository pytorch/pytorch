#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/ATen.h>

#include <functional>
#include <memory>
#include <vector>

namespace torch { namespace autograd {

using function_constructor = std::function<std::shared_ptr<Node>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                                     const function_constructor& ctr);

///  Checks that inputs contains exactly `args` items and that the first `required_args`
/// items are not nullptr. If not specified, `required_args` defaults to `args`.
TORCH_API void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args=-1, bool allow_undefined=false);

// The requires_grad argument is used to know if the inplace operation needs
// gradient to be setup for it.
// In particular, we can have tensor.requires_grad() != requires_grad when writing
// a Tensor that requires gradients inplace into a Tensor that does not require gradients:
// a = torch.rand(2)
// b = torch.rand(2, requires_grad=True)
// a.copy_(b)
inline void check_inplace(const Tensor& tensor, bool requires_grad) {
  if (requires_grad && GradMode::is_enabled()) {
    auto diff_view_meta = impl::get_view_autograd_meta(tensor);
    if (diff_view_meta && diff_view_meta->has_bw_view()) {
      // This can throw or warn
      handle_view_on_rebase(diff_view_meta);
      if (tensor.requires_grad() && tensor._base().is_leaf()) {
          AT_ERROR(
            "a view of a leaf Variable that requires grad is being used in an in-place operation.");
      }
    }
    if (tensor.requires_grad() && tensor.is_leaf()) {
      AT_ERROR(
        "a leaf Variable that requires grad is being used in an in-place operation.");
    }
  }
}

inline void check_inplace(const TensorList tensors, bool requires_grad) {
  for (const auto& tensor : tensors) {
    check_inplace(tensor, requires_grad);
  }
}

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  void operator()(const c10::optional<at::Tensor>& tensor) {
    if (tensor.has_value()) {
      (*this)(*tensor);
    }
  }
  bool short_circuit() {
    return out;
  }
};

template <typename... Args>
inline bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

inline void set_history(
    at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  AT_ASSERT(grad_fn);
  if (variable.defined()) {
    // If the codegen triggers this, you most likely want to add your newly added function
    // to the DONT_REQUIRE_DERIVATIVE list in tools/autograd/gen_variable_type.py
    TORCH_INTERNAL_ASSERT(isDifferentiableType(variable.scalar_type()));
    auto output_nr =
        grad_fn->add_input_metadata(variable);
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

inline void set_history(
    std::vector<Variable>&& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

inline void set_history(
    std::vector<Variable>& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

// TODO: Blegh, bare references
inline void rebase_history(Variable& var, std::shared_ptr<Node> grad_fn) {
  if (grad_fn && var.defined()) {
    grad_fn->add_input_metadata(var);
    impl::rebase_history(var, {std::move(grad_fn), 0});
  }
}

inline void rebase_history(std::vector<Variable>&& vars, std::shared_ptr<Node> grad_fn) {
  if (grad_fn) {
    for (auto& var : vars) {
      if (var.defined()) {
        // TODO: eliminate const_cast
        // NOLINTNEXTLINE(bugprone-use-after-move)
        auto output_nr = grad_fn->add_input_metadata(var);
        impl::rebase_history(var, {std::move(grad_fn), output_nr});
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
      }
    }
  }
}

inline bool isFwGradDefined(const c10::optional<Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

}}
