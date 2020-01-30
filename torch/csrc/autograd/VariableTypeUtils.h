#pragma once

#include <torch/csrc/autograd/generated/VariableType.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/ir.h>

#include <torch/csrc/utils/variadic.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

inline void check_inplace(const Tensor& tensor) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.requires_grad() && GradMode::is_enabled()) {
    if (var.is_view()) {
      // NB: is_view() ==> get_autograd_meta()
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(impl::get_autograd_meta(var));
      // Unsafe because we don't want to trigger the creation of the view's custom grad_fn
      auto grad_fn = impl::grad_fn_unsafe(var);
      if (grad_fn) {
        TORCH_CHECK(diff_view_meta->allow_rebase_history,
            "The ", diff_view_meta->output_nr_, "th output of ", grad_fn->name(),
            " is being modified inplace but this is not allowed as it would prevent correct gradient computation.");
      } else {
        TORCH_CHECK(diff_view_meta->allow_rebase_history,
            "A view created in no_grad mode is being modified inplace but this is not allowed as the expected"
            " behavior is unclear. You should have both the view and inplace inside the no_grad block if you"
            " do NOT want the change to be tracked by the autograd. Or both outside the no_grad block if you"
            " want the change to tracked.");

      }
    }
    if (var.is_leaf()) {
      AT_ERROR(
        "a leaf Variable that requires grad is being used in an in-place operation.");
    }
  }
}

inline void throw_error_out_requires_grad(const char* name) {
  AT_ERROR(
      name, "(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.");
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
        auto output_nr = grad_fn->add_input_metadata(var);
        impl::rebase_history(var, {std::move(grad_fn), output_nr});
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
      }
    }
  }
}

inline void increment_version(Tensor & t) {
  impl::bump_version(as_variable_ref(t));
}

struct Flatten : IterArgs<Flatten> {
  Flatten(variable_list& out) : out(out) {}
  variable_list& out;
  void operator()(const at::Tensor& x) { out.emplace_back(x); }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out.insert(out.end(), xs.begin(), xs.end());
  }
};

template<typename... Args> inline variable_list flatten_tensor_args(Args&&... args) {
  variable_list out;
  out.reserve(count_tensors(std::forward<Args>(args)...));
  Flatten(out).apply(std::forward<Args>(args)...);
  return out; // RVO
}

// See NOTE [ Autograd View Variables ] for details.
inline Tensor as_differentiable_view(const Tensor & base, Tensor tensor, bool allow_rebase_history) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  return make_variable_differentiable_view(std::move(base_var), std::move(tensor), allow_rebase_history);
}

// See NOTE [ Autograd View Variables ] for details.
inline std::vector<Tensor> as_differentiable_view(const Tensor & base, std::vector<Tensor> tensors,
                                                  bool allow_rebase_history) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  for(Tensor &tensor : tensors) {
    tensor = make_variable_differentiable_view(base_var, std::move(tensor), allow_rebase_history);
  }
  return tensors;
}

// See NOTE [ Autograd View Variables ] for details.
inline Tensor as_non_differentiable_view(const Tensor & base, Tensor tensor) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  return make_variable_non_differentiable_view(std::move(base_var), std::move(tensor));
}

// See NOTE [ Autograd View Variables ] for details.
inline std::vector<Tensor> as_non_differentiable_view(const Tensor & base, std::vector<Tensor> tensors) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  for(Tensor &tensor : tensors) {
    tensor = make_variable_non_differentiable_view(base_var, std::move(tensor));
  }
  return tensors;
}

inline void check_no_requires_grad(const Tensor& tensor, const char* name) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.defined() && var.requires_grad()) {
    std::string msg = "the derivative for '";
    msg += name;
    msg += "' is not implemented";
    throw std::runtime_error(msg);
  }
}

inline void check_no_requires_grad(TensorList tensors, const char* name) {
  for (auto& tensor : tensors) {
    check_no_requires_grad(tensor, name);
  }
}

// Assumed that saved tensor lists are never inplace outputs
inline std::vector<SavedVariable> make_saved_variable_list(TensorList tensors) {
  return fmap(tensors, [](const Tensor& tensor) -> SavedVariable {
      return SavedVariable{tensor, false /* is output */}; });
}

inline std::vector<std::vector<int64_t>> to_args_sizes(TensorList tensors) {
  std::vector<std::vector<int64_t>> args_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    args_sizes[i] = tensors[i].sizes().vec();
  }
  return args_sizes;
}

}} // namespace torch::autograd
