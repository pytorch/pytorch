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
  if (var.requires_grad() && var.is_leaf() && GradMode::is_enabled()) {
    AT_ERROR(
      "a leaf Variable that requires grad has been used in an in-place operation.");
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
    var.rebase_history({std::move(grad_fn), 0});
  }
}

inline void rebase_history(std::vector<Variable>&& vars, std::shared_ptr<Node> grad_fn) {
  if (grad_fn) {
    for (auto& var : vars) {
      if (var.defined()) {
        // TODO: eliminate const_cast
        auto output_nr = grad_fn->add_input_metadata(var);
        var.rebase_history({std::move(grad_fn), output_nr});
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
      }
    }
  }
}

inline void increment_version(Tensor & t) {
  as_variable_ref(t).bump_version();
}

inline bool isFloatingPoint(ScalarType s) {
  return s == kFloat || s == kDouble || s == kHalf;
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
inline Tensor as_view(const Tensor & base, Tensor tensor, bool is_differentiable = true) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  return make_variable_view(std::move(base_var), std::move(tensor), is_differentiable);
}

// See NOTE [ Autograd View Variables ] for details.
inline std::vector<Tensor> as_view(const Tensor & base, std::vector<Tensor> tensors,
                                   bool is_differentiable = true) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  for(Tensor &tensor : tensors) {
    tensor = make_variable_view(base_var, std::move(tensor), is_differentiable);
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

// NOTE: For now, there is no guarantee that the tensors returned from
// out-of-place ATen ops are not Variables. For example, the following operators:
//
// 1. `coalesce()` (called from `VariableType::coalesce()`)
// 2. `_embedding_bag_cpu()` (called from `VariableType::_embedding_bag()`)
//
// can return its input or tensors created using the input's options, which can
// potentially be Variables because inputs to ATen ops can be Variables.
//
// In the near future, once we make every tensor a Variable, these two
// `as_variable()` functions are no-op and we can remove them.
inline Tensor as_variable(Tensor tensor) {
  return tensor.is_variable() ? tensor : make_variable(std::move(tensor), /*requires_grad=*/false);
}

inline std::vector<Tensor> as_variable(TensorList tl) {
  return fmap(tl, [](const Tensor& t) -> Tensor {
      return t.is_variable() ? t : make_variable(t, /*requires_grad=*/false);
  });
}

template <typename... Tensors, size_t... Is>
std::tuple<Tensors...> as_variable_impl(
    std::tuple<Tensors...> tensors,
    Indices<Is...>) {
  // Expand the integer parameter pack into a sequence of Variable
  // constructions. This turns into (boolean omitted):
  // Variable(std::get<0>(tensors)), Variable(std::get<1>(tensors)), ...
  return std::tuple<Tensors...>(
      as_variable(std::get<Is>(tensors))...);
}

// NB: Because this was not forward declared, recursive std::tuple won't work.
// You can probably rejigger this to make it supported if you really need it.
template <typename... Tensors>
std::tuple<Tensors...> as_variable(std::tuple<Tensors...> tensors) {
  // `sizeof...(Tensors)` gets us the size of the `Tensors` parameter pack at
  // compile time. We use it to parameterize a `MakeIndices` class, which will
  // expand into an Indices object containing the numbers 0 to
  // sizeof...(Tensors) - 1.
  return as_variable_impl(
      tensors, typename MakeIndices<sizeof...(Tensors)>::indices());
}

inline std::vector<std::vector<int64_t>> to_args_sizes(TensorList tensors) {
  std::vector<std::vector<int64_t>> args_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    args_sizes[i] = tensors[i].sizes().vec();
  }
  return args_sizes;
}

}} // namespace torch::autograd
