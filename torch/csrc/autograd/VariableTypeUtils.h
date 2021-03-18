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
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>

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

// The requires_grad argument is used to know if the inplace operation needs
// gradient to be setup for it.
// In particular, we can have tensor.requires_grad() != requires_grad when writing
// a Tensor that requires gradients inplace into a Tensor that does not require gradients:
// a = torch.rand(2)
// b = torch.rand(2, requires_grad=True)
// a.copy_(b)
inline void check_inplace(const Tensor& tensor, bool requires_grad) {
  if (requires_grad && GradMode::is_enabled()) {
    if (tensor.is_view()) {
      // NB: is_view() ==> get_autograd_meta()
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(impl::get_autograd_meta(tensor));
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

inline void throw_error_out_requires_grad(const char* name) {
  AT_ERROR(
      name, "(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.");
}

inline void throw_error_for_complex_autograd(const Tensor& tensor, const char* name) {
  if (tensor.requires_grad()) {
    TORCH_CHECK(!tensor.is_complex(), name,
                " does not support automatic differentiation for outputs with complex dtype.");
  }
}

inline void throw_error_for_complex_autograd(const TensorList& tensorlist, const char* name) {
  for (auto tensor: tensorlist) {
    throw_error_for_complex_autograd(tensor, name);
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
        auto output_nr = grad_fn->add_input_metadata(var);
        impl::rebase_history(var, {std::move(grad_fn), output_nr});
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
      }
    }
  }
}

inline void increment_version(Tensor & t) {
  impl::bump_version(t);
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
inline Tensor as_view(const Tensor & base, const Tensor & tensor, bool is_bw_differentiable,
        bool is_fw_differentiable, std::function<Tensor(const Tensor&)> view_func=nullptr,
        CreationMeta creation_meta=CreationMeta::DEFAULT, bool allow_tensor_metadata_change=true) {
  if (!isForwardADEnabled()) {
    // Fast codepath for backward only code
    // It is useful as it avoids the creation of the temporary c10<optional> which makes
    // a significant difference when measuring instruction count for a single "t.view(-1)" call from c++.
    if (is_bw_differentiable) {
      if (base.is_view()) {
        auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(base));
        const auto& base_bw_info = diff_view_meta->get_backward_view();
        creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
        return make_variable_differentiable_view(tensor, base_bw_info.chain(base, tensor, view_func),
                                                 c10::nullopt, creation_meta, allow_tensor_metadata_change);
      } else {
        return make_variable_differentiable_view(tensor, ViewInfo(base, view_func),
                                                 c10::nullopt, creation_meta, allow_tensor_metadata_change);
      }
    } else {
      TORCH_CHECK(creation_meta == CreationMeta::DEFAULT,
                  "Non-backward differentiable views must have creation_meta=CreationMeta::DEFAULT");
      return make_variable_non_differentiable_view(base, std::move(tensor), allow_tensor_metadata_change);
    }
  }
  // Create both the forward and backward info that are needed
  c10::optional<ViewInfo> new_bw_info;
  c10::optional<ViewInfo> new_fw_info;

  if (is_bw_differentiable) {
    if (base.is_view()) {
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(base));
      const auto& base_bw_info = diff_view_meta->get_backward_view();
      new_bw_info = base_bw_info.chain(base, tensor, view_func);
    } else {
      new_bw_info = ViewInfo(base, view_func);
    }
  } else {
    TORCH_CHECK(creation_meta == CreationMeta::DEFAULT,
                "Non-backward differentiable views must have creation_meta=CreationMeta::DEFAULT");
  }

  if (is_fw_differentiable) {
    // Check if base is a forward differentiable view
    auto base_meta = torch::autograd::impl::get_autograd_meta(base);
    auto is_view = base_meta && base_meta->is_view_;
    if (is_view && static_cast<DifferentiableViewMeta*>(base_meta)->has_fw_view()) {
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(base_meta);
      const auto& base_fw_info = diff_view_meta->get_forward_view();
      new_fw_info = base_fw_info.chain(base, tensor, view_func);
    } else {
      new_fw_info = ViewInfo(base, view_func);
    }
  }

  if (is_fw_differentiable || is_bw_differentiable) {
    if (base.is_view()) {
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(base));
      creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
    }
    return make_variable_differentiable_view(tensor, std::move(new_bw_info), std::move(new_fw_info),
                                             creation_meta, allow_tensor_metadata_change);
  } else {
    return make_variable_non_differentiable_view(base, tensor, allow_tensor_metadata_change);
  }
}

// See NOTE [ Autograd View Variables ] for details.
inline std::vector<Tensor> as_view(const Tensor & base, std::vector<Tensor>& tensors, bool is_bw_differentiable,
                                   bool is_fw_differentiable, CreationMeta creation_meta=CreationMeta::DEFAULT) {
  c10::optional<ViewInfo> new_bw_info = c10::nullopt;
  c10::optional<ViewInfo> new_fw_info = c10::nullopt;

  if (is_bw_differentiable) {
    if (base.is_view()) {
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(base));
      const auto& base_bw_info = diff_view_meta->get_backward_view();
      TORCH_INTERNAL_ASSERT(creation_meta == CreationMeta::MULTI_OUTPUT_NODE || creation_meta == CreationMeta::MULTI_OUTPUT_SAFE,
                            "Functions that result multiple view must have a creation meta reflecting this behavior.");
      // It is ok to create a ViewInfo where only the base is correct in this case as inplace operations on such views are
      // not allowed
      new_bw_info = ViewInfo(base_bw_info.base_, /* view_func */ nullptr);
    } else {
      new_bw_info = ViewInfo(base, /* view_func */ nullptr);
    }
  } else {
    TORCH_CHECK(creation_meta == CreationMeta::DEFAULT,
                "Non-backward differentiable views must have creation_meta=CreationMeta::DEFAULT");
  }
  if (isForwardADEnabled() && is_fw_differentiable) {
    // Check if base is a forward differentiabble view
    auto base_meta = torch::autograd::impl::get_autograd_meta(base);
    auto is_view = base_meta && base_meta->is_view_;
    if (is_view && static_cast<DifferentiableViewMeta*>(base_meta)->has_fw_view()) {
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(base_meta);
      const auto& base_fw_info = diff_view_meta->get_forward_view();
      TORCH_INTERNAL_ASSERT(creation_meta == CreationMeta::MULTI_OUTPUT_NODE || creation_meta == CreationMeta::MULTI_OUTPUT_SAFE,
                            "Functions that result multiple view must have a creation meta reflecting this behavior.");
      // It is ok to create a ViewInfo where only the base is correct in this case as inplace operations on such views are
      // not allowed
      new_fw_info = ViewInfo(base_fw_info.base_, /* view_func */ nullptr);
    } else {
      new_fw_info = ViewInfo(base, /* view_func */ nullptr);
    }
  }

  if ((is_fw_differentiable || is_bw_differentiable) && base.is_view()) {
    auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(base));
    creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
  }

  for(Tensor &tensor : tensors) {
    if (is_fw_differentiable || is_bw_differentiable) {
      tensor = make_variable_differentiable_view(tensor, new_bw_info, new_fw_info, creation_meta);
    } else {
      tensor = make_variable_non_differentiable_view(base, tensor);
    }
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

inline void check_no_requires_grad(const c10::optional<Tensor>& tensor, const char* name) {
  if (tensor.has_value()) {
    check_no_requires_grad(*tensor, name);
  }
}

inline void check_no_requires_grad(TensorList tensors, const char* name) {
  for (auto& tensor : tensors) {
    check_no_requires_grad(tensor, name);
  }
}

inline void check_no_requires_grad(const c10::List<c10::optional<Tensor>>& tensors, const char* name) {
  for (c10::optional<Tensor> tensor : tensors) {
    if (tensor.has_value()) {
      check_no_requires_grad(*tensor, name);
    }
  }
}

// Assumed that saved tensor lists are never inplace outputs
inline std::vector<SavedVariable> make_saved_variable_list(TensorList tensors) {
  return fmap(tensors, [](const Tensor& tensor) -> SavedVariable {
      return SavedVariable{tensor, false /* is output */}; });
}

// Assumed that saved tensor lists are never inplace outputs
inline std::vector<SavedVariable> make_saved_variable_list(const c10::List<c10::optional<at::Tensor>>& tensors) {
  return fmap(tensors, [](const c10::optional<Tensor>& tensor) -> SavedVariable {
    if (tensor.has_value()) {
      return SavedVariable{*tensor, false /* is output */};
    } else {
      return SavedVariable{Tensor(), false /* is output */};
    }
  });
}

inline std::vector<std::vector<int64_t>> to_args_sizes(TensorList tensors) {
  std::vector<std::vector<int64_t>> args_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    args_sizes[i] = tensors[i].sizes().vec();
  }
  return args_sizes;
}

inline std::vector<ScalarType> to_args_scalartypes(TensorList tensors) {
  std::vector<ScalarType> args_scalartypes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    args_scalartypes[i] = tensors[i].scalar_type();
  }
  return args_scalartypes;
}
}} // namespace torch::autograd
