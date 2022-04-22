#pragma once

#include <c10/util/irange.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/basic_ops.h>

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

namespace torch { namespace autograd {

// The requires_grad argument is used to know if the inplace operation needs
// gradient to be setup for it.
// In particular, we can have tensor.requires_grad() != requires_grad when writing
// a Tensor that requires gradients inplace into a Tensor that does not require gradients:
// a = torch.rand(2)
// b = torch.rand(2, requires_grad=True)
// a.copy_(b)
inline void check_inplace(const at::Tensor& tensor, bool requires_grad) {
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

inline void check_inplace(const at::TensorList tensors, bool requires_grad) {
  for (const auto& tensor : tensors) {
    check_inplace(tensor, requires_grad);
  }
}

inline void throw_error_out_requires_grad(const char* name) {
  AT_ERROR(
      name, "(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.");
}

inline void throw_error_for_complex_autograd(const at::Tensor& tensor, const char* name) {
  if (tensor.requires_grad()) {
    TORCH_CHECK(!tensor.is_complex(), name,
                " does not support automatic differentiation for outputs with complex dtype.");
  }
}

inline void throw_error_for_complex_autograd(const at::TensorList& tensorlist, const char* name) {
  for (const auto& tensor: tensorlist) {
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
        // NOLINTNEXTLINE(bugprone-use-after-move)
        auto output_nr = grad_fn->add_input_metadata(var);
        impl::rebase_history(var, {std::move(grad_fn), output_nr});
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
      }
    }
  }
}

inline void increment_version(const at::Tensor & t) {
  impl::bump_version(t);
}

struct Flatten : IterArgs<Flatten> {
  Flatten(variable_list& out) : out(out) {}
  variable_list& out;
  void operator()(const at::Tensor& x) { out.emplace_back(x); }
  void operator()(const c10::optional<at::Tensor>& x) {
    if (x.has_value()) out.emplace_back(x.value());
  }
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
inline at::Tensor as_view(const at::Tensor & base, const at::Tensor & tensor, bool is_bw_differentiable,
        bool is_fw_differentiable, std::function<at::Tensor(const at::Tensor&)> view_func=nullptr,
        CreationMeta creation_meta=CreationMeta::DEFAULT, bool allow_tensor_metadata_change=true) {
  // Note [View of inference tensor]
  // For inference tensor this code can only be hit outside InferenceMode
  // since ADInplaceOrView is in the default_included_set.
  // If Inplace and View were separate dispatch keys we can just put Inplace
  // in the default_included_set, so that view ops on inference tensor doesn't
  // have to go through as_view even outside InferenceMode.
  if (base.is_inference()) return tensor;

  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(base);

  // To speed up the most common case, we specially handle when both the forward and backward
  // view infos are the same, and so a single shared ViewInfo can be used for both of them.
  if ((!diff_view_meta || diff_view_meta->shared_view_info()) && is_bw_differentiable && is_fw_differentiable) {
    if (diff_view_meta) {
      creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
      return make_variable_differentiable_view(tensor, diff_view_meta->get_backward_view().chain(base, tensor, view_func),
                        c10::nullopt, /*shared_view_info*/ true, creation_meta, allow_tensor_metadata_change);
    } else {
      return make_variable_differentiable_view(tensor, ViewInfo(base, view_func), c10::nullopt, /*shared_view_info*/ true,
                                             creation_meta, allow_tensor_metadata_change);
    }
  }

  // If they cannot be shared, create the required view infos
  c10::optional<ViewInfo> new_bw_info;
  c10::optional<ViewInfo> new_fw_info;

  if (is_bw_differentiable) {
    if (diff_view_meta && diff_view_meta->has_bw_view()) {
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
    if (diff_view_meta && diff_view_meta->has_fw_view()) {
      const auto& base_fw_info = diff_view_meta->get_forward_view();
      new_fw_info = base_fw_info.chain(base, tensor, view_func);
    } else {
      new_fw_info = ViewInfo(base, view_func);
    }
  }

  if (is_fw_differentiable || is_bw_differentiable) {
    if (diff_view_meta && diff_view_meta->has_bw_view()) {
      creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
    }
    return make_variable_differentiable_view(tensor, std::move(new_bw_info), std::move(new_fw_info), /*shared_view_info*/ false,
                                             creation_meta, allow_tensor_metadata_change);
  } else {
    return make_variable_non_differentiable_view(base, tensor, allow_tensor_metadata_change);
  }
}

// See NOTE [ Autograd View Variables ] for details.
inline std::vector<at::Tensor> as_view(const at::Tensor & base, std::vector<at::Tensor>& tensors, bool is_bw_differentiable,
                                   bool is_fw_differentiable, CreationMeta creation_meta=CreationMeta::DEFAULT) {
  // See Note [View of inference tensor]
  if (base.is_inference()) return tensors;

  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(base);

  // Special case when view info can be shared for forward and backward differentiable views
  if ((!diff_view_meta || diff_view_meta->shared_view_info()) && is_bw_differentiable && is_fw_differentiable) {
    c10::optional<ViewInfo> new_shared_info;
    if (diff_view_meta) {
      // TODO: fix fb internal use-case so that it doesn't trigger this internal assert when the base is not a view.
      // For now, we only do that same (wrong) thing as the old code which is to only check when the inputs is a
      // backward differentiable view
      if (diff_view_meta->has_bw_view()) {
        TORCH_INTERNAL_ASSERT(creation_meta == CreationMeta::NO_GRAD_MODE || creation_meta == CreationMeta::INFERENCE_MODE ||
            creation_meta == CreationMeta::MULTI_OUTPUT_NODE,
            "Functions that result multiple view must have a creation meta reflecting this behavior or more restrictive.");
      }
      creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
      const auto& base_bw_info = diff_view_meta->get_backward_view();
      new_shared_info = ViewInfo(base_bw_info.base_, /* view_func */ nullptr);
    } else {
      new_shared_info = ViewInfo(base, /* view_func */ nullptr);
    }

    for(at::Tensor &tensor : tensors) {
      if (is_fw_differentiable || is_bw_differentiable) {
        tensor = make_variable_differentiable_view(tensor, new_shared_info, c10::nullopt, /*shared_view_info*/ true, creation_meta);
      } else {
        tensor = make_variable_non_differentiable_view(base, tensor);
      }
    }
    return tensors;
  }

  c10::optional<ViewInfo> new_bw_info = c10::nullopt;
  c10::optional<ViewInfo> new_fw_info = c10::nullopt;

  if (is_bw_differentiable) {
    auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(base);
    if (diff_view_meta && diff_view_meta->has_bw_view()) {
      const auto& base_bw_info = diff_view_meta->get_backward_view();
      // TODO: fix fb internal use-case so that it doesn't trigger this internal assert when the base is not a view.
      // In this code, the assert should be outside of the if statement.
      TORCH_INTERNAL_ASSERT(creation_meta == CreationMeta::NO_GRAD_MODE || creation_meta == CreationMeta::INFERENCE_MODE ||
          creation_meta == CreationMeta::MULTI_OUTPUT_NODE,
          "Functions that result multiple view must have a creation meta reflecting this behavior or more restrictive.");
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
  if (is_fw_differentiable) {
    // Check if base is a forward differentiabble view
    auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(base);
    if (diff_view_meta && diff_view_meta->has_fw_view()) {
      const auto& base_fw_info = diff_view_meta->get_forward_view();
      TORCH_INTERNAL_ASSERT(creation_meta == CreationMeta::NO_GRAD_MODE || creation_meta == CreationMeta::INFERENCE_MODE ||
          creation_meta == CreationMeta::MULTI_OUTPUT_NODE,
          "Functions that result multiple view must have a creation meta reflecting this behavior or more restrictive.");
      // It is ok to create a ViewInfo where only the base is correct in this case as inplace operations on such views are
      // not allowed
      new_fw_info = ViewInfo(base_fw_info.base_, /* view_func */ nullptr);
    } else {
      new_fw_info = ViewInfo(base, /* view_func */ nullptr);
    }
  }

  if ((is_fw_differentiable || is_bw_differentiable) && base.is_view()) {
    // is_view() => diff_view_meta
    auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(base);
    creation_meta = propagate_creation_meta(diff_view_meta->get_creation_meta(), creation_meta);
  }

  for(at::Tensor &tensor : tensors) {
    if (is_fw_differentiable || is_bw_differentiable) {
      tensor = make_variable_differentiable_view(tensor, new_bw_info, new_fw_info, /*shared_view_info*/ false, creation_meta);
    } else {
      tensor = make_variable_non_differentiable_view(base, tensor);
    }
  }
  return tensors;
}

inline void check_no_requires_grad(const at::Tensor& tensor, const char* name,
                                   const char* fn_name="", bool check_grad_mode=true) {
  TORCH_CHECK(!(tensor.defined() && tensor.requires_grad()) || !(check_grad_mode && GradMode::is_enabled()),
              "The function '", fn_name, "' is not differentiable with respect to argument '", name,
              "'. This input cannot have requires_grad True.");
}

inline void check_no_requires_grad(const c10::optional<at::Tensor>& tensor, const char* name, const char* fn_name="") {
  if (tensor.has_value()) {
    check_no_requires_grad(*tensor, name, fn_name);
  }
}

inline void check_no_requires_grad(at::TensorList tensors, const char* name, const char* fn_name="") {
  // GradMode check is expensive, so check it only once for TensorLists
  if (!GradMode::is_enabled()) {
    return;
  }
  for (auto& tensor : tensors) {
    check_no_requires_grad(tensor, name, fn_name, /*check_grad_mode*/ false);
  }
}

inline void check_no_requires_grad(const c10::List<c10::optional<at::Tensor>>& tensors, const char* name, const char* fn_name="") {
  // GradMode check is expensive, so check it only once for TensorLists
  if (!GradMode::is_enabled()) {
    return;
  }
  for (c10::optional<at::Tensor> tensor : tensors) {
    if (tensor.has_value()) {
      check_no_requires_grad(*tensor, name, fn_name, /*check_grad_mode*/ false);
    }
  }
}

// Assumed that saved tensor lists are never inplace outputs
inline std::vector<SavedVariable> make_saved_variable_list(at::TensorList tensors) {
  return fmap(tensors, [](const at::Tensor& tensor) -> SavedVariable {
      return SavedVariable{tensor, false /* is output */}; });
}

// Assumed that saved tensor lists are never inplace outputs
inline std::vector<SavedVariable> make_saved_variable_list(const c10::List<c10::optional<at::Tensor>>& tensors) {
  return fmap(tensors, [](const c10::optional<at::Tensor>& tensor) -> SavedVariable {
    if (tensor.has_value()) {
      return SavedVariable{*tensor, false /* is output */};
    } else {
      return SavedVariable{at::Tensor(), false /* is output */};
    }
  });
}

inline std::vector<std::vector<int64_t>> to_args_sizes(at::TensorList tensors) {
  std::vector<std::vector<int64_t>> args_sizes(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    args_sizes[i] = tensors[i].sizes().vec();
  }
  return args_sizes;
}

inline std::vector<c10::ScalarType> to_args_scalartypes(at::TensorList tensors) {
  std::vector<c10::ScalarType> args_scalartypes(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    args_scalartypes[i] = tensors[i].scalar_type();
  }
  return args_scalartypes;
}

}} // namespace torch::autograd
