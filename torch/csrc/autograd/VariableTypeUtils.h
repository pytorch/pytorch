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

inline void check_inplace(const Tensor& tensor) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.requires_grad() && GradMode::is_enabled()) {
    if (var.is_view()) {
      // NB: is_view() ==> get_autograd_meta()
      auto diff_view_meta = static_cast<DifferentiableViewMeta*>(impl::get_autograd_meta(var));
      // This can throw or warn
      handle_view_on_rebase(diff_view_meta);
    }
    if (var.is_leaf()) {
      AT_ERROR(
        "a leaf Variable that requires grad is being used in an in-place operation.");
    }
  }
}

inline void check_inplace(const TensorList tensors) {
  for (const auto& tensor : tensors) {
    check_inplace(tensor);
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
inline Tensor as_view(const Tensor & base, Tensor tensor, bool is_differentiable,
        c10::optional<std::function<Tensor(const Tensor&)>> view_func=c10::nullopt,
        CreationMeta creation_meta=CreationMeta::DEFAULT) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    // Set `view_func` using the root base as input.
    // `view_func` is used to recover views in backward when either as_strided is not supported
    // or the view function changes the metadata which is not recorded by as_strided
    // See Note [View + Inplace update on base tensor] and [View + Inplace update on view tensor]
    // for more details how we use this function in backward.
    auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(base_var));
    if (view_func.has_value()) {
      auto fn = view_func.value();
      // both current_view and it's parent have a view_func
      if (diff_view_meta->has_view_fn()) {
        auto prev_fn = diff_view_meta->view_fn();
        view_func = [=](const at::Tensor& root_base) {
          auto temp = prev_fn(root_base);
          return fn(temp);
        };
      } else {
        // current_view has a view_func and but it's parent doesn't have one
        if(base_var.unsafeGetTensorImpl()->support_as_strided()) {
          auto size = base.sizes().vec();
          auto stride = base.strides().vec();
          auto storage_offset = base.storage_offset();
          view_func = [=](const at::Tensor& root_base) {
            auto temp = root_base.as_strided(size, stride, storage_offset);
            return fn(temp);
          };
        } else {
          // When base_var is a view but doesn't carry a view_fn in DifferentiableViewMeta, it's
          // a view that doesn't support inplace update, e.g. unbind.
          // In this case we should throw an error when inplace update happens in **forward**.
          // One would naturally think the following function will be first called in backward pass.
          // But the first call site is indeed in **forward** pass when we refresh `grad_fn`
          // triggered by inplace update.
          // Search Note [View + Inplace update for view tensor] to for the call site.
          view_func = [=](const at::Tensor& root_base) {
            TORCH_CHECK(false, "This view is the output of a function that returns multiple views."
                    "Such functions do not allow the output views to be modified inplace."
                    "You should replace the inplace operation by an out-of-place one");
            return root_base;
          };
        }
      }
    } else if(diff_view_meta->has_view_fn()) {
      // if current_view doesn't have a view_func but it's parent has one
      auto prev_view_fn = diff_view_meta->view_fn();
      auto size = tensor.sizes().vec();
      auto stride = tensor.strides().vec();
      auto storage_offset = tensor.storage_offset();
      view_func = [=](const at::Tensor& root_base) {
        auto temp = prev_view_fn(root_base);
        return temp.as_strided(size, stride, storage_offset);
      };
    }
    base_var = base_var._base();
  }
  if (is_differentiable) {
    return make_variable_differentiable_view(std::move(base_var), std::move(tensor), creation_meta, std::move(view_func));
  } else {
    TORCH_CHECK(creation_meta == CreationMeta::DEFAULT,
                "Non-differentiable views must have creation_meta=CreationMeta::DEFAULT");
    return make_variable_non_differentiable_view(std::move(base_var), std::move(tensor));
  }
}

// See NOTE [ Autograd View Variables ] for details.
inline std::vector<Tensor> as_view(const Tensor & base, std::vector<Tensor> tensors, bool is_differentiable,
                                   CreationMeta creation_meta=CreationMeta::DEFAULT) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var._base();
  }
  for(Tensor &tensor : tensors) {
    if (is_differentiable) {
      tensor = make_variable_differentiable_view(base_var, std::move(tensor), creation_meta);
    } else {
      TORCH_CHECK(creation_meta == CreationMeta::DEFAULT,
                  "Non-differentiable views must have creation_meta=CreationMeta::DEFAULT");
      tensor = make_variable_non_differentiable_view(base_var, std::move(tensor));
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
