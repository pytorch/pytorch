#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

namespace torch {
namespace autograd {

namespace detail {

template <typename F>
struct ForEachTensor : IterArgs<ForEachTensor<F>> {
  F fn_;
  mutable size_t idx_arg_ = 0;
  mutable size_t idx_tensor_ = 0;
  ForEachTensor(F fn) : fn_(fn) {}
  void operator()(const variable_list& tensors) const {
    for (const auto& t : tensors) {
      fn_(idx_arg_, idx_tensor_, t);
      idx_tensor_++;
    }
    idx_arg_++;
  }
  void operator()(const at::TensorList& tensors) const {
    for (const auto& t : tensors) {
      fn_(idx_tensor_, idx_arg_ , t);
      idx_tensor_++;
    }
    idx_arg_++;
  }
  void operator()(const c10::optional<at::Tensor>& x) const {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (x.has_value() && x.value().defined()) {
      fn_(idx_tensor_, idx_arg_, x.value());
      idx_tensor_++;
    }
    idx_arg_++;
  }
  void operator()(const at::Tensor& x) const {
    fn_(idx_tensor_, idx_arg_, x);
    idx_tensor_++;
    idx_arg_++;
  }
  template <typename T>
  void operator()(const T& x) const {
    // no op
  }
};

template <typename F, typename... Args>
inline void for_each_input_tensor(F fn, Args&&... args) {
  ForEachTensor<F>(fn).apply(std::forward<Args>(args)...);
}


template <class Mapper, class... Args, size_t... Indices>
void for_each_tuple(
    std::tuple<Args...>& tuple,
    const Mapper& mapper,
    std::index_sequence<Indices...>) {
  // We need a context where the parameter pack can be expanded
  auto l = { (mapper(std::forward<Args>(std::get<Indices>(tuple))), 0)...};
}

template<typename F>
void for_each_output_tensor(at::Tensor& t, F fn) {
  fn(0, 0, t);
}

template<typename T, typename F>
void for_each_output_tensor(T& not_tup, F fn) {
  (ForEachTensor<F>(fn))(not_tup);
}

template<class... Args, typename F>
void for_each_output_tensor(std::tuple<Args...>& tup, F fn) {
  for_each_tuple(tup,  ForEachTensor<F>(fn), std::index_sequence_for<Args...>());
}

// Why are there all these overloads? and why are there placeholders internal asserts?
template<typename ReturnT, typename... Args>
ReturnT call_with_new_first_arg(ReturnT(*op)(Args...), const at::Tensor& new_t, const at::Tensor& old_t, Args&&... args) {
  return op(new_t, std::forward<Args>(args)...);
}

template<typename ReturnT, typename... Args>
ReturnT call_with_new_first_arg(ReturnT(*op)(Args...), const at::Tensor& new_t, Args&&... args) {
  // Placeholder
  TORCH_INTERNAL_ASSERT(false, "Expect first argument to be a tensor when need_view_func=true");
  return op(std::forward<Args>(args)...);
}

template<typename... Ts>
at::Tensor get_first_return(std::tuple<at::Tensor, Ts...>& tup) {
  return std::get<0>(tup);
}

template<typename... Ts>
at::Tensor get_first_return(std::tuple<Ts...>& tup) {
  TORCH_INTERNAL_ASSERT(false, "Expect first return to be a single tensor when need_view_func=true");
  return at::Tensor();
}

// We don't need this anymore
inline bool isFwGradDefined(const c10::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

inline at::Tensor get_first_return(const at::Tensor& t) {
  return t;
}

// Will this be ambiguous?
template<typename T>
at::Tensor get_first_return(const T& t) {
  TORCH_INTERNAL_ASSERT(false, "Expect first return to be a single tensor when need_view_func=true");
  return at::Tensor();
}

template<typename T>
struct TD;

template<typename T>
T call_as_view(T&& t, const at::Tensor* aliased_input, std::function<at::Tensor(const at::Tensor&)> view_func) {
  // Placeholder because compiler doesn't realize that non-view won't reach here
  TORCH_INTERNAL_ASSERT(false);
  return std::forward<T>(t);
}

template<typename... Args>
variable_list call_as_view(variable_list&& output, const at::Tensor* aliased_input, std::function<at::Tensor(const at::Tensor&)> view_func) {
  TORCH_INTERNAL_ASSERT(view_func == nullptr);
  return as_view(
    /* base=*/*aliased_input,
    /* output=*/output,
    /* is_bw_differentiable=*/true,
    /* is_fw_differentiable=*/true,
    /* creation_meta=*/InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
    // ^ pass in creation meta unecessarily even if not isDifferentiableType
}

template<typename... Args>
std::tuple<variable_list, Args...> call_as_view(std::tuple<variable_list, Args...> outputs, const at::Tensor* aliased_input, std::function<at::Tensor(const at::Tensor&)> view_func) {
  TORCH_INTERNAL_ASSERT(view_func == nullptr);
  auto result = as_view(
    /* base=*/*aliased_input,
    /* output=*/std::get<0>(outputs),
    /* is_bw_differentiable=*/true,
    /* is_fw_differentiable=*/true,
    /* creation_meta=*/InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
    // ^ pass in creation meta unecessarily even if not isDifferentiableType
  std::get<0>(outputs) = result;
  return outputs;
}

template<typename... Args>
std::tuple<at::Tensor, Args...> call_as_view(std::tuple<at::Tensor, Args...> outputs, const at::Tensor* aliased_input, std::function<at::Tensor(const at::Tensor&)> view_func) {
  if (isDifferentiableType(std::get<0>(outputs).scalar_type())) {
    auto result = as_view(
      /* base=*/*aliased_input,
      /* output=*/std::get<0>(outputs),
      /* is_bw_differentiable=*/true,
      /* is_fw_differentiable=*/true,
      /* view_func=*/view_func,
      /* creation_meta=*/InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
    std::get<0>(outputs) = result;
    return outputs;
  } else {
    auto result = as_view(
      /* base=*/*aliased_input,
      /* output=*/std::get<0>(outputs),
      /* is_bw_differentiable=*/false,
      /* is_fw_differentiable=*/false);
    std::get<0>(outputs) = result;
    return outputs;
  }
}

inline at::Tensor call_as_view(at::Tensor&& output, const at::Tensor* aliased_input, std::function<at::Tensor(const at::Tensor&)> view_func) {
  // Factor this out
  if (isDifferentiableType(output.scalar_type())) {
      return as_view(
        /* base=*/*aliased_input,
        /* output=*/output,
        /* is_bw_differentiable=*/true,
        /* is_fw_differentiable=*/true,
        /* view_func=*/view_func,
        /* creation_meta=*/InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  } else {
    return as_view(
    /* base=*/*aliased_input,
    /* output=*/output,
    /* is_bw_differentiable=*/false,
    /* is_fw_differentiable=*/false);
  }
}

template<typename T>
struct remove_reference_if_rvalue_reference {
  using type = std::conditional_t<
                std::is_rvalue_reference<T>::value,
                std::remove_reference_t<T>,
                T>;
};

static const std::vector<std::string> NEEDS_METADATA_CHANGE = {"aten::view_as_complex", "aten::view_as_real", "aten::_conj", "aten::_neg_view"};

// template<typename ReturnT>
// typename detail::remove_reference_if_rvalue_reference<ReturnT>::type
// post_process(ReturnT&& ret, const std::vector<const at::Tensor*>& tensors_to_increment_version, const at::Tensor* aliased_input, std::function<at::Tensor(const at::Tensor&)> view_func, bool is_view_) {
//   for (const auto i : c10::irange(tensors_to_increment_version.size())) {
//     increment_version(*tensors_to_increment_version[i]);
//   }
//   if (is_view_) {
//     return detail::call_as_view(std::forward<ReturnT>(ret), aliased_input, view_func);
//   } else {
//     return std::forward<ReturnT>(ret);
//   }
// };

} // namespace detail

template<typename ReturnT, typename... Args>
struct ADInplaceOrViewFallback2Impl {
public:
  ADInplaceOrViewFallback2Impl(ReturnT(*op)(Args...), std::string fqname, std::string schema_string)
   : op_(op), fqname_(fqname), schema_(torch::jit::parseSchema(schema_string)) {
    // Precompute whether the operator is in-place or view or neither
    // This allows us to short circuit quickly when it is neither
    const auto& op_name = schema_.operator_name().name;
    const auto& arguments = schema_.arguments();
    const auto& returns = schema_.returns();
    const auto num_arguments = arguments.size();
    const auto num_returns = returns.size();
    int aliased_input_idx = -1;
    int aliased_output_idx = -1;

    for (int idx_arg = 0; idx_arg < num_arguments; idx_arg++) {
      const auto& alias_info = arguments[idx_arg].alias_info();
      if (alias_info.has_value()) {
        // Assume that if any of the alias info for input is write, then the same
        // input must be returned as-is as output (maybe parseSchema already checks this?)
        if (alias_info->isWrite()) {
          any_is_inplace_ = true;
        } else {
          AT_ASSERT(aliased_input_idx == -1, "Expected a single input to be aliased, but observed at least 2 input with alias info");
          aliased_input_idx = idx_arg;
        }
      }
    }
    for (int idx_ret = 0; idx_ret < num_returns; idx_ret++) {
      const auto& alias_info = returns[idx_ret].alias_info();
      if (alias_info.has_value()) {
        if (!alias_info->isWrite()) {
          AT_ASSERT(aliased_output_idx == -1, "Expected a single output to be aliased, but observed at least 2 outputs with alias info");
          aliased_output_idx = idx_ret;
        }
      }
    }
    TORCH_INTERNAL_ASSERT((aliased_input_idx == -1 && aliased_output_idx == -1) ||
      (aliased_input_idx == 0 && aliased_output_idx == 0))

    is_view_ = aliased_input_idx != -1;
    needs_metadata_change_ = std::find(
      detail::NEEDS_METADATA_CHANGE.begin(), detail::NEEDS_METADATA_CHANGE.end(), op_name) != detail::NEEDS_METADATA_CHANGE.end();
  };

  ReturnT operator() (Args&&... args) {
    if (!is_view_ && !any_is_inplace_) {
      at::AutoDispatchBelowADInplaceOrView guard;
      return op_(std::forward<Args>(args)...);
    }
    const auto& op_name = schema_.operator_name().name;
    const auto& arguments = schema_.arguments();
    const auto& returns = schema_.returns();
    const auto num_arguments = arguments.size();
    const auto num_returns = returns.size();

    std::vector<const at::Tensor*> tensors_to_increment_version;
    const at::Tensor* aliased_input = nullptr;

    detail::for_each_input_tensor([&](size_t idx_tensor, size_t idx_arg, const at::Tensor& t) {
      const auto& alias_info = arguments[idx_arg].alias_info();
      if (alias_info.has_value()) {
        if (alias_info->isWrite()) {
          tensors_to_increment_version.push_back(&t);
        } else {
          AT_ASSERT(aliased_input == nullptr, "Expected a single input to be aliased, but observed at least 2 outputs with alias info");
          aliased_input = &t;
        }
      }
    }, std::forward<Args>(args)...);

    std::function<at::Tensor(const at::Tensor&)> view_func = nullptr;
    // For this to compile, we should extract it out its creation, just like the others
    // if (is_view_ && (needs_metadata_change_  && !aliased_input->unsafeGetTensorImpl()->support_as_strided())) {
    //   view_func = [=](const at::Tensor& t) {
    //     return detail::get_first_return(detail::call_with_new_first_arg(op_, t, std::forward<Args>(args)...));
    //   };
    // }

    // We need to call into a separate function because sometimes we return references and
    // sometimes we return by value, and the code needs to work for both cases.
    // Having a templated function that takes a univeral reference is useful for handling this.
    // There might be a simpler way though
    auto post_process = [&](auto&& ret) -> typename detail::remove_reference_if_rvalue_reference<decltype(ret)>::type {
      for (const auto i : c10::irange(tensors_to_increment_version.size())) {
        increment_version(*tensors_to_increment_version[i]);
      }
      if (is_view_) {
        return detail::call_as_view(std::forward<decltype(ret)>(ret), aliased_input, view_func);
      } else {
        return std::forward<decltype(ret)>(ret);
      }
    };
    // We should do the redispatch ourselves (this code doesn't quite work yet...)
    // User needs to pass in ns::op_name now; there might be a way to extract that from the scheam though
    // auto operatorHandle = c10::Dispatcher::singleton()
    //   .findSchemaOrThrow(fqname_.c_str(), "")
    //   .typed<decltype(op_)>();
    // at::AutoDispatchBelowADInplaceOrView guard;

    // This lambda doesn't seem to have overhead
    // return detail::post_process(operatorHandle.call(std::forward<Args>(args)...), tensors_to_increment_version, aliased_input, view_func, is_view_);
    return post_process(op_(std::forward<Args>(args)...));
  }
private:
  ReturnT(*op_)(Args...);
  std::string fqname_;
  c10::FunctionSchema schema_;
  bool any_is_inplace_ = false;
  bool is_view_ = false;
  bool needs_metadata_change_ = false;
};

// Bringing this back, so we can measure performance of
// boxed autograd + boxed InplaceOrView vs templated autograd + templated InplaceOrView
template<typename ReturnT, typename... Args>
struct AutogradNotImplementedFallback2Impl {
public:
  AutogradNotImplementedFallback2Impl(ReturnT(*op)(Args&&...), std::string fqname, std::string schema_string)
   : op_(op), fqname_(fqname), schema_(torch::jit::parseSchema(schema_string)) {};
  ReturnT operator() (Args&&... args) {
    // We should really have a conditional here, and dispatch past InplaceOrView if neither view nor inplace
    // but for the purposes of testing perf of InplaceOrView, we keep this here.
    // We should also be "redispatching"
    at::AutoDispatchBelowAutograd guard;
    auto operatorHandle = c10::Dispatcher::singleton()
      .findSchemaOrThrow(fqname_.c_str(), "")
      .typed<decltype(op_)>();
    return operatorHandle.call(std::forward<Args>(args)...);
  }
private:
  ReturnT(*op_)(Args&&...);
  std::string fqname_;
  c10::FunctionSchema schema_;
};

// We should instead do computation

template<typename ReturnT, typename... Args>
ADInplaceOrViewFallback2Impl<ReturnT, Args...> ADInplaceOrViewFallback2(ReturnT(*op)(Args...), std::string fqname, std::string schema_str) {
  return ADInplaceOrViewFallback2Impl<ReturnT, Args...>(op, fqname, schema_str);
}

template<typename ReturnT, typename... Args>
AutogradNotImplementedFallback2Impl<ReturnT, Args...> autogradNotImplementedFallback2(ReturnT(*op)(Args...), std::string fqname, std::string schema_str) {
  return AutogradNotImplementedFallback2Impl<ReturnT, Args...>(op, fqname, schema_str);
}

TORCH_API torch::CppFunction autogradNotImplementedFallback();

TORCH_API torch::CppFunction ADInplaceOrViewFallback();

#define REGISTER_AUTOGRAD_NOT_IMPLEMENTED_FALLBACK(ns, op)      \
  TORCH_LIBRARY_IMPL(ns, Autograd, m) {                         \
    m.impl(op, autogradNotImplementedFallback());                \
  }                                                             \
  TORCH_LIBRARY_IMPL(ns, ADInplaceOrView, m) {                  \
    m.impl(op, ADInplaceOrViewFallback());                       \
  }

#define REGISTER_AUTOGRAD_NOT_IMPLEMENTED_FALLBACK2(ns, op, fqname, schema_str, fn)     \
  TORCH_LIBRARY_IMPL(ns, Autograd, m) {                                     \
    m.impl(op, autogradNotImplementedFallback());                            \
  }                                                                         \
  TORCH_LIBRARY_IMPL(ns, ADInplaceOrView, m) {                              \
    m.impl(op, ADInplaceOrViewFallback2(fn, fqname, schema_str));                            \
  }

}} // namespace torch::autograd
