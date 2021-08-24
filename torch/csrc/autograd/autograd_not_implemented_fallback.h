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

}

template<typename ReturnT, typename... Args>
struct ADInplaceOrViewFallback2Impl {
public:
  ADInplaceOrViewFallback2Impl(ReturnT(*op)(Args&&...), std::string schema_string)
   : op_(op), schema_(torch::jit::parseSchema(schema_string)) {
    // Precompute whether the operator is in-place or view or neither
    // This allows us to short circuit quickly when it is neither
    const auto& op_name = schema_.operator_name().name;
    const auto& arguments = schema_.arguments();
    const auto& returns = schema_.returns();
    const auto num_arguments = arguments.size();
    const auto num_returns = returns.size();

    for (int idx_arg = 0; idx_arg < num_arguments; idx_arg++) {
      const auto& alias_info = arguments[idx_arg].alias_info();
      if (alias_info.has_value()) {
        // Assume that if any of the alias info for input is write, then the same
        // input must be returned as-is as output (maybe parseSchema already checks this?)
        if (alias_info->isWrite()) {
          any_is_inplace_ = true;
        } else {
          AT_ASSERT(aliased_input_idx_ == -1, "Expected a single input to be aliased, but observed at least 2 input with alias info");
          aliased_input_idx_ = idx_arg;
        }
      }
    }

    for (int idx_ret = 0; idx_ret < num_returns; idx_ret++) {
      const auto& alias_info = returns[idx_ret].alias_info();
      if (alias_info.has_value()) {
        if (!alias_info->isWrite()) {
          AT_ASSERT(aliased_output_idx_ == -1, "Expected a single output to be aliased, but observed at least 2 outputs with alias info");
          aliased_output_idx_ = idx_ret;
        }
      }
    }
  };

  ReturnT operator() (Args&&... args) {
    if (aliased_input_idx_ == -1 && !any_is_inplace_) {
      // Fallthrough if neither in-place nor view
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
    const at::Tensor* aliased_output = nullptr;

    detail::for_each_input_tensor([&](size_t idx_tensor, size_t idx_arg, const at::Tensor& t) {
      const auto& alias_info = arguments[idx_arg].alias_info();
      if (alias_info.has_value()) {
        if (alias_info->isWrite()) {
          tensors_to_increment_version.push_back(&t);
        } else {
          AT_ASSERT(aliased_output == nullptr, "Expected a single output to be aliased, but observed at least 2 outputs with alias info");
          aliased_output = &t;
        }
      }
    }, args...);

    ReturnT outputs;
    {
      at::AutoDispatchBelowADInplaceOrView guard;
      // Do we need to forward here?
      // Is there a case where we'd want the args to be moved from?
      // another issue with the templated version that we cannot redispatch? we can only call directly back into
      // the user should be expected to call the dispatcher? or... can we?
      outputs = std::move(op_(args...));
    }

    detail::for_each_output_tensor(outputs, [&](size_t idx_tensor, size_t idx_arg, const at::Tensor& t) {
      const auto& alias_info = arguments[idx_arg].alias_info();
      if (alias_info.has_value() && !alias_info->isWrite()) {
        AT_ASSERT(aliased_input == nullptr, "Expected a single input to be aliased, but observed at least 2 inputs with alias info");
        aliased_input = &t;
      }
    });

    for (const auto i : c10::irange(tensors_to_increment_version.size())) {
      increment_version(*tensors_to_increment_version[i]);
    }

    // Make a base-view relationship
    if (aliased_output != nullptr) {
      auto result = as_view(
        /* base=*/*aliased_input,
        /* output=*/*aliased_output,
        /* is_bw_differentiable=*/ true,
        /* is_fw_differentiable=*/ true,
        /* view_func=*/ nullptr, // What are the consequences of not providing this?
        /* creation_meta=*/ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
    }
    // We need to replace the first
    return outputs;
  }
private:
  ReturnT(*op_)(Args&&...);
  c10::FunctionSchema schema_;
  bool any_is_inplace_ = false;
  int aliased_input_idx_ = -1;
  int aliased_output_idx_ = -1;
};


// We should instead do computation

template<typename ReturnT, typename... Args>
ADInplaceOrViewFallback2Impl<ReturnT, Args...> ADInplaceOrViewFallback2(ReturnT(*op)(Args&&...), std::string schema_str) {
  return ADInplaceOrViewFallback2Impl<ReturnT, Args...>(op, schema_str);
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


#define REGISTER_AUTOGRAD_NOT_IMPLEMENTED_FALLBACK2(ns, op, schema_str)     \
  TORCH_LIBRARY_IMPL(ns, Autograd, m) {                                     \
    m.impl(op, AutogradNotImplementedFallback());                            \
  }                                                                         \
  TORCH_LIBRARY_IMPL(ns, ADInplaceOrView, m) {                              \
    m.impl(op, autogradInplaceOrViewFallback2());                            \
  }

}} // namespace torch::autograd
