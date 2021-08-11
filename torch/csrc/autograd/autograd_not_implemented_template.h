#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/autograd.h>
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
  void operator()(const TensorList& tensors) const {
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
    }
    idx_tensor_++;
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

// TODO: We do not have access to this because of C++ reasons
// when building as an extension (why?)- this is available from custom_function.cpp
bool isFwGradDefined(const c10::optional<Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
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
void for_each_output_tensor(const at::Tensor& t, F fn) {
  fn(0, 0, t);
}

template<typename T, typename F>
enable_if_t<!std::__is_tuple_like<T>::value> for_each_output_tensor(T& not_tup, F fn) {
  (ForEachTensor<F>(fn))(not_tup);
}

template<class... Args, typename F>
void for_each_output_tensor(std::tuple<Args...>& tup, F fn) {
  for_each_tuple(tup,  ForEachTensor<F>(fn), std::index_sequence_for<Args...>());
}

}

template<typename ReturnT, typename... Args>
struct autogradNotImplementedFallback2 {
public:
  autogradNotImplementedFallback2(ReturnT(*op)(Args&&...), std::string schema_string)
   : op_(op), schema_(torch::jit::parseSchema(schema_string)) {};
  ReturnT operator() (Args&&... args) {
    const auto& op_name = schema_.operator_name().name;
    const auto& arguments = schema_.arguments();
    const auto& returns = schema_.returns();
    const auto num_arguments = arguments.size();
    const auto num_returns = returns.size();
    // const bool grad_mode = GradMode::is_enabled();

    // Keep track of which outputs are output of in-place modification
    // so we can rebase_history if necessary
    std::vector<bool> is_inplace_output;
    std::vector<bool> is_aliased_output;
    is_inplace_output.reserve(num_returns);
    is_aliased_output.reserve(num_returns);
    for (const auto i : c10::irange(num_returns)) {
      const auto& alias_info = returns[i].alias_info();
      is_inplace_output.push_back(alias_info.has_value() && alias_info->isWrite());
      is_aliased_output.push_back(alias_info.has_value());
    }
    std::vector<const at::Tensor*> tensors_requiring_grad;
    size_t num_tensor_inputs = 0;  // Only used for DEBUG-only checks

    bool any_requires_grad = compute_requires_grad(std::forward<Args>(args)...);
    // std::cout << "any_requires_grad: " << any_requires_grad << std::endl;

    detail::for_each_input_tensor([&](size_t _, size_t idx_arg, const at::Tensor& t) {
      TORCH_CHECK(t.defined(), "Expected argument ", idx_arg, " of ", op_name, " to be defined.");
      if (t.requires_grad()) {
        tensors_requiring_grad.push_back(&t);
      }
      num_tensor_inputs++;
      TORCH_CHECK_NOT_IMPLEMENTED(!detail::isFwGradDefined(t), "Trying to use forward AD with ", op_name, " that does not support it.");
    }, std::forward<Args>(args)...);

    std::shared_ptr<NotImplemented> grad_fn;
    if (any_requires_grad) {
      grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented(schema_.operator_name().name), deleteNode);
      // We can avoid creating tensors_requiring_grad vector if we modify collect_next_edges to ignore
      // inputs that don't require grad
      grad_fn->set_next_edges(collect_next_edges(tensors_requiring_grad));
    }

    detail::for_each_input_tensor([&](size_t idx_tensor, size_t idx_arg, const at::Tensor& t) {
      const auto& alias_info = arguments[idx_arg].alias_info();
      if (alias_info.has_value() && alias_info->isWrite()) {
        check_inplace(t, any_requires_grad);
      }
    }, std::forward<Args>(args)...);

    #ifndef NDEBUG
    // See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
    std::vector<c10::intrusive_ptr<TensorImpl>> impl_saved;
    impl_saved.reserve(num_tensor_inputs);
    std::vector<c10::optional<Storage>> storage_saved;
    storage_saved.reserve(num_tensor_inputs);
    detail::for_each_input_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
      storage_saved.push_back(t.has_storage() ? c10::optional<Storage>(t.storage()) : c10::nullopt);
    }, std::forward<Args>(args)...);
    detail::for_each_input_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
      impl_saved.push_back(t.getIntrusivePtr());
    }, std::forward<Args>(args)...);
    #endif
    ReturnT outputs;
    {
      at::AutoDispatchBelowADInplaceOrView guard;
      outputs = std::move(op_(std::forward<Args>(args)...));
    }
    #ifndef NDEBUG
    detail::for_each_input_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
      if (storage_saved.at(idx).has_value())
        AT_ASSERT(storage_saved.at(idx).value().is_alias_of(t.storage()));
    }, std::forward<Args>(args)...);
    detail::for_each_input_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
      if (impl_saved.at(idx))
        AT_ASSERT(impl_saved.at(idx) == t.getIntrusivePtr());
    }, std::forward<Args>(args)...);
    // Do we have alias information for tensors in tensorlist outputs?
    detail::for_each_output_tensor(outputs, [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
      if (!is_inplace_output[idx_ret])
        AT_ASSERT(t.use_count() <= 1);  // Okay to return undefined tensor
    });
    // TODO: Add check if view ops return tensor that is aliased with the right input
    detail::for_each_output_tensor(outputs, [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
      if (!is_aliased_output[idx_ret] && t.has_storage())
        AT_ASSERT(t.storage().use_count() == 1);
    });
    #endif

    if (any_requires_grad) {
      detail::for_each_output_tensor(outputs, [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
        if (isDifferentiableType(t.scalar_type())) {
          if (is_inplace_output[idx_ret]) {
            rebase_history(const_cast<at::Tensor&>(t), grad_fn);
          } else {
            set_history(const_cast<at::Tensor&>(t), grad_fn);
          }
        }
      });
    }
    return outputs;
  }
private:
  ReturnT(*op_)(Args&&...);
  c10::FunctionSchema schema_;
};

}
}
