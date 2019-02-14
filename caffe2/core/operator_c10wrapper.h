#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include "caffe2/core/operator.h"
#include <c10/util/ArrayRef.h>
#include <c10/util/Metaprogramming.h>
#include <ATen/core/ivalue.h>

namespace caffe2 {

namespace details {
template <size_t...>
struct true_t : std::true_type {};
template <class T>
using is_output_arg = std::is_same<Tensor*, T>;
template <class ParameterDef>
using extract_type_t = typename ParameterDef::type;
} // namespace details

/**
 * To make a c10 operator "C10Add" callable from caffe2 as "C2MyAddOpName", just
 * write
 *
 *     REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(C10Add, C2MyAddOpName)
 *
 */

template <
    const c10::OperatorHandle& (*OperatorHandle)(),
    class Context,
    bool use_array_input,
    size_t num_output_parameters,
    class ParameterDefTuple>
class C10OperatorWrapper final : public Operator<Context> {
 public:
  static_assert(
      c10::guts::is_instantiation_of<std::tuple, ParameterDefTuple>::value,
      "");
  using ParameterTuple =
      c10::guts::typelist::to_tuple_t<c10::guts::typelist::map_t<
          details::extract_type_t,
          c10::guts::typelist::from_tuple_t<ParameterDefTuple>>>;

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  C10OperatorWrapper(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        op_(OperatorHandle()),
        kernel_(at::nullopt),
        parameters_(parse_parameters_(
            operator_def,
            c10::guts::make_index_sequence<num_parameters()>())) {

    AT_ASSERT(operator_def.output_size() == op_.schema().returns().size());
    AT_ASSERT(operator_def.input_size() == num_inputs());
  }

  size_t num_inputs() {
    return op_.schema().arguments().size() - num_output_parameters - num_parameters();
  }

  static constexpr size_t num_parameters() {
    return std::tuple_size<ParameterDefTuple>::value;
  }

  bool RunOnDevice() override {
    // due to caching the stack_, concurrent calling is not allowed.
    // TODO thread_local might fix this
    std::lock_guard<std::mutex> lock(mutex_);

    AT_ASSERT(stack_.size() == 0);

    pushInputs_();
    pushParameters_(guts::make_index_sequence<num_parameters()>());
    pushOutputParameters_();

    callKernel_();

    popOutputs_();

    return true;
  }

 private:
  template <size_t... ParameterIndex>
  ParameterTuple parse_parameters_(
      const OperatorDef& operator_def,
      c10::guts::index_sequence<ParameterIndex...>) {
    return ParameterTuple{Parameter<ParameterIndex>(operator_def)...};
  }

  template <size_t Index>
  details::extract_type_t<
      typename std::tuple_element<Index, ParameterDefTuple>::type>
  Parameter(const OperatorDef& operator_def) {
    using Parameter =
        typename std::tuple_element<Index, ParameterDefTuple>::type;
    return Parameter::parse(ArgumentHelper(operator_def));
  }

  void pushInputs_() {
    if (use_array_input) {
      stack_.emplace_back(ivalue::TensorList::create(array_inputs_()));
    } else {
      for (size_t i = 0; i < num_inputs(); ++i) {
        stack_.emplace_back(at::Tensor(C10Tensor(Input(i))));
      }
    }
  }

  template<size_t... ParameterIndex>
  void pushParameters_(guts::index_sequence<ParameterIndex...>) {
    (void)std::initializer_list<int>{(
      stack_.emplace_back(std::get<ParameterIndex>(parameters_))
    , 0)...};
  }

  void pushOutputParameters_() {
    for (size_t i = 0; i < num_output_parameters; ++i) {
      caffe2::Tensor preallocated_output_tensor = OperatorBase::OutputTensorOrUndefined(i);
      if (preallocated_output_tensor.defined()) {
        stack_.emplace_back(at::Tensor(std::move(preallocated_output_tensor)));
      } else {
        stack_.emplace_back(IValue());
      }
    }
  }

  void callKernel_() {
    AT_ASSERT(stack_.size() == op_.schema().arguments().size());
    if (!kernel_.has_value()) {
      // TODO if kernel is already set, try re-dispatch to assert it goes to the same kernel
      kernel_ = c10::Dispatcher::singleton().lookup(op_, &stack_);
    }
    kernel_->call(&stack_);
  }

  void popOutputs_() {
    AT_ASSERT(stack_.size() == op_.schema().returns().size());
    for (size_t i = 0; i < op_.schema().returns().size(); ++i) {
      OperatorBase::SetOutputTensor(i, Tensor(C10Tensor(std::move(stack_[i]).toTensor())));
    }
    stack_.clear();
  }

  std::vector<at::Tensor> array_inputs_() {
    std::vector<at::Tensor> result;
    result.reserve(InputSize());
    for (size_t i = 0; i < InputSize(); ++i) {
      result.push_back(at::Tensor(c10::C10Tensor(Input(i))));
    }
    return result;
  }

  c10::OperatorHandle op_;
  c10::optional<OpKernel> kernel_;

  // this is stored as a member here to avoid having to re-allocate a stack
  // for each call. Between kernel calls, stack_.size() == 0, but capacity
  // should not need to be grown anymore after the first call.
  std::vector<IValue> stack_;
  std::mutex mutex_;

  ParameterTuple parameters_;
};

template <class ParameterDef>
struct ParameterHelper final {
  using type = typename ParameterDef::type;
  static typename ParameterDef::type parse(const ArgumentHelper& helper) {
    return helper.GetSingleArgument<typename ParameterDef::type>(
        ParameterDef::name(), ParameterDef::default_value());
  }
};

C10_DECLARE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

// TODO Currently we only register the CPU variant. This is going to be fixed
//      once the tensor detemplatization lands.
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(OperatorHandle, Name, NumOutputParameters)  \
  C10_REGISTER_CLASS(                                                                         \
      C10OperatorRegistry,                                                                    \
      Name,                                                                                   \
      C10OperatorWrapper<OperatorHandle, CPUContext, false, NumOutputParameters, std::tuple<>>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS( \
    OperatorHandle, Name, NumOutputParameters, ...)                \
  C10_REGISTER_CLASS(                                              \
      C10OperatorRegistry,                                         \
      Name,                                                        \
      C10OperatorWrapper<                                          \
          OperatorHandle,                                          \
          CPUContext,                                              \
          false,                                                   \
          NumOutputParameters,                                     \
          std::tuple<__VA_ARGS__>>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT( \
    OperatorHandle, Name, NumOutputParameters)                      \
  C10_REGISTER_CLASS(                                               \
      C10OperatorRegistry,                                          \
      Name,                                                         \
      C10OperatorWrapper<OperatorHandle, CPUContext, true, NumOutputParameters, std::tuple<>>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS( \
    OperatorHandle, Name, NumOutputParameters, ...)                                \
  C10_REGISTER_CLASS(                                                              \
      C10OperatorRegistry,                                                         \
      Name,                                                                        \
      C10OperatorWrapper<                                                          \
          OperatorHandle,                                                          \
          CPUContext,                                                              \
          true,                                                                    \
          NumOutputParameters,                                                     \
          std::tuple<__VA_ARGS__>>)

} // namespace caffe2
