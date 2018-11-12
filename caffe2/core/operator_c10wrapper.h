#pragma once

#include "caffe2/core/dispatch/Dispatcher.h"
#include "caffe2/core/operator.h"
#include <ATen/core/ArrayRef.h>
#include <c10/util/Metaprogramming.h>

namespace caffe2 {

namespace details {
template <size_t...>
struct true_t : std::true_type {};
template <class State>
inline std::shared_ptr<State> init_state() {
  return std::make_shared<State>();
}
template <>
inline std::shared_ptr<void> init_state<void>() {
  return std::shared_ptr<void>();
}
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
 * Note: This wrapper currently only supports C10 ops that have exactly one
 * output and take that in the last parameter as "Tensor* output".
 * TODO: Figure out a better way to handle output parameters
 */

template <
    class OpSchemaDef,
    class Context,
    class State,
    bool use_array_input,
    class ParameterDefTuple>
class C10OperatorWrapper final : public Operator<Context> {
  using Schema = c10::OpSchema<OpSchemaDef>;

 public:
  static_assert(
      c10::guts::is_instantiation_of<std::tuple, ParameterDefTuple>::value,
      "");
  using ParameterTuple =
      c10::guts::typelist::to_tuple_t<c10::guts::typelist::map_t<
          details::extract_type_t,
          c10::guts::typelist::from_tuple_t<ParameterDefTuple>>>;

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  static constexpr bool op_has_context_argument = std::is_same<
      BaseContext*,
      c10::guts::typelist::last_t<
          typename Schema::signature::parameter_types>>::value;
  static constexpr bool op_has_state_argument =
      !std::is_same<void, State>::value;

  C10OperatorWrapper(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        state_(details::init_state<State>()),
        parameters_(parse_parameters_(
            operator_def,
            c10::guts::make_index_sequence<num_parameters()>())) {}

  static constexpr size_t num_inputs() {
    return Schema::signature::num_args - num_outputs() - num_parameters() -
        (op_has_context_argument ? 1 : 0) - (op_has_state_argument ? 1 : 0);
  }

  static constexpr size_t num_parameters() {
    return std::tuple_size<ParameterDefTuple>::value;
  }

  static constexpr size_t num_outputs() {
    return c10::guts::typelist::count_if<
        details::is_output_arg,
        typename Schema::signature::parameter_types>::value;
  }

  bool RunOnDevice() override {
    RunOnDevice_(
        c10::guts::make_index_sequence<num_inputs()>(),
        c10::guts::make_index_sequence<num_outputs()>(),
        c10::guts::make_index_sequence<num_parameters()>());
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

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && op_has_context_argument &&
          op_has_state_argument && !use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        Input(InputIndex)...,
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...,
        state_.get(),
        static_cast<BaseContext*>(&context_));
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && op_has_context_argument &&
          !op_has_state_argument && !use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        Input(InputIndex)...,
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...,
        static_cast<BaseContext*>(&context_));
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && !op_has_context_argument &&
          op_has_state_argument && !use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        Input(InputIndex)...,
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...,
        state_.get());
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && !op_has_context_argument &&
          !op_has_state_argument && !use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        Input(InputIndex)...,
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...);
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && op_has_context_argument &&
          op_has_state_argument && use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        at::ArrayRef<const Tensor*>(array_inputs_()),
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...,
        state_.get(),
        static_cast<BaseContext*>(&context_));
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && op_has_context_argument &&
          !op_has_state_argument && use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        at::ArrayRef<const Tensor*>(array_inputs_()),
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...,
        static_cast<BaseContext*>(&context_));
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && !op_has_context_argument &&
          op_has_state_argument && use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        at::ArrayRef<const Tensor*>(array_inputs_()),
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...,
        state_.get());
  }

  template <
      size_t... InputIndex,
      size_t... OutputIndex,
      size_t... ParameterIndex>
  c10::guts::enable_if_t<
      details::true_t<InputIndex...>::value && !op_has_context_argument &&
          !op_has_state_argument && use_array_input,
      void>
  RunOnDevice_(
      c10::guts::index_sequence<InputIndex...>,
      c10::guts::index_sequence<OutputIndex...>,
      c10::guts::index_sequence<ParameterIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(
        at::ArrayRef<const Tensor*>(array_inputs_()),
        Output(OutputIndex)...,
        std::get<ParameterIndex>(parameters_)...);
  }

  std::vector<const Tensor*> array_inputs_() {
    std::vector<const Tensor*> result;
    result.reserve(InputSize());
    for (size_t i = 0; i < InputSize(); ++i) {
      result.push_back(&Input(i));
    }
    return result;
  }

  std::shared_ptr<State> state_;

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
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(OpSchemaDef, State, Name) \
  C10_REGISTER_CLASS(                                                       \
      C10OperatorRegistry,                                                  \
      Name,                                                                 \
      C10OperatorWrapper<OpSchemaDef, CPUContext, State, false, std::tuple<>>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS( \
    OpSchemaDef, State, Name, ...)                                 \
  C10_REGISTER_CLASS(                                              \
      C10OperatorRegistry,                                         \
      Name,                                                        \
      C10OperatorWrapper<                                          \
          OpSchemaDef,                                             \
          CPUContext,                                              \
          State,                                                   \
          false,                                                   \
          std::tuple<__VA_ARGS__>>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT( \
    OpSchemaDef, State, Name)                                       \
  C10_REGISTER_CLASS(                                               \
      C10OperatorRegistry,                                          \
      Name,                                                         \
      C10OperatorWrapper<OpSchemaDef, CPUContext, State, true, std::tuple<>>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS( \
    OpSchemaDef, State, Name, ...)                                                 \
  C10_REGISTER_CLASS(                                                              \
      C10OperatorRegistry,                                                         \
      Name,                                                                        \
      C10OperatorWrapper<                                                          \
          OpSchemaDef,                                                             \
          CPUContext,                                                              \
          State,                                                                   \
          true,                                                                    \
          std::tuple<__VA_ARGS__>>)

} // namespace caffe2
