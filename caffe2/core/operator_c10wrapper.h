#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include "caffe2/core/operator.h"
#include <c10/util/ArrayRef.h>
#include <c10/util/Metaprogramming.h>
#include <ATen/core/ivalue.h>

namespace caffe2 {

/**
 * To make a c10 operator "C10Add" callable from caffe2 as "C2MyAddOpName", just
 * write
 *
 *     REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(C10Add, C2MyAddOpName)
 *
 */

namespace detail {
template <class Context>
class C10OperatorWrapper final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  C10OperatorWrapper(const c10::OperatorHandle& op, const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        op_(op),
        kernel_(at::nullopt),
        has_preallocated_outputs_(op_.schema().arguments().back().name() == detail::PREALLOCATED_OUTPUT_ARGNAME) {
    AT_ASSERT(!has_preallocated_outputs_ || op_.schema().arguments().back().type()->isSubtypeOf(OptionalType::create(ListType::ofTensors())));

    AT_ASSERT(operator_def.output_size() == op_.schema().returns().size());
    AT_ASSERT(operator_def.input_size() + (has_preallocated_outputs_ ? 1 : 0) <= op_.schema().arguments().size()); // '<=' because there might be caffe2 arguments
  }

  bool RunOnDevice() override {
    // due to caching the stack_, concurrent calling is not allowed.
    // TODO thread_local might fix this
    std::lock_guard<std::mutex> lock(mutex_);

    pushInputs_();
    callKernel_();
    popOutputs_();

    return true;
  }

 private:
  void pushInputs_() {
    AT_ASSERT(stack_.size() == 0);
    stack_.reserve(op_.schema().arguments().size() + (has_preallocated_outputs_ ? 1 : 0));

    size_t input_tensor_index = 0;

    for (const auto& argument : op_.schema().arguments()) {
      if (argument.name() == detail::PREALLOCATED_OUTPUT_ARGNAME) {
        // note: if detail::PREALLOCATED_OUTPUT_ARGNAME was at the end of the argument list,
        // then has_preallocated_outputs_ would be true.
        AT_ASSERTM(has_preallocated_outputs_, "Error in caffe2->c10 wrapper: Operator schema has a parameter named ", detail::PREALLOCATED_OUTPUT_ARGNAME, ", but it's not at the end of the argument list");

        AT_ASSERTM(argument.type()->isSubtypeOf(OptionalType::create(ListType::ofTensors())), "Error in caffe2->c10 wrapper: Operator schema has a parameter named ", detail::PREALLOCATED_OUTPUT_ARGNAME, ", but it's not of type TensorList?");
        stack_.emplace_back(preallocated_outputs_());

      } else if (argument.type()->isSubtypeOf(TensorType::get())) {
        AT_ASSERTM(input_tensor_index < InputSize(), "Error in caffe2->c10 wrapper: Too few tensor arguments given (", InputSize(), "), operator schema expected more.");
        stack_.emplace_back(at::Tensor(Input(input_tensor_index++)));

      } else if (argument.type()->isSubtypeOf(ListType::ofTensors())) {
        AT_ASSERTM(input_tensor_index == 0, "Error in caffe2->c10 wrapper: Schema can only have either one or more Tensor inputs or one TensorList input.");
        stack_.emplace_back(ivalue::TensorList::create(array_inputs_()));
        input_tensor_index = InputSize();

      } else {
        stack_.emplace_back(get_nontensor_argument_(argument));
      }

      AT_ASSERTM(input_tensor_index == InputSize(), "Error in caffe2->c10 wrapper: Number of caffe2 operator inputs (", InputSize(), ") doesn't match number of tensor arguments (", input_tensor_index, ") in the c10 operator schema.");
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
      result.emplace_back(Input(i));
    }
    return result;
  }

  std::vector<at::Tensor> preallocated_outputs_() {
    std::vector<at::Tensor> result;
    result.reserve(OutputSize());
    for (size_t i = 0; i < OutputSize(); ++i) {
      result.emplace_back(OperatorBase::OutputTensorOrUndefined(i));
    }
    return result;
  }

  IValue get_nontensor_argument_(const c10::Argument& argument) {
    if (argument.type()->isSubtypeOf(IntType::get())) {
      if (argument.default_value().has_value()) {
        return OperatorBase::GetSingleArgument<int>(argument.name(), argument.default_value()->toInt());
      } else {
        AT_CHECK(OperatorBase::HasSingleArgumentOfType<int>(argument.name()), "Error in caffe2->c10 wrapper: Expected argument '", argument.name(), "' missing or wrong type (expected int).");
        return OperatorBase::GetSingleArgument<int>(argument.name(), 0);
      }
    } else if (argument.type()->isSubtypeOf(FloatType::get())) {
      if (argument.default_value().has_value()) {
        return OperatorBase::GetSingleArgument<double>(argument.name(), argument.default_value()->toDouble());
      } else {
        AT_CHECK(OperatorBase::HasSingleArgumentOfType<double>(argument.name()), "Error in caffe2->c10 wrapper: Expected argument '", argument.name(), "' missing or wrong type (expected double).");
        return OperatorBase::GetSingleArgument<double>(argument.name(), 0);
      }
    } else if (argument.type()->isSubtypeOf(BoolType::get())) {
      if (argument.default_value().has_value()) {
        return OperatorBase::GetSingleArgument<bool>(argument.name(), argument.default_value()->toBool());
      } else {
        AT_CHECK(OperatorBase::HasSingleArgumentOfType<bool>(argument.name()), "Error in caffe2->c10 wrapper: Expected argument '", argument.name(), "' missing or wrong type (expected bool).");
        return OperatorBase::GetSingleArgument<bool>(argument.name(), 0);
      }
    } else {
      // TODO Support more types
      AT_ERROR("Error in caffe2->c10 wrapper: Unsupported argument type ", argument.type()->str(), " in c10 operator schema");
    }
  }

  c10::OperatorHandle op_;
  c10::optional<OpKernel> kernel_;
  bool has_preallocated_outputs_;

  // this is stored as a member here to avoid having to re-allocate a stack
  // for each call. Between kernel calls, stack_.size() == 0, but capacity
  // should not need to be grown anymore after the first call.
  std::vector<IValue> stack_;
  std::mutex mutex_;
};

template<class Context, const c10::OperatorHandle& (*OperatorHandle)()>
inline std::unique_ptr<C10OperatorWrapper<Context>> createC10OperatorWrapper(const OperatorDef& operator_def, Workspace* ws) {
  return c10::guts::make_unique<C10OperatorWrapper<Context>>(OperatorHandle(), operator_def, ws);
}

}

// TODO Also register c10 operators on mobile
#ifndef C10_MOBILE
// TODO Currently we only register the CPU variant. This is going to be fixed
//      once the tensor detemplatization lands.
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(OperatorHandle, Name)        \
  REGISTER_CPU_OPERATOR_CREATOR(                                                   \
      Name,                                                                        \
      detail::createC10OperatorWrapper<CPUContext, OperatorHandle>                 \
  )
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CUDA(OperatorHandle, Name)       \
  REGISTER_CUDA_OPERATOR_CREATOR(                                                  \
      Name,                                                                        \
      detail::createC10OperatorWrapper<CUDAContext, OperatorHandle>                \
  )
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_HIP(OperatorHandle, Name)        \
  REGISTER_HIP_OPERATOR_CREATOR(                                                   \
      Name,                                                                        \
      detail::createC10OperatorWrapper<HIPContext, OperatorHandle>                 \
  )
#else
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(OperatorHandle, Name)
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CUDA(OperatorHandle, Name)
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_HIP(OperatorHandle, Name)
#endif
} // namespace caffe2
