#pragma once

#include <ctc.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/common_cudnn.h"

#define CTC_CHECK(condition)           \
  do {                                 \
    ctcStatus_t status = condition;    \
    CAFFE_ENFORCE_EQ(                  \
        status,                        \
        CTC_STATUS_SUCCESS,            \
        " Error at: ",                 \
        __FILE__,                      \
        ":",                           \
        __LINE__,                      \
        ": ",                          \
        ::ctcGetStatusString(status)); \
  } while (0)

namespace caffe2 {

namespace detail {

template <typename Context>
ctcComputeInfo workspaceInfo(const Context& context);

}

template <typename T, typename Context>
class CTCOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CTCOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_test_(
            OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
    CAFFE_ENFORCE(
        (is_test_ && OutputSize() == 2) || (!is_test_ && OutputSize() == 3));
  }

  bool RunOnDevice() override {
    // inputs
    const auto& inputs = Input(INPUTS);
    const auto minibatchSize = inputs.dim(1);
    const auto alphabetSize = inputs.dim(2);
    const auto& labels = OperatorBase::template Input<TensorCPU>(LABELS);
    const auto& labelLengths =
        OperatorBase::template Input<TensorCPU>(LABEL_LENGTHS);
    const auto& inputLengths =
        OperatorBase::template Input<TensorCPU>(INPUT_LENGTHS);

    // outputs
    Tensor<Context>* gradients = nullptr;
    TensorCPU* costs;
    Tensor<Context>* workspace;
    if (!is_test_) {
      // [grads, costs, workspace] to maintain backward compatibility
      gradients = Output(0);
      gradients->ResizeLike(inputs);
      costs = OperatorBase::template Output<TensorCPU>(1);
      costs->ResizeLike(labelLengths);
      workspace = Output(2);
    } else {
      // [costs, workspace]
      costs = OperatorBase::template Output<TensorCPU>(0);
      costs->ResizeLike(labelLengths);
      workspace = Output(1);
    }

    size_t workspaceSizeBytes;
    CTC_CHECK(get_workspace_size(
        labelLengths.template data<int>(),
        inputLengths.template data<int>(),
        alphabetSize,
        minibatchSize,
        detail::workspaceInfo(context_),
        &workspaceSizeBytes));
    workspace->Resize(workspaceSizeBytes);
    CTC_CHECK(compute_ctc_loss(
        inputs.template data<T>(),
        gradients ? gradients->template mutable_data<T>() : nullptr,
        labels.template data<int>(),
        labelLengths.template data<int>(),
        inputLengths.template data<int>(),
        alphabetSize,
        minibatchSize,
        costs->template mutable_data<T>(),
        workspace->template mutable_data<uint8_t>(),
        detail::workspaceInfo(context_)));
    return true;
  }

private:
 bool is_test_;

 INPUT_TAGS(INPUTS, LABELS, LABEL_LENGTHS, INPUT_LENGTHS);
};
}

#undef CTC_CHECK
