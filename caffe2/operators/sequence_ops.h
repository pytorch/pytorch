#ifndef CAFFE2_OPERATORS_SEQUENCE_OPS_H_
#define CAFFE2_OPERATORS_SEQUENCE_OPS_H_

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <class Context>
class GatherPaddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GatherPaddingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        startPaddingWidth_(
            OperatorBase::GetSingleArgument<int>("padding_width", 1)),
        endPaddingWidth_(
            OperatorBase::GetSingleArgument<int>("end_padding_width", -1)) {
    CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
    if (endPaddingWidth_ < 0) {
      endPaddingWidth_ = startPaddingWidth_;
    }
  }

  bool RunOnDevice() override {
    if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
      Output(0)->Resize(std::vector<TIndex>(0));
      if (OutputSize() == 2) {
        Output(1)->Resize(std::vector<TIndex>(0));
      }
      return true;
    }
    return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  int startPaddingWidth_;
  int endPaddingWidth_;
};

template <class Context>
class RemovePaddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RemovePaddingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        startPaddingWidth_(
            OperatorBase::GetSingleArgument<int>("padding_width", 1)),
        endPaddingWidth_(
            OperatorBase::GetSingleArgument<int>("end_padding_width", -1)) {
    CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
    if (endPaddingWidth_ < 0) {
      endPaddingWidth_ = startPaddingWidth_;
    }
  }

  bool RunOnDevice() override {
    if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
      Output(0)->CopyFrom(Input(0), &context_);
      if (OutputSize() == 2) {
        Output(1)->CopyFrom(Input(1), &context_);
      }
      return true;
    }
    return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  int startPaddingWidth_;
  int endPaddingWidth_;
};

template <class Context>
class AddPaddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AddPaddingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        startPaddingWidth_(
            OperatorBase::GetSingleArgument<int>("padding_width", 1)),
        endPaddingWidth_(
            OperatorBase::GetSingleArgument<int>("end_padding_width", -1)) {
    CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
    if (endPaddingWidth_ < 0) {
      endPaddingWidth_ = startPaddingWidth_;
    }
  }

  bool RunOnDevice() override {
    if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
      Output(0)->CopyFrom(Input(0), &context_);
      if (OutputSize() == 2) {
        Output(1)->CopyFrom(Input(1), &context_);
      }
      return true;
    }
    return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  int startPaddingWidth_;
  int endPaddingWidth_;
};

template <class Context>
class PadEmptySamplesOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  PadEmptySamplesOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SEQUENCE_OPS_H_
