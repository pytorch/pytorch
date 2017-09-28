/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
