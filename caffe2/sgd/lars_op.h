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

#ifndef CAFFE2_OPERATORS_LARS_OP_H_
#define CAFFE2_OPERATORS_LARS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class LarsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LarsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(OperatorBase::GetSingleArgument<float>("offset", 0.5)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dX = Input(1);
    CAFFE_ENFORCE(
        dX.size() == X.size(), "Gradient size doesn't match parameter size.");
    CAFFE_ENFORCE_GE(offset_, 0);

    auto* lr_rescale = Output(0);
    lr_rescale->Resize(vector<TIndex>{1});

    Compute(
        dX.size(),
        X.template data<T>(),
        dX.template data<T>(),
        offset_,
        lr_rescale->template mutable_data<T>());

    return true;
  }

 private:
  void Compute(
      TIndex N,
      const T* X_data,
      const T* dX_data,
      T offset,
      T* lr_rescale_data);

  T offset_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LARS_OP_H_
