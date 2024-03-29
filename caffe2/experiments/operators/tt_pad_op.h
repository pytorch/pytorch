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

#ifndef CAFFE2_OPERATORS_TT_PAD_OP_H_
#define CAFFE2_OPERATORS_TT_PAD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class TTPadOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTPadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<int64_t>("scale", 0)) {
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("scale"), "Argument `scale` is missing.");
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* X_pad = Output(0);
    CAFFE_ENFORCE(&X == X_pad);

    CAFFE_ENFORCE(X.dim() == 2, X.dim());

    auto X_dim0 = X.size(0);
    auto X_dim1 = X.size(1);

    auto* X_orig_dim0 = Output(1, {1}, at::dtype<int64_t>());
    *X_orig_dim0->template mutable_data<int64_t>() = X_dim0;

    if (X_dim0 % scale_ != 0) {
      int64_t padded_dim0 = (X_dim0 / scale_ + 1) * scale_;
      auto dim0_diff = padded_dim0 - X_dim0;
      // set growthPct to the upper bound percentage: (100 * scale_ / X_dim0)
      X_pad->Extend(dim0_diff, 100 * scale_ / X_dim0);

      auto* X_pad_data = X_pad->template mutable_data<T>();
      int64_t X_size = X_dim0 * X_dim1;
      memset(X_pad_data + X_size, 0, dim0_diff * X_dim1 * sizeof(T));
    }

    return true;
  }

 protected:
  int64_t scale_;
};

template <typename T, class Context, class Engine = DefaultEngine>
class TTPadGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTPadGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& G = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE(&G == output);

    auto old_dim0 = *Input(1).template data<int64_t>();
    auto new_dim0 = G.size(0);

    if (old_dim0 < new_dim0) {
      output->ShrinkTo(old_dim0);
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TT_PAD_OP_H_
