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

#pragma once

#include "caffe2/core/operator.h"

#include <stdlib.h>
#include <time.h>

namespace caffe2 {

template <typename T, class Context>
void weight_scale_update(
    int N,
    const T* w,
    const T scale,
    int64_t iter,
    int64_t stepsize,
    int64_t update_upper_bound,
    T* nw,
    Context* context) {
  const auto w_size = N * sizeof(float);
  if (iter % stepsize != 0 || iter >= update_upper_bound) {
    memcpy(nw, w, w_size);
    return;
  }
  // perform the weight scaling
  caffe2::math::Scale<T, T, Context>(N, scale, w, nw, context);
}

template <typename T, class Context>
class WeightScaleOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  WeightScaleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        stepsize_(OperatorBase::GetSingleArgument<int64_t>(
            "stepsize",
            std::numeric_limits<int64_t>::max())),
        update_upper_bound_(OperatorBase::GetSingleArgument<int64_t>(
            "upper_bound_iter",
            std::numeric_limits<int64_t>::max())),
        scale_(this->template GetSingleArgument<T>("scale", 1.0f)) {}

  bool RunOnDevice() override {
    Output(OUTPUT_WEIGHTS)->ResizeLike(Input(WEIGHTS));

    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0] + 1;

    weight_scale_update<T, Context>(
        Input(WEIGHTS).size(),
        Input(WEIGHTS).template data<T>(),
        scale_,
        iter,
        stepsize_,
        update_upper_bound_,
        Output(OUTPUT_WEIGHTS)->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  int64_t stepsize_;
  int64_t update_upper_bound_;
  T scale_;
  INPUT_TAGS(WEIGHTS, ITER);
  OUTPUT_TAGS(OUTPUT_WEIGHTS);
};
} // namespace caffe2
