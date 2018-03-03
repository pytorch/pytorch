/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace detail {

template <typename T, typename Context>
void VariableLengthSequencePadding(
    int N,
    int B,
    int M,
    T* X,
    const int32_t* seqLengths,
    const T padValue,
    Context* /*context*/) {
  for (int j = 0; j < B; j++) {
    for (int i = seqLengths[j]; i < N; i++) {
      EigenVectorArrayMap<T>(X + B * M * i + M * j, M).setConstant(padValue);
    }
  }
}

} // namespace detail

template <typename T, typename Context>
class VariableLengthSequencePaddingOp : public Operator<Context> {
 public:
  VariableLengthSequencePaddingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws)
      {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    const auto N = Input(INPUT).dim(0);
    const auto B = Input(INPUT).dim(1);
    const auto M = Input(INPUT).dim(2);

    auto X = Output(OUTPUT)->template mutable_data<T>();

    auto seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();

    detail::VariableLengthSequencePadding<T, Context>(
      N, B, M, X, seqLengths, 0, &context_);
    return true;
  }

 protected:
  INPUT_TAGS(INPUT, SEQ_LENGTHS);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace caffe2
