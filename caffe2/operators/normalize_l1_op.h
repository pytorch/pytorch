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

#ifndef CAFFE2_OPERATORS_NORMALIZE_L1_OP_H_
#define CAFFE2_OPERATORS_NORMALIZE_L1_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class NormalizeL1Op final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(NormalizeL1Op)

  bool RunOnDevice() override {
    const auto& x = Input(0);
    auto* y = Output(0);
    const auto* xData = x.template data<T>();
    y->ResizeLike(x);
    auto* yData = y->template mutable_data<T>();

    const auto canonical_axis = x.canonical_axis_index(
        OperatorBase::GetSingleArgument<int>("axis", -1));
    const int m = x.dim32(canonical_axis);
    const int n = x.size() / m;
    const int sf = x.size_from_dim(canonical_axis + 1);
    DoNormalize(xData, yData, m, n, sf);
    return true;
  }

 private:
  void
  DoNormalize(const T* xData, T* yData, const int m, const int n, const int sf);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NORMALIZE_L1_OP_H_
