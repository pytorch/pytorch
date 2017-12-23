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

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

void SoftmaxCPU(
    CPUContext& context,
    const int N,
    const int D,
    const float* Xdata,
    float* Ydata,
    float* scale,
    const float* sum_multiplier,
    bool logarithmic,
    float* rowmax) {
  math::RowwiseMax<float, CPUContext>(N, D, Xdata, rowmax, &context);
  // Put the intermediate result X - max(X) into Y
  context.template Copy<float, CPUContext, CPUContext>(N * D, Xdata, Ydata);
  // Subtract the max (for numerical reasons)
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasNoTrans,
      N,
      D,
      1,
      -1,
      rowmax,
      sum_multiplier,
      1,
      Ydata,
      &context);
  // Exponentiation
  math::Exp<float, CPUContext>(N * D, Ydata, Ydata, &context);
  math::Gemv<float, CPUContext>(
      CblasNoTrans, N, D, 1, Ydata, sum_multiplier, 0, scale, &context);
  // Do division
  // TODO(Yangqing): maybe implement it more beautifully?
  if (!logarithmic) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        Ydata[i * D + j] /= scale[i];
      }
    }
  } else {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        Ydata[i * D + j] =
            Xdata[i * D + j] - rowmax[i] - log(fmaxf(scale[i], 1e-20));
      }
    }
  }
}

} // namespace caffe2
