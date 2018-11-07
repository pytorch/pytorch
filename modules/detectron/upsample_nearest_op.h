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

#ifndef UPSAMPLE_NEAREST_OP_H_
#define UPSAMPLE_NEAREST_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class UpsampleNearestOp final : public Operator<Context> {
 public:
  UpsampleNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(this->template GetSingleArgument<int>("scale", 2)) {
    DCHECK_GE(scale_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    auto out_shape = X.sizes().vec();
    out_shape[X.dim() - 1] *= scale_;
    out_shape[X.dim() - 2] *= scale_;
    Y->Resize(out_shape);

    int d1;
    int d2;
    int d3;
    if (X.dim() == 3) {
      d1 = Y->dim32(0);
      d2 = Y->dim32(1);
      d3 = Y->dim32(2);
    } else {
      d1 = Y->dim32(0) * Y->dim32(1);
      d2 = Y->dim32(2);
      d3 = Y->dim32(3);
    }

    const T *input_data = X.template data<T>();
    T *output_data = Y->template mutable_data<T>();
    int scaled_d2 = d2 / scale_;
    int scaled_d3 = d3 / scale_;

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd
#else
#pragma omp parallel for
#endif
#endif
    for (int i = 0; i < d1; ++i) {
      for (int j = 0; j < d2; ++j) {
        for (int u = 0; u < d3; ++u) {
          int ii = (i * d2 + j) * d3 + u;
          int scaled_u = u / scale_;
          int scaled_j = j / scale_;
          int ipidx = ((i * scaled_d2) + scaled_j) * scaled_d3 + scaled_u;
          output_data[ii] = input_data[ipidx];
        }
      }
    }

    return true;
  }

 protected:
  int scale_;
};

template <typename T, class Context>
class UpsampleNearestGradientOp final : public Operator<Context> {
 public:
  UpsampleNearestGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<int>("scale", 2)) {
    DCHECK_GE(scale_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int scale_;
};

} // namespace caffe2

#endif // UPSAMPLE_NEAREST_OP_H_
