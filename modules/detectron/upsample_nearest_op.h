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
    auto translate_idx = [](int ii, int d1, int d2, int d3, int scale_factor) {
      int x, y, z, w;
      w = ii % d3;
      ii = ii/d3;
      z = ii % d2;
      ii = ii/d2;
      y = ii % d1;
      ii = ii/d1;
      x = ii;
      w = w/scale_factor;
      z = z/scale_factor;
      d2 /= scale_factor;
      d3 /= scale_factor;
      return (((x*d1+y)*d2)+z)*d3+w;
    };

    auto& X = Input(0);
    auto* Y = Output(0);
    auto out_shape = X.dims().vec();
    out_shape[X.ndim() - 1] *= scale_;
    out_shape[X.ndim() - 2] *= scale_;
    Y->Resize(out_shape);

    int d1;
    int d2;
    int d3;
    if (X.ndim() == 3) {
      d1 = Y->dim32(0);
      d2 = Y->dim32(1);
      d3 = Y->dim32(2);
    } else {
      d1 = Y->dim32(1);
      d2 = Y->dim32(2);
      d3 = Y->dim32(3);
    }

    const T *input_data = X.template data<T>();
    T *output_data = Y->template mutable_data<T>();

    for (int ii = 0; ii < Y->size(); ii++) {
      int ipidx = translate_idx(ii, d1, d2, d3, scale_);
      output_data[ii] = input_data[ipidx];
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
