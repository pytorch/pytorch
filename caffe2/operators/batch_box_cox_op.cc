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

#include "caffe2/operators/batch_box_cox_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool BatchBoxCoxOp<CPUContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& lambda1 = Input(LAMBDA1);
  auto& lambda2 = Input(LAMBDA2);
  CAFFE_ENFORCE_GE(data.ndim(), 1);
  auto N = data.dim(0);
  auto D = data.size_from_dim(1);

  auto* output = Output(0);
  output->ResizeLike(Input(DATA));
  auto* output_ptr = output->template mutable_data<T>();

  if (data.size() <= 0) {
    return true;
  }

  CAFFE_ENFORCE_EQ(lambda1.size(), D);
  CAFFE_ENFORCE_EQ(lambda2.size(), D);

  const auto* data_ptr = data.template data<T>();
  const auto* lambda1_ptr = lambda1.template data<T>();
  const auto* lambda2_ptr = lambda2.template data<T>();

  // eigen is column-major
  auto data_m = ConstEigenMatrixMap<T>(data_ptr, D, N);
  auto output_m = EigenMatrixMap<T>(output_ptr, D, N);

  const T k_eps = 1e-6;
  for (TIndex i = 0; i < D; i++) {
    T lambda1_v = lambda1_ptr[i];
    T lambda2_v = lambda2_ptr[i];

    output_m.row(i).array() = (data_m.row(i).array() + lambda2_v).max(k_eps);

    if (lambda1_v == 0) {
      output_m.row(i).array() = output_m.row(i).array().log();
    } else {
      output_m.row(i).array() =
          (output_m.row(i).array().pow(lambda1_v) - 1) / lambda1_v;
    }
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(BatchBoxCox, BatchBoxCoxOp<CPUContext>);
OPERATOR_SCHEMA(BatchBoxCox)
    .NumInputs(3)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Input `data` is a N * D matrix. Apply box-cox transform for each column.
`lambda1` and `lambda2` is of size D that defines the hyper-parameters for
the transform of each column `x` of the input `data`:

    ln(x + lambda2), if lambda1 == 0
    ((x + lambda2)^lambda1 - 1)/lambda1, if lambda1 != 0

)DOC")
    .Input(0, "data", "input float or double N * D matrix")
    .Input(1, "lambda1", "tensor of size D with the same type as data")
    .Input(2, "lambda2", "tensor of size D with the same type as data")
    .Output(0, "output", "output matrix that applied box-cox transform");

GRADIENT_NOT_IMPLEMENTED_YET(BatchBoxCox);
}
}
