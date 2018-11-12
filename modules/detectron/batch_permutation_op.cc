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

#include "batch_permutation_op.h"
#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {

#ifdef CAFFE2_USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    BatchPermutation,
    IDEEPFallbackOp<BatchPermutationOp<float, CPUContext>>);
#endif

REGISTER_CPU_OPERATOR(BatchPermutation, BatchPermutationOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    BatchPermutationGradient,
    BatchPermutationGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(BatchPermutation)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Permute the batch elements of the input tensor X according to the permutation
specified in the input indices.

Warning: this op does not verify that indices is a valid permutation; gradient
comptuation is only correct if indices is a permutation.
)DOC")
    .Input(
        0,
        "X",
        "Tensor of at least 1D shape (N, D0, D1, ...).")
    .Input(
        1,
        "indices",
        "1D tensor of type int with shape (N, ) specifying a valid permutation "
        "of the indices in [0, N - 1] (inclusive).")
    .Output(
        0,
        "Y",
        "Tensor with the same shape as X where the (D0, D1, ...) dimensional "
        "batch elements of X are permuted according to the input indices.");

OPERATOR_SCHEMA(BatchPermutationGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(
        0,
        "indices",
        "See BatchPermutation.")
    .Input(
        1,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

template <>
bool BatchPermutationOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& indices = Input(1);
  auto* Y = Output(0);

  CAFFE_ENFORCE_EQ(indices.dim(), 1, "indices must be 1-d");
  CAFFE_ENFORCE_EQ(
    X.dim32(0), indices.dim32(0),
    "X.dim32(0) must be equal to indices.dim32(0)",
    "(",
    X.dim32(0),
    " vs. ",
    indices.dim32(0),
    ")");

  Y->ResizeLike(X);

  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  const float *src = X.template data<float>();
  float *dst = Y->template mutable_data<float>();

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd
#else
#pragma omp parallel for
#endif 
#endif  
  for (int i = 0; i < N; i++) {
    int idx = indices.template data<int>()[i];

    std::memcpy(dst + i * C * H * W, src + idx * C * H * W, sizeof(float) * C * H * W);
  }

  return true;
}

class GetBatchPermutationGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BatchPermutationGradient",
        "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(BatchPermutation, GetBatchPermutationGradient);

} // namespace caffe2
