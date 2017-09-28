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

#include "caffe2/mkl/operators/operator_fallback_mkl.h"

#include "caffe2/mkl/utils/mkl_operator.h"
#include "caffe2/operators/cross_entropy_op.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/operators/loss_op.h"
#include "caffe2/operators/softmax_op.h"
#include "caffe2/operators/utility_ops.h"

// can add more non-MKL operators if needed
namespace caffe2 {
REGISTER_MKL_OPERATOR(
    Softmax,
    mkl::MKLFallbackOp<SoftmaxOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    LabelCrossEntropy,
    mkl::MKLFallbackOp<LabelCrossEntropyOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    AveragedLoss,
    mkl::MKLFallbackOp<AveragedLoss<float, CPUContext>>);

// filter operators
REGISTER_MKL_OPERATOR(
    XavierFill,
    mkl::MKLFallbackOp<XavierFillOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    ConstantFill,
    mkl::MKLFallbackOp<ConstantFillOp<CPUContext>>);
REGISTER_MKL_OPERATOR(
    GaussianFill,
    mkl::MKLFallbackOp<GaussianFillOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    MSRAFill,
    mkl::MKLFallbackOp<MSRAFillOp<float, CPUContext>>);

} // namespace caffe2
