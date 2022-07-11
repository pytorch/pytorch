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

#include <functional>

#include "caffe2/contrib/fakelowp/fp16_fc_acc_op.h"
#include "caffe2/core/init.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/fc_inference.h"

namespace caffe2 {

template <>
int Fp16FCAccOp<CPUContext, DefaultEngine, false>::runs = 0;
template <>
float Fp16FCAccOp<CPUContext, DefaultEngine, false>::total_error = 0.0;
template <>
float Fp16FCAccOp<CPUContext, DefaultEngine, false>::total_error_with_bias =
    0.0;

template <>
int Fp16FCAccOp<CPUContext, DefaultEngine, true>::runs = 0;
template <>
float Fp16FCAccOp<CPUContext, DefaultEngine, true>::total_error = 0.0;
template <>
float Fp16FCAccOp<CPUContext, DefaultEngine, true>::total_error_with_bias = 0.0;

REGISTER_CPU_OPERATOR(
    Fp16FCAcc32,
    Fp16FCAccOp<
        CPUContext,
        DefaultEngine,
        false /* USE_ACC_FP16 */,
        true /* USE_TMP_ACCUMULATOR */,
        false /* ADD_BIAS_FIRST */>);

using namespace std::placeholders;

OPERATOR_SCHEMA(Fp16FCAcc32)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        std::bind(CostInferenceForFC, _1, _2, false)))
    .SetDoc(R"DOC(Same as FC)DOC");

REGISTER_CPU_OPERATOR(
    Fp16FCAcc16,
    Fp16FCAccOp<
        CPUContext,
        DefaultEngine,
        true /* USE_ACC_FP16 */,
        true /* USE_TMP_ACCUMULATOR */,
        false /* ADD_BIAS_FIRST */>);

OPERATOR_SCHEMA(Fp16FCAcc16)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        std::bind(CostInferenceForFC, _1, _2, false)))
    .SetDoc(R"DOC(Same as FC)DOC");

REGISTER_CPU_OPERATOR(
    Fp16FCAcc32NNPI,
    Fp16FCAccOp<
        CPUContext,
        DefaultEngine,
        false /* USE_ACC_FP16 */,
        false /* USE_TMP_ACCUMULATOR */,
        true /* ADD_BIAS_FIRST */>);

OPERATOR_SCHEMA(Fp16FCAcc32NNPI)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        std::bind(CostInferenceForFC, _1, _2, false)))
    .SetDoc(R"DOC(Same as FC)DOC");

REGISTER_CPU_OPERATOR(
    Fp16FCAcc16NNPI,
    Fp16FCAccOp<
        CPUContext,
        DefaultEngine,
        true /* USE_ACC_FP16 */,
        false /* USE_TMP_ACCUMULATOR */,
        true /* ADD_BIAS_FIRST */>);

OPERATOR_SCHEMA(Fp16FCAcc16NNPI)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        std::bind(CostInferenceForFC, _1, _2, false)))
    .SetDoc(R"DOC(Same as FC)DOC");
} // namespace caffe2
