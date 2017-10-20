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

#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorBoolFill, GivenTensorFillOp<bool, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorStringFill,
    GivenTensorFillOp<std::string, CPUContext>);

NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(GivenTensorDoubleFill);
NO_GRADIENT(GivenTensorBoolFill);
NO_GRADIENT(GivenTensorIntFill);
NO_GRADIENT(GivenTensorInt64Fill);
NO_GRADIENT(GivenTensorStringFill);

OPERATOR_SCHEMA(GivenTensorFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>);
OPERATOR_SCHEMA(GivenTensorDoubleFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(
        FillerTensorInference<TensorProto_DataType_DOUBLE>);
OPERATOR_SCHEMA(GivenTensorBoolFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_BOOL>);
OPERATOR_SCHEMA(GivenTensorIntFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT32>);
OPERATOR_SCHEMA(GivenTensorInt64Fill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT64>);
OPERATOR_SCHEMA(GivenTensorStringFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(
        FillerTensorInference<TensorProto_DataType_STRING>);

} // namespace caffe2
