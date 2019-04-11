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

#include "parallel_segment_reduction_op.h"

namespace caffe2 {

namespace intra_op_parallel {

// registering 5 input gradient with main output
// gradient of SparseLengthsWeightedSum
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    INTRA_OP_PARALLEL,
    ParallelAbstractLengthsWithMainInputGradientOp<
        float,
        int,
        WeightedSumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*SparseFused*/,
        true /*GradientNeedIndices*/>);

// registering 4 input version
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsIndicesInGradientWeightedSumGradient,
    INTRA_OP_PARALLEL,
    ParallelAbstractLengthsGradientOp<
        float,
        int,
        WeightedSumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);

// registering 3 input version
// gradient of SparseLengthsSum
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsIndicesInGradientSumGradient,
    INTRA_OP_PARALLEL,
    ParallelAbstractLengthsGradientOp<
        float,
        int,
        SumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    TBB,
    ParallelAbstractLengthsWithMainInputGradientOp<
        float,
        int,
        WeightedSumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*SparseFused*/,
        true /*GradientNeedIndices*/>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsIndicesInGradientWeightedSumGradient,
    TBB,
    ParallelAbstractLengthsGradientOp<
        float,
        int,
        WeightedSumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsIndicesInGradientSumGradient,
    TBB,
    ParallelAbstractLengthsGradientOp<
        float,
        int,
        SumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);
#endif

} // namespace intra_op_parallel

} // namespace caffe2
