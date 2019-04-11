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

#include "parallel_lengths_reducer_ops.h"

namespace caffe2 {

namespace intra_op_parallel {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsSum,
    INTRA_OP_PARALLEL,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        0,
        0>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsWeightedSum,
    INTRA_OP_PARALLEL,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        1,
        0>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsMean,
    INTRA_OP_PARALLEL,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        0,
        1>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsPositionalWeightedSum,
    INTRA_OP_PARALLEL,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        1,
        0,
        1>);

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsSum,
    TBB,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        0,
        0>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsWeightedSum,
    TBB,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        1,
        0>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsMean,
    TBB,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        0,
        1>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseLengthsPositionalWeightedSum,
    TBB,
    ParallelSparseLengthsReductionOp<
        float,
        TensorTypes<float, at::Half>,
        1,
        0,
        1>);
#endif

} // namespace intra_op_parallel

} // namespace caffe2
