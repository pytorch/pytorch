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

#include "caffe2/operators/lengths_reducer_ops.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Use _STR option because the schema is declared using _STR version too in
// generic fashion. Otherwise it'd break schema declaration check.
// TODO(dzhulgakov): remove _STR when all lengths ops are off generic version.

REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 0>);
REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsWeightedSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 1, 0>);
REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsMean",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 1>);

} // namespace caffe2
