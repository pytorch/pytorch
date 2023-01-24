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

#include "caffe2/experiments/operators/fully_connected_op_sparse.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC_Sparse, FullyConnectedOp_SPARSE<float, CPUContext>);

OPERATOR_SCHEMA(FC_Sparse).NumInputs(5).NumOutputs(1);
} // namespace
} // namespace caffe2
