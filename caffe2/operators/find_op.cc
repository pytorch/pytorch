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

#include "caffe2/operators/find_op.h"

namespace caffe2 {

OPERATOR_SCHEMA(Find)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(1)
    .Input(0, "index", "Index (integers)")
    .Input(1, "query", "Needles / query")
    .Output(
        0,
        "query_indices",
        "Indices of the needles in index or 'missing value'")
    .Arg("missing_value", "Placeholder for items that are not found")
    .SetDoc(R"DOC(Finds elements of second input from first input,
                outputting the last (max) index for each query.
                If query not find, inserts missing_value.
                See IndexGet() for a version that modifies the index when
                values are not found.
            )DOC");

REGISTER_CPU_OPERATOR(Find, FindOp<CPUContext>)

} // namespace caffe2
