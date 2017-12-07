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
#include "caffe2/operators/assert_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Assert, AssertOp<CPUContext>);

OPERATOR_SCHEMA(Assert)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Assertion op. Takes in a tensor of bools, ints, longs, or long longs and checks
if all values are true when coerced into a boolean. In other words, for non-bool
types this asserts that all values in the tensor are non-zero.
	)DOC")
    .Arg(
        "error_msg",
        "An error message to print when the assert fails.",
        false);

} // namespace caffe2
