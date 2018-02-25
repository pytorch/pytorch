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

#include "caffe2/operators/if_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(If, IfOp<CPUContext>);

OPERATOR_SCHEMA(If)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
'If' control operator, first input is a scalar boolean blob that stores condition
value. Accepts 'then_net' (required) and 'else_net' (optional) arguments for 'then' and
'else' subnets respectively. Subnets are executed in the same workspace as 'If'.
    )DOC")
    .Arg("then_net", "Net executed when condition is true")
    .Arg("else_net", "Net executed when condition is false (optional)")
    .Input(0, "condition", "Scalar boolean condition")
    .AllowInplace([](int in, int out) -> bool { return true; });

} // namespace caffe2
