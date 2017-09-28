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

#include "caffe2/operators/free_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Free, FreeOp<CPUContext>);
SHOULD_NOT_DO_GRADIENT(Free);

OPERATOR_SCHEMA(Free)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SameNumberOfOutput()
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(
Frees the content of the blobs. The input and output blobs should be
one-to-one inplace.)DOC");
} // namespace caffe2
