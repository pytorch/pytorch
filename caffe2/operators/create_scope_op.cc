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

#include "caffe2/operators/create_scope_op.h"

CAFFE2_DEFINE_bool(
    caffe2_workspace_stack_debug,
    false,
    "Enable debug checks for CreateScope's workspace stack");

namespace caffe2 {
CAFFE_KNOWN_TYPE(detail::WorkspaceStack);

template <>
bool CreateScopeOp<CPUContext>::RunOnDevice() {
  auto* ws_stack = OperatorBase::Output<detail::WorkspaceStack>(0);
  ws_stack->clear();
  return true;
}

REGISTER_CPU_OPERATOR(CreateScope, CreateScopeOp<CPUContext>);

SHOULD_NOT_DO_GRADIENT(CreateScope);

OPERATOR_SCHEMA(CreateScope).NumInputs(0).NumOutputs(1).SetDoc(R"DOC(
'CreateScope' operator initializes and outputs empty scope that is used
by Do operator to store local blobs
    )DOC");

} // namespace caffe2
