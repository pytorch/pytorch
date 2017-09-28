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

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace {

class GetAllBlobNamesOp final : public Operator<CPUContext> {
 public:
  GetAllBlobNamesOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        include_shared_(GetSingleArgument<int>("include_shared", true)),
        ws_(ws) {}

  bool RunOnDevice() override {
    auto* out = Output(0);
    const auto& blobs = include_shared_ ? ws_->Blobs() : ws_->LocalBlobs();
    out->Resize(blobs.size());
    std::copy(blobs.begin(), blobs.end(), out->mutable_data<std::string>());
    return true;
  }

 private:
  bool include_shared_;
  Workspace* ws_;
};

REGISTER_CPU_OPERATOR(GetAllBlobNames, GetAllBlobNamesOp);
OPERATOR_SCHEMA(GetAllBlobNames)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Return a 1D tensor of strings containing the names
of each blob in the active workspace.
)DOC")
    .Arg(
        "include_shared",
        "(bool, default true) Whether to include blobs "
        "inherited from parent workspaces.")
    .Output(0, "blob_names", "1D tensor of strings containing blob names.");
SHOULD_NOT_DO_GRADIENT(GetAllBlobNamesOp);
}
}
