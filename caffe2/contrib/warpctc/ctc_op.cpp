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

#include "ctc_op.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace detail {
template <>
ctcComputeInfo workspaceInfo<CPUContext>(const CPUContext& /*context*/) {
  ctcComputeInfo result;
  result.loc = CTC_CPU;
  result.num_threads = 1;
  return result;
}
}

REGISTER_CPU_OPERATOR(CTC, CTCOp<float, CPUContext>);
OPERATOR_SCHEMA(CTC)
    .NumInputs(4)
    .NumOutputs(3);
//    .EnforceInputOutputGradient({{0, 0}});

namespace {
class GetCTCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Copy", "", vector<string>{O(0)}, vector<string>{GI(0)});
  }
};
}
REGISTER_GRADIENT(CTC, GetCTCGradient);
}
