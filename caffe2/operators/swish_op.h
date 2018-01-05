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

#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <class Context>
class SwishGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SwishGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
  }

 protected:
  INPUT_TAGS(X, Y, DY);
  OUTPUT_TAGS(DX);
};

class GetSwishGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SwishGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

} // namespace caffe2
