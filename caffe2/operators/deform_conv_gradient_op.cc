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

#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/deform_conv_op.h"
#include "caffe2/operators/deform_conv_op_impl.h"

namespace caffe2 {

OPERATOR_SCHEMA(DeformConvGradient).NumInputs(4, 4).NumOutputs(2, 4);

namespace {

class GetDeformConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 4);

    ArgumentHelper argsHelper(def_);

    auto compute_dX =
        !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

    if (def_.input_size() == 4) {
      if (compute_dX) {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2), GI(3), GI(0)});
      } else {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2), GI(3)});
      }
    } else {
      if (compute_dX) {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2), GI(0)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      } else {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      }
    }
  }
};
REGISTER_GRADIENT(DeformConv, GetDeformConvGradient);

} // namespace
} // namespace caffe2
