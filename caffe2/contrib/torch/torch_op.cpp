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

#include "torch_op.h"

namespace caffe2 {

namespace torch {

const char* TyTraits<CPUContext>::moduleTy = "float";
const char* TyTraits<CPUContext>::tensorTy = "torch.FloatTensor";
const char* TyTraits<CPUContext>::prelude = R"(
        require 'torch'
        require 'nn'
)";
}

struct GetTorchGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    std::vector<std::string> gradientInputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientInputs.push_back(I(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(GO(i));
    }
    std::vector<std::string> gradientOutputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientOutputs.push_back(GI(i));
    }

    return SingleGradientDef(
        "TorchGradient", "", gradientInputs, gradientOutputs);
  }
};


REGISTER_CPU_OPERATOR(Torch, TorchOp<CPUContext>);
REGISTER_CPU_OPERATOR(TorchInit, TorchInitOp<CPUContext>);
REGISTER_CPU_OPERATOR(TorchGradient, TorchGradientOp<CPUContext>);
REGISTER_GRADIENT(Torch, GetTorchGradient);
OPERATOR_SCHEMA(Torch).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(TorchInit);
OPERATOR_SCHEMA(TorchGradient).AllowInplace([](int, int) { return true; });
}
