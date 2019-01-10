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
