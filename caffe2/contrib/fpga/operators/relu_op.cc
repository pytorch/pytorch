#include "caffe2/operators/relu_op.h"

#include "../common_fpga.h"
#include "../context.h"
#include "../context_intel_fpga.h"
#include "../operator.h"

namespace caffe2 {

namespace {

template <typename T>
class FPGAReluOp final : public Operator<OpenCLContext> {
 public:
  USE_OPERATOR_FUNCTIONS(OpenCLContext);

  FPGAReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDevice() {
    FPGAContextSingleton ctx = *static_cast<FPGAContextSingleton*>(
        context_.GetSingleton(FPGAEngine::name));
    const auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE_GE(X.numel(), 0);
    Y->ResizeLike(X);
    // TODO: fix this to work with all dimensions
    CAFFE_ENFORCE_EQ(X.dim(), 2);
    ctx.ReLU(
        X.template data<T>(),
        Y->template mutable_data<T>(),
        X.size(0),
        X.size(1));
    return true;
  }
};

template <typename T>
class FPGAReluGradientOp final : public Operator<OpenCLContext> {
 public:
  USE_OPERATOR_FUNCTIONS(OpenCLContext);

  FPGAReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    FPGAContextSingleton ctx = *static_cast<FPGAContextSingleton*>(
        context_.GetSingleton(FPGAEngine::name));
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
    CAFFE_ENFORCE_GE(Y.numel(), 0);
    CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
    dX->ResizeLike(Y);

    // TODO: fix this to work with all dimensions
    CAFFE_ENFORCE_EQ(Y.dim(), 2);

    ctx.ReLUGrad(
        Y.template data<float>(),
        dY.template data<float>(),
        dX->template mutable_data<float>(),
        Y.size(0),
        Y.size(1));
    return true;
  }
};

} // namespace

REGISTER_OPENCL_OPERATOR_WITH_ENGINE(Relu, FPGA, FPGAReluOp<float>);
REGISTER_OPENCL_OPERATOR_WITH_ENGINE(
    ReluGradient,
    FPGA,
    FPGAReluGradientOp<float>);

} // namespace caffe2
