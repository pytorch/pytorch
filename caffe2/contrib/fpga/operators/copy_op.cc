#include "caffe2/core/operator.h"
#include "../common_fpga.h"
#include "../context.h"
#include "../operator.h"

namespace caffe2 {
namespace {

class CopyToOpenCLOp final : public Operator<OpenCLContext> {
 public:
  CopyToOpenCLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDeviceWithEngine(const std::string& engine) {
    const auto& X = InputBlob(0).Get<Tensor>();
    auto* Y = Output(0);
    Y->Resize(X.sizes());

    if (engine == "FPGA") {
      context_.Copy<float, bfloat16, CPUContext, OpenCLContext>(X, *Y);
    } else {
      context_.Copy<float, float, CPUContext, OpenCLContext>(X, *Y);
    }
    return true;
  }

  bool RunOnDevice() override {
    return RunOnDeviceWithEngine(engine());
  }
};

REGISTER_OPENCL_OPERATOR_WITH_ENGINE(CopyToOpenCL, FPGA, CopyToOpenCLOp);
OPERATOR_SCHEMA(CopyToOpenCL).NumInputs(1).NumOutputs(1);

class CopyFromOpenCLOp final : public Operator<OpenCLContext> {
 public:
  CopyFromOpenCLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDeviceWithEngine(const std::string& engine) {
    const auto& X = Input(0);
    auto* Y = BlobGetMutableTensor(Outputs()[0], CPU);
    Y->Resize(X.sizes());

    auto& ctx = *context_.GetSingleton(engine); // this->engine());
    ctx.queues.back().finish();
    if (engine == "FPGA") {
      context_.Copy<bfloat16, float, OpenCLContext, CPUContext>(X, *Y);
    } else {
      context_.Copy<float, float, OpenCLContext, CPUContext>(X, *Y);
    }
    return true;
  }

  bool RunOnDevice() override {
    return RunOnDeviceWithEngine(engine());
  }
};

// TODO: have specific operators dependding on the engine,
// something like
// REGISTER_OPENCL_OPERATOR_WITH_ENGINE(CopyToOpenCL, FPGA, CopyToFPGAOp);
// REGISTER_OPENCL_OPERATOR(CopyToOpenCL, CopyToGenericOpenCLOp);
REGISTER_OPENCL_OPERATOR_WITH_ENGINE(CopyFromOpenCL, FPGA, CopyFromOpenCLOp);
OPERATOR_SCHEMA(CopyFromOpenCL).NumInputs(1).NumOutputs(1);

} // namespace
} // namespace caffe2
