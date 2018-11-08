#ifndef CAFFE2_FB_OPENCL_OPERATORS_FULLY_CONNECTED_OP_H_
#define CAFFE2_FB_OPENCL_OPERATORS_FULLY_CONNECTED_OP_H_

#include "caffe2/operators/fully_connected_op.h"
#include "../common_fpga.h"
#include "../context.h"
#include "../operator.h"

namespace caffe2 {
template <>
bool FullyConnectedOp<OpenCLContext, FPGAEngine>::RunOnDevice() {
  CAFFE_ENFORCE(Input(0).template IsType<float>());
  return DoRunWithType<
      float, // X
      float, // W
      float, // B
      float, // Y
      float>(); // Math
}

template <>
bool FullyConnectedGradientOp<OpenCLContext, FPGAEngine>::RunOnDevice() {
  return DoRunWithType<
      float, //  X
      float, //  W
      float, // dY
      float, //  B
      float, // dX
      float, // dW
      float, // dB
      float>(); // Math
}

REGISTER_OPENCL_OPERATOR_WITH_ENGINE(
    FC,
    FPGA,
    FullyConnectedOp<OpenCLContext, FPGAEngine>);
REGISTER_OPENCL_OPERATOR_WITH_ENGINE(
    FCGradient,
    FPGA,
    FullyConnectedGradientOp<OpenCLContext, FPGAEngine>);

REGISTER_OPENCL_OPERATOR_WITH_ENGINE(
    FCTransposed,
    FPGA,
    FullyConnectedOp<
        OpenCLContext,
        FPGAEngine,
        false /* don't transpose weight */>);
REGISTER_OPENCL_OPERATOR_WITH_ENGINE(
    FCTransposedGradient,
    FPGA,
    FullyConnectedGradientOp<
        OpenCLContext,
        FPGAEngine,
        false /* don't transpose weight */>);

// TODO: REGISTER_GRADIENT

} // namespace caffe2

#endif
