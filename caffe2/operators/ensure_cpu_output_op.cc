#include "caffe2/operators/ensure_cpu_output_op.h"

namespace caffe2 {

// From CPU Context, the op takes CPU tensor as input, and produces
// TensorCPU
REGISTER_CPU_OPERATOR(EnsureCPUOutput, EnsureCPUOutputOp<CPUContext>);

OPERATOR_SCHEMA(EnsureCPUOutput)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), cpu_option);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
This Op always create TensorCPU output, and may involves cross-device MemCpy.
Under CPU Context, this Op takes TensorCPU as input. Under the CUDA Context,
this Op accepts either CUDA or CPU Tensor input.
)DOC")
    .Input(0, "input", "The input CUDA or CPU tensor.")
    .Output(0, "output", "TensorCPU that is a copy of the input.");

NO_GRADIENT(EnsureCPUOutput);
} // namespace caffe2
