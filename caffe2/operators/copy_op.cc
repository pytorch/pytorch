#include "caffe2/operators/copy_op.h"

namespace caffe2 {

// From CPU, copy it to whatever the current context
REGISTER_CPU_OPERATOR(
    CopyFromCPUInput,
    CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(Copy, CopyOp<CPUContext, CPUContext, CPUContext>);

OPERATOR_SCHEMA(Copy)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .InheritOnnxSchema("Identity")
    .SetDoc(R"DOC(
Copy input tensor into output, potentially across devices.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/copy_op.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/copy_op.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Copy",
    ["input"],
    ["output"]
)

workspace.FeedBlob("input", np.random.rand(3,3))
print("input:", workspace.FetchBlob("input"))
workspace.RunOperatorOnce(op)
print("output:", workspace.FetchBlob("output"))

```

**Result**

```

input:
[[0.16826761 0.68168217 0.55196001]
 [0.19735483 0.34837823 0.69015595]
 [0.09448514 0.57390828 0.37097193]]
output:
[[0.16826761 0.68168217 0.55196001]
 [0.19735483 0.34837823 0.69015595]
 [0.09448514 0.57390828 0.37097193]]

```

</details>

)DOC")
    .Input(0, "input", "(*Tensor*): input tensor to copy")
    .Output(0, "output", "(*Tensor*): copy of input tensor");

OPERATOR_SCHEMA(CopyGPUToCPU)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      CAFFE_ENFORCE(
          def.has_device_option(),
          "CopyGPUToCPU op should have cuda device option.");
      auto& cuda_option = def.device_option();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), cuda_option);
      vector<DeviceOption> out_dev(def.output_size(), cpu_option);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Copy tensor for GPU to CPU context. Must be run under GPU device option.
)DOC")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(CopyCPUToGPU)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      CAFFE_ENFORCE(
          def.has_device_option(),
          "CopyCPUToGPU op should have cuda device option.");
      auto& cuda_option = def.device_option();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), cpu_option);
      vector<DeviceOption> out_dev(def.output_size(), cuda_option);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Copy tensor for CPU to GPU context. Must be run under GPU device option.
)DOC")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(CopyFromCPUInput)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), cpu_option);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Take a CPU input tensor and copy it to an output in the current
Context (GPU or CPU). This may involves cross-device MemCpy.
)DOC")
    .Input(0, "input", "The input CPU tensor.")
    .Output(0, "output", "either a TensorCUDA or a TensorCPU");

OPERATOR_SCHEMA(CopyOnDeviceLike)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Copy input tensor into output to the specific device.")
    .Input(0, "input", "The input tensor.")
    .Input(1, "dst", "Tensor, on which device the copy will be performed.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

struct GetCopyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CopyOnDeviceLike",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Copy, GetCopyGradient);

struct GetGPUToCPUGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (g_output_[0].IsDense()) {
      return SingleGradientDef(
          "CopyCPUToGPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return vector<OperatorDef>{CreateOperatorDef(
                                     "CopyCPUToGPU",
                                     "",
                                     std::vector<string>{GO_I(0)},
                                     std::vector<string>{GI_I(0)}),
                                 CreateOperatorDef(
                                     "CopyCPUToGPU",
                                     "",
                                     std::vector<string>{GO_V(0)},
                                     std::vector<string>{GI_V(0)})};
    }
  }
};
REGISTER_GRADIENT(CopyGPUToCPU, GetGPUToCPUGradient);

struct GetCPUToGPUGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (g_output_[0].IsDense()) {
      return SingleGradientDef(
          "CopyGPUToCPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return vector<OperatorDef>{CreateOperatorDef(
                                     "CopyGPUToCPU",
                                     "",
                                     std::vector<string>{GO_I(0)},
                                     std::vector<string>{GI_I(0)}),
                                 CreateOperatorDef(
                                     "CopyGPUToCPU",
                                     "",
                                     std::vector<string>{GO_V(0)},
                                     std::vector<string>{GI_V(0)})};
    }
  }
};
REGISTER_GRADIENT(CopyCPUToGPU, GetCPUToGPUGradient);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(
    CopyGPUToCPU,
    "_caffe2::CopyGPUToCPU(Tensor input) -> Tensor");

C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(
    CopyCPUToGPU,
    "_caffe2::CopyCPUToGPU(Tensor input) -> Tensor");
