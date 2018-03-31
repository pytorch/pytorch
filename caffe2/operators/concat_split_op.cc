#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
namespace {
std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>> splitOpDevInfer(
    const OperatorDef& def) {
  auto op_device =
      def.has_device_option() ? def.device_option() : DeviceOption();
  vector<DeviceOption> in_dev(def.input_size(), op_device);
  vector<DeviceOption> out_dev(def.output_size(), op_device);

  // If we obtain split from input tensor, then 2nd input's type is always CPU.
  if (def.input_size() == SplitOp<CPUContext>::kSplitOpInputSize) {
    CAFFE_ENFORCE_GT(in_dev.size(), 1);
    in_dev[1] = DeviceOption();
  }
  return std::make_pair(in_dev, out_dev);
}
} // namespace.

REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);
OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')")
    .Arg("axis", "Which axis to split on")
    .Arg("split", "length of each output")
    .Arg("order", "Either NHWC or NCWH, will split on C axis, defaults to NCHW")
    .DeviceInferenceFunction(splitOpDevInfer)
    .SetDoc(R"DOC(
Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'split' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC")
    .InheritOnnxSchema("Split");

namespace {
OpSchema::Cost CostInferenceForConcat(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  ArgumentHelper helper(def);
  const int axis = helper.HasArgument("axis")
      ? helper.GetSingleArgument<int>("axis", -1)
      : GetDimFromOrderString(
            helper.GetSingleArgument<string>("order", "NCHW"));
  bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
  const int canonical_axis = canonical_axis_index_(axis, in[0].dims_size());
  CAFFE_ENFORCE_GT(in.size(), 0);
  vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
  if (add_axis) {
    out_shape.insert(out_shape.begin() + canonical_axis, in.size());
  } else {
    for (int i = 1; i < in.size(); ++i) {
      out_shape[canonical_axis] += in[i].dims(canonical_axis);
    }
  }
  int size = 1;
  for (auto& s : out_shape) {
    size *= s;
  }

  struct OpSchema::Cost cost;
  cost.flops = 0;
  cost.bytes_moved = size * sizeof(float);
  cost.params_bytes = 0;
  return cost;
}

std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>>
concatOpDevInfer(const OperatorDef& def) {
  auto op_device =
      def.has_device_option() ? def.device_option() : DeviceOption();
  vector<DeviceOption> in_dev(def.input_size(), op_device);
  vector<DeviceOption> out_dev(def.output_size(), op_device);

  // 2nd output's type is always CPU irrespective of op's device option.
  CAFFE_ENFORCE_GT(out_dev.size(), 1);
  out_dev[1] = DeviceOption();
  return std::make_pair(in_dev, out_dev);
}
} // namespace

REGISTER_CPU_OPERATOR(Concat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .Arg("axis", "Which axis to concat on")
    .Arg(
        "order",
        "Either NHWC or NCHW, will concat on C axis, defaults to NCHW")
    .Arg(
        "add_axis",
        "Pass 1 to add the axis specified in arg 'axis' to all "
        "input tensors")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
      const int canonical_axis = canonical_axis_index_(axis, in[0].dims_size());
      CAFFE_ENFORCE_GT(in.size(), 0);
      vector<int> split_shape(1, in.size());
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE_EQ(
              in[0].dims().size(),
              in[i].dims().size(),
              "All inputs of Concat should have same dims when add_axis = 1. "
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims().size(); ++j) {
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims when add_axis = 1. "
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE_EQ(
              in[0].dims().size(),
              in[i].dims().size(),
              "All inputs of Concat should have same dims except "
              "canonical_axis dim that is equal to ",
              canonical_axis,
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims().size(); ++j) {
            if (j == canonical_axis) {
              continue;
            }
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims except "
                "canonical_axis dim that is equal to ",
                canonical_axis,
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }

        for (int i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      if (def.output_size() == 1) {
        return vector<TensorShape>{
            CreateTensorShape(out_shape, in[0].data_type())};
      }
      return vector<TensorShape>{
          CreateTensorShape(out_shape, in[0].data_type()),
          CreateTensorShape(split_shape, TensorProto::INT32)};
    })
    .CostInferenceFunction(CostInferenceForConcat)
    .DeviceInferenceFunction(concatOpDevInfer)
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Output(0, "concat_result", "Concatenated tensor")
    .Output(1, "split_info", "The dimensions of the inputs.")
    .InheritOnnxSchema("Concat");

// Backward compatibility names.
REGISTER_CPU_OPERATOR(DepthSplit, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(DepthSplit)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .SetDoc("Backward compatible operator name for Split.");
OPERATOR_SCHEMA(DepthConcat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .SetDoc("Backward compatible operator name for Concat.");

class GetSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> output_grads;
    for (int i = 0; i < def_.output_size(); ++i) {
      if (!GradOut(i).IsEmpty()) {
        output_grads.push_back(GO(i));
      }
    }
    if (output_grads.empty()) {
      return {};
    }
    return SingleGradientDef(
        "Concat",
        "",
        output_grads,
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }
};
REGISTER_GRADIENT(Split, GetSplitGradient);
REGISTER_GRADIENT(DepthSplit, GetSplitGradient);

class GetConcatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (GradOut(0).IsEmpty()) {
      return {};
    }
    vector<string> grads;
    for (int i = 0; i < def_.input_size(); ++i) {
      grads.push_back(GI(i));
    }
    return SingleGradientDef("Split", "", vector<string>{GO(0), O(1)}, grads);
  }
};
REGISTER_GRADIENT(Concat, GetConcatGradient);
REGISTER_GRADIENT(DepthConcat, GetConcatGradient);
} // namespace caffe2
