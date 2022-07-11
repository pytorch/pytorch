#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

std::vector<TensorShape> TensorInferenceForConvGradient(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  CAFFE_ENFORCE_EQ(in.size(), 3U, "ConvGradient requires 3 inputs");

  if (in[0].unknown_shape()) {
    std::vector<TensorShape> out(1);
    out[0].set_unknown_shape(true);
    return out;
  }
  ArgumentHelper helper(def);
  const auto no_bias = helper.GetSingleArgument<int>("no_bias", 0);
  const auto n_outputs = def.output_size();
  vector<TensorShape> out(n_outputs);

  // FILTER_GRAD has the same shape as FILTER
  out[0] = in[1];
  if (!no_bias) {
    vector<int64_t> bias_shape = {in[1].dims(0)};
    out[1] = CreateTensorShape(bias_shape, in[1].data_type());
  }

  if (n_outputs == 3 || (no_bias && n_outputs == 2)) {
    // INPUT_GRAD has the same shape as INPUT
    out[out.size() - 1] = in[0];
  }

  return out;
}

OpSchema::Cost CostInferenceForConvGradient(
    const OperatorDef& def,
    const vector<TensorShape>& inputs) {
  CAFFE_ENFORCE_EQ(inputs.size(), 3U, "ConvGradient requires 3 inputs");
  ArgumentHelper helper(def);
  const auto order =
      StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
  const auto no_bias = helper.GetSingleArgument<int>("no_bias", 0);
  const auto n_outputs = def.output_size();

  const auto& outputs = TensorInferenceForConvGradient(def, inputs);
  const auto& X = inputs[0];
  const auto& filter = inputs[1];
  const auto& dY = inputs[2];
  const auto N = X.dims(0);
  const auto M = filter.dims(0);
  const auto C =
      (order == StorageOrder::NCHW ? X.dims(1) : X.dims(X.dims_size() - 1));
  const auto output_image_size =
      (order == StorageOrder::NCHW
           ? nElemFromDim(dY, 2)
           : nElemBetweenDim(dY, 1, dY.dims_size() - 1));
  auto kernel_elem =
      (order == StorageOrder::NCHW
           ? nElemFromDim(filter, 2)
           : nElemBetweenDim(filter, 1, filter.dims_size() - 1));

  struct OpSchema::Cost c;
  c.flops = N * 2 * M * kernel_elem * C * output_image_size;
  if (!no_bias) {
    c.flops += N * (M * output_image_size);
  }
  if (n_outputs == 3 || (no_bias && n_outputs == 2)) {
    c.flops += N * 2 * M * kernel_elem * C * output_image_size;
  }

  c.bytes_read = (nElemFromDim(X) + nElemFromDim(filter) + nElemFromDim(dY)) *
      sizeof(float);

  for (auto i = 0; i < n_outputs; i++) {
    c.bytes_written += nElemFromDim(outputs[i]) * sizeof(float);
  }
  c.params_bytes = nElemFromDim(filter) * sizeof(float);

  return c;
}

REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(ConvGradient)
    .NumInputs(2, 3)
    .NumOutputs(1, 3)
    .TensorInferenceFunction(TensorInferenceForConvGradient)
    .CostInferenceFunction(CostInferenceForConvGradient);

REGISTER_CPU_OPERATOR(Conv1DGradient, ConvGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(Conv1DGradient).NumInputs(2, 3).NumOutputs(1, 3);

REGISTER_CPU_OPERATOR(Conv2DGradient, ConvGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(Conv2DGradient).NumInputs(2, 3).NumOutputs(1, 3);

REGISTER_CPU_OPERATOR(Conv3DGradient, ConvGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(Conv3DGradient).NumInputs(2, 3).NumOutputs(1, 3);

class GetConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);

    ArgumentHelper argsHelper(def_);

    // NOLINTNEXTLINE(modernize-use-bool-literals)
    auto compute_dX = !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

    if (def_.input_size() == 3) {
      if (compute_dX) {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(2), GI(0)});
      } else {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(2)});
      }
    } else {
      if (compute_dX) {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(0)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      } else {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      }
    }
  }
};
REGISTER_GRADIENT(Conv, GetConvGradient);
REGISTER_GRADIENT(Conv1D, GetConvGradient);
REGISTER_GRADIENT(Conv2D, GetConvGradient);
REGISTER_GRADIENT(Conv3D, GetConvGradient);

} // namespace caffe2
