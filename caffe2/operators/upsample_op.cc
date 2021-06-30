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

#include "caffe2/operators/upsample_op.h"

#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool UpsampleBilinearOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);

  if (InputSize() == 2) {
    const auto& scales = Input(1);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    const float* scales_data = scales.data<float>();
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }

  const int batch_size = X.dim32(0);
  const int num_channels = X.dim32(1);
  const int input_height = X.dim32(2);
  const int input_width = X.dim32(3);
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_width = input_width * width_scale_;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_height = input_height * height_scale_;
  auto* Y = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());

  const float* input = X.data<float>();
  float* output = Y->mutable_data<float>();
  int channels = num_channels * batch_size;

  const float rheight = (output_height > 1)
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? (float)(input_height - 1) / (output_height - 1)
      : 0.f;
  const float rwidth =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
  for (int h2 = 0; h2 < output_height; ++h2) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const float h1r = rheight * h2;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int h1 = h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const float h1lambda = h1r - h1;
    const float h0lambda = (float)1. - h1lambda;
    for (int w2 = 0; w2 < output_width; ++w2) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const float w1r = rwidth * w2;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const int w1 = w1r;
      const int w1p = (w1 < input_width - 1) ? 1 : 0;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const float w1lambda = w1r - w1;
      const float w0lambda = (float)1. - w1lambda;
      const float* Xdata = &input[h1 * input_width + w1];
      float* Ydata = &output[h2 * output_width + w2];
      for (int c = 0; c < channels; ++c) {
        Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
            h1lambda *
                (w0lambda * Xdata[h1p * input_width] +
                 w1lambda * Xdata[h1p * input_width + w1p]);
        Xdata += input_width * input_height;
        Ydata += output_width * output_height;
      }
    }
  }

  return true;
}

template <>
bool UpsampleBilinearGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);

  if (InputSize() == 3) {
    const auto& scales = Input(2);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    const float* scales_data = scales.data<float>();
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }

  const auto inputDims = dY.sizes();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0);
  const int num_channels = dY.dim32(1);
  const int input_height = dY.dim32(2);
  const int input_width = dY.dim32(3);
  const int output_height = X.dim32(2);
  const int output_width = X.dim32(3);
  auto* dX = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());
  math::Set<float, CPUContext>(
      dX->numel(), 0.0f, dX->mutable_data<float>(), &context_);

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  int channels = num_channels * batch_size;

  const float rheight = (input_height > 1)
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? (float)(output_height - 1) / (input_height - 1)
      : 0.f;
  const float rwidth =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      (input_width > 1) ? (float)(output_width - 1) / (input_width - 1) : 0.f;

  for (int h2 = 0; h2 < input_height; ++h2) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const float h1r = rheight * h2;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int h1 = h1r;
    const int h1p = (h1 < output_height - 1) ? 1 : 0;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const float h1lambda = h1r - h1;
    const float h0lambda = (float)1. - h1lambda;
    for (int w2 = 0; w2 < input_width; ++w2) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const float w1r = rwidth * w2;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const int w1 = w1r;
      const int w1p = (w1 < output_width - 1) ? 1 : 0;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const float w1lambda = w1r - w1;
      const float w0lambda = (float)1. - w1lambda;
      float* pos1 = &dXdata[h1 * output_width + w1];
      const float* pos2 = &dYdata[h2 * input_width + w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += h0lambda * w0lambda * pos2[0];
        pos1[w1p] += h0lambda * w1lambda * pos2[0];
        pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];
        pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];
        pos1 += output_width * output_height;
        pos2 += input_width * input_height;
      }
    }
  }

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(UpsampleBilinear, UpsampleBilinearOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    UpsampleBilinearGradient,
    UpsampleBilinearGradientOp<float, CPUContext>);

// Input: X, output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(UpsampleBilinear)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .SetDoc(R"DOC(
Resizes the spatial dimensions of the input using bilinear
interpolation. The `width_scale` and `height_scale` arguments
control the size of the output, which is given by:
output_width = floor(input_width * width_scale)
output_height = floor(output_height * height_scale)
)DOC")
    .Input(0, "X", "Input tensor")
    .Input(
        1,
        "scales",
        "1D, 2-element, Scales tensor, [height_scale, width_scale]")
    .Output(0, "Y", "Output tensor");

// Input: dY, output: dX
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(UpsampleBilinearGradient)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension");

class GetUpsampleBilinearGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (def_.input().size() == 2) {
      // this is a hack to support the second input as dynamic
      // width_scale and height_scale to align with onnx change
      return SingleGradientDef(
          "UpsampleBilinearGradient",
          "",
          vector<string>{GO(0), I(0), I(1)},
          vector<string>{GI(0)});
    }
    return SingleGradientDef(
        "UpsampleBilinearGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(UpsampleBilinear, GetUpsampleBilinearGradient);

} // namespace caffe2
