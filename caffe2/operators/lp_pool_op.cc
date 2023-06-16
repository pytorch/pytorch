// TODO: reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"

namespace caffe2 {

using std::max;
using std::min;

struct LpPoolFunctor {
  explicit LpPoolFunctor(const OperatorBase& /* op */) {}
};

template <>
bool PoolOp<float, CPUContext, LpPoolFunctor>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));
  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();
  math::Set<float, CPUContext>(Y->numel(), 0, Ydata, &context_);
  // The main loop
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);
  int pooled_height = Y->dim32(2);
  int pooled_width = Y->dim32(3);

  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_[0] - pads_[0];
          int wstart = pw * stride_[1] - pads_[1];
          int hend = min(hstart + kernel_[0], height);
          int wend = min(wstart + kernel_[1], width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              Ydata[pool_index] += std::pow(std::abs(Xdata[input_index]), p);
            }
          }
          Ydata[pool_index] = std::pow(Ydata[pool_index], inv_p);
        }
      }
      // Do offset.
      Xdata += height * width;
      Ydata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, LpPoolFunctor>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);

  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();
  math::Set<float, CPUContext>(Y->numel(), 0, Ydata, &context_);
  // The main loop
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_[0] - pads_[0];
        int wstart = pw * stride_[1] - pads_[1];
        int hend = min(hstart + kernel_[0], height);
        int wend = min(wstart + kernel_[1], width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        const int pool_index = (ph * pooled_width + pw) * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (h * width + w) * channels;
            for (int c = 0; c < channels; ++c) {
              Ydata[pool_index + c] +=
                  std::pow(std::abs(Xdata[input_index + c]), p);
            }
          }
        }
        for (int c = 0; c < channels; ++c) {
          Ydata[pool_index + c] = std::pow(Ydata[pool_index + c], inv_p);
        }
      }
    }
    // Do offset.
    Xdata += X.numel() / X.dim32(0);
    Ydata += Y->numel() / Y->dim32(0);
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, LpPoolFunctor>::
    RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  const auto& Y = Input(1);
  auto& dY = Input(2);

  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);

  // TODO(Yangqing): Add shape checks.
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  math::Set<float, CPUContext>(
      X.numel(), 0, dX->template mutable_data<float>(), &context_);
  const float* dYdata = dY.data<float>();
  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  float* dXdata = dX->template mutable_data<float>();

  int channels = X.dim32(1);
  CAFFE_ENFORCE_EQ(channels, dY.dim32(1));
  int height = X.dim32(2);
  int width = X.dim32(3);
  ConvPoolOpBase<CPUContext>::ComputePads({height, width});
  int pooled_height = dY.dim32(2);
  int pooled_width = dY.dim32(3);
  // The main loop
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_[0] - pads_[0];
          int wstart = pw * stride_[1] - pads_[1];
          int hend = min(hstart + kernel_[0], height);
          int wend = min(wstart + kernel_[1], width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
              dXdata[h * width + w] += dYdata[ph * pooled_width + pw] *
                  Xdata[h * width + w] *
                  std::pow(std::abs(Xdata[h * width + w]), p - 2) /
                  std::pow(Ydata[ph * pooled_width + pw], p - 1);
            }
          }
        }
      }
      // offset
      dXdata += height * width;
      dYdata += pooled_height * pooled_width;
      Ydata += pooled_height * pooled_width;
      Xdata += height * width;
    }
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, LpPoolFunctor>::
    RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  const auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.dim(), 4);

  // TODO(Yangqing): Add shape checks.
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  math::Set<float, CPUContext>(
      X.numel(), 0, dX->template mutable_data<float>(), &context_);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  // The main loop
  int height = X.dim32(1);
  int width = X.dim32(2);
  ConvPoolOpBase<CPUContext>::ComputePads({height, width});
  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);

  int pooled_height = dY.dim32(1);
  int pooled_width = dY.dim32(2);
  int channels = X.dim32(3);
  CAFFE_ENFORCE_EQ(channels, dY.dim32(3));
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_[0] - pads_[0];
        int wstart = pw * stride_[1] - pads_[1];
        int hend = min(hstart + kernel_[0], height);
        int wend = min(wstart + kernel_[1], width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            for (int c = 0; c < channels; ++c) {
              dXdata[(h * width + w) * channels + c] +=
                  dYdata[(ph * pooled_width + pw) * channels + c] *
                  Xdata[(h * width + w) * channels + c] *
                  std::pow(
                      std::abs(Xdata[(h * width + w) * channels + c]), p - 2) /
                  std::pow(
                      Ydata[(ph * pooled_width + pw) * channels + c], p - 1);
            }
          }
        }
      }
    }
    // offset
    dXdata += X.numel() / X.dim32(0);
    dYdata += dY.numel() / dY.dim32(0);
    Xdata += X.numel() / X.dim32(0);
    Ydata += Y.numel() / Y.dim32(0);
  }
  return true;
}

REGISTER_CPU_OPERATOR(LpPool, PoolOp<float, CPUContext, LpPoolFunctor>);
REGISTER_CPU_OPERATOR(
    LpPoolGradient,
    PoolGradientOp<float, CPUContext, LpPoolFunctor>);

OPERATOR_SCHEMA(LpPool)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
`LpPool` consumes an input blob and applies max pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. $L_p$ pooling consists of taking the $L_p$ norm of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/lp_pool_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LpPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
    p=2.0
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[[[-1.1113514  -1.1173418  -0.1504435   0.1327146  -1.2221841  -0.5654315 ]
   [-1.9209646  -0.04675794  0.8604731   1.2042469   0.28154245   0.38656202]
   [-0.8772837  -0.03264008  0.26222762  0.28526652  0.321102    -2.5891325 ]
   [-0.9248281   1.440776   -0.56832    -0.6017927   1.2262512   -2.1443934 ]
   [ 0.5194415  -1.6858683   0.45221648  0.65029615 -0.8574544    0.8121054 ]
   [ 0.25902653  0.4934758   0.49870652 -0.48134378 -0.9178449   -0.07626943]]]]

Y:
 [[[[2.4851248 1.49361   1.4290358]
   [1.9240153 0.9139378 3.5928857]
   [1.8500228 1.0525136 1.4976646]]]]

```

</details>

)DOC")
    .Arg("p", "(*float*): type of $L_p$ norm to use (default=2.0)")
    .Arg("kernel", "(*int*): the size of the window to take a max over")
    .Arg("stride", "(*int*): the stride of the window")
    .Arg("pad", "(*int*): implicit zero padding to be added on both sides")
    .Arg(
        "dilation",
        "(*int*): parameter that controls the stride of elements in the window")
    .Arg("order", "(*string*): order of blob dimensions (default=\"NCHW\")")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Output(0, "Y", "(*Tensor`<float>`*): output tensor");

OPERATOR_SCHEMA(LpPoolGradient).NumInputs(3).NumOutputs(1);

class GetPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LpPool, GetPoolGradient);
} // namespace caffe2
