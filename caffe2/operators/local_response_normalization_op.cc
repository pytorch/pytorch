#include "caffe2/operators/local_response_normalization_op.h"

namespace caffe2 {

template<>
bool LRNOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  // Note(Yangqing): this one is copied from my Caffe implementation.
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int image_size = C * H * W;
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->template mutable_data<float>();

  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();
  math::Set<float, CPUContext>(X.numel(), bias_, scale_data, &context_);
  Tensor padded_square(vector<int64_t>{C + size_ - 1, H, W}, CPU);
  float* padded_square_data = padded_square.template mutable_data<float>();
  math::Set<float, CPUContext>(
      padded_square.numel(), 0., padded_square_data, &context_);
  const float alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < N; ++n) {
    // compute the padded square
    math::Sqr<float, CPUContext>(image_size, Xdata + image_size * n,
                                 padded_square_data + pre_pad_ * H * W,
                                 &context_);
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      math::Axpy<float, CPUContext>(
          H * W, alpha_over_size, padded_square_data + c * H * W,
          scale_data + image_size * n, &context_);
    }
    for (int c = 1; c < C; ++c) {
      float* this_scale_slice = scale_data + n * image_size + c * H * W;
      // copy previous scale
      context_.CopyFromCPU<float>(
          H * W, this_scale_slice - H * W, this_scale_slice);
      // add head
      math::Axpy<float, CPUContext>(
          H * W, alpha_over_size, padded_square_data + (c + size_ - 1) * H * W,
          this_scale_slice, &context_);
      // subtract tail
      math::Axpy<float, CPUContext>(
          H * W, -alpha_over_size, padded_square_data + (c - 1) * H * W,
          this_scale_slice, &context_);
    }
  }
  math::Powx<float, CPUContext>(
      X.numel(), scale_data, -beta_, Ydata, &context_);
  math::Mul<float, CPUContext>(X.numel(), Ydata, Xdata, Ydata, &context_);
  return true;
}

template<>
bool LRNOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  // Note(Yangqing): This one is copied from my Decaf implementation. How many
  // variants have I written...?
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const int num_rows = N * H * W;
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->template mutable_data<float>();

  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();

  Tensor padded_square(vector<int64_t>(1, C + size_ - 1), CPU);
  float* padded_square_data = padded_square.template mutable_data<float>();
  math::Set<float, CPUContext>(
      padded_square.numel(), 0., padded_square_data, &context_);
  const float alpha_over_size = alpha_ / size_;

  for (int n = 0; n < num_rows; ++n) {
    for (int c = 0; c < C; ++c) {
      padded_square_data[c + pre_pad_] =
          Xdata[n * C + c] * Xdata[n * C + c] * alpha_over_size;
    }
    float accum_scale = 0.;
    for (int i = 0; i < size_ - 1; ++i) {
      accum_scale += padded_square_data[i];
    }
    for (int c = 0; c < C; ++c) {
      accum_scale += padded_square_data[c + size_ - 1];
      scale_data[n * C + c] = bias_ + accum_scale;
      accum_scale -= padded_square_data[c];
    }
  }
  math::Powx<float, CPUContext>(
      X.numel(), scale_data, -beta_, Ydata, &context_);
  math::Mul<float, CPUContext>(X.numel(), Ydata, Xdata, Ydata, &context_);
  return true;
}

template <>
bool LRNGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int image_size = C * H * W;
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.numel(), Y.numel());
  DCHECK_EQ(X.numel(), dY.numel());
  dX->ResizeLike(X);

  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();

  Tensor padded_ratio(vector<int64_t>{C + size_ - 1, H, W}, CPU);
  float* padded_ratio_data = padded_ratio.template mutable_data<float>();
  // Compute scale(copied from LRNOp) - reusing padded_ratio
  math::Set<float, CPUContext>(X.numel(), bias_, scale_data, &context_);
  math::Set<float, CPUContext>(
      padded_ratio.numel(), 0., padded_ratio_data, &context_);
  const float alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < N; ++n) {
    // compute the padded square
    math::Sqr<float, CPUContext>(image_size, Xdata + image_size * n,
                                 padded_ratio_data + pre_pad_ * H * W,
                                 &context_);
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      math::Axpy<float, CPUContext>(
          H * W, alpha_over_size, padded_ratio_data + c * H * W,
          scale_data + image_size * n, &context_);
    }
    for (int c = 1; c < C; ++c) {
      float* this_scale_slice = scale_data + n * image_size + c * H * W;
      // copy previous scale
      context_.CopyFromCPU<float>(
          H * W, this_scale_slice - H * W, this_scale_slice);
      // add head
      math::Axpy<float, CPUContext>(
          H * W, alpha_over_size, padded_ratio_data + (c + size_ - 1) * H * W,
          this_scale_slice, &context_);
      // subtract tail
      math::Axpy<float, CPUContext>(
          H * W, -alpha_over_size, padded_ratio_data + (c - 1) * H * W,
          this_scale_slice, &context_);
    }
  }

  math::Set<float, CPUContext>(
      padded_ratio.numel(), 0., padded_ratio_data, &context_);
  Tensor accum_ratio(vector<int64_t>{H, W}, CPU);
  float* accum_ratio_data = accum_ratio.template mutable_data<float>();

  const float cache_ratio = 2. * alpha_ * beta_ / size_;
  const int inverse_pre_pad = size_ - (size_ + 1) / 2;

  int offset = 0;
  for (int n = 0; n < N; ++n) {
    // first, compute diff_i * y_i / s_i
    math::Mul<float, CPUContext>(
        image_size, dYdata + offset, Ydata + offset,
        padded_ratio_data + inverse_pre_pad * H * W, &context_);
    math::Div<float, CPUContext>(
        image_size, padded_ratio_data + inverse_pre_pad * H * W,
        scale_data + offset,
        padded_ratio_data + inverse_pre_pad * H * W, &context_);
    // Now, compute the accumulated ratios and the bottom diff
    math::Set<float, CPUContext>(
        accum_ratio.numel(), 0., accum_ratio_data, &context_);
    for (int c = 0; c < size_ - 1; ++c) {
      math::Axpy<float, CPUContext>(H * W, 1,
                                    padded_ratio_data + c * H * W,
                                    accum_ratio_data, &context_);
    }
    for (int c = 0; c < C; ++c) {
      for (int hw = 0; hw < H * W; ++hw) {
        accum_ratio_data[hw] += padded_ratio_data[(c + size_ - 1) * H * W + hw];
        dXdata[offset] =
            dYdata[offset] * pow(scale_data[offset], -beta_) -
            cache_ratio * accum_ratio_data[hw] * Xdata[offset];
        accum_ratio_data[hw] -= padded_ratio_data[c * H * W + hw];
        ++offset;
      }
    }
  }
  return true;
}

template <>
bool LRNGradientOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const int num_rows = N * H * W;
  const float* Xdata = X.data<float>();
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.numel(), Y.numel());
  DCHECK_EQ(X.numel(), dY.numel());
  dX->ResizeLike(X);
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);
  Tensor padded_ratio(vector<int64_t>(1, C + size_ - 1), CPU);
  float* padded_ratio_data = padded_ratio.template mutable_data<float>();
  float* scale_data = scale_->template mutable_data<float>();
  // Compute scale(copied from LRNOp) - reusing padded_ratio
  math::Set<float, CPUContext>(X.numel(), bias_, scale_data, &context_);
  math::Set<float, CPUContext>(
      padded_ratio.numel(), 0., padded_ratio_data, &context_);
  const float alpha_over_size = alpha_ / size_;

  for (int n = 0; n < num_rows; ++n) {
    for (int c = 0; c < C; ++c) {
      padded_ratio_data[c + pre_pad_] =
          Xdata[n * C + c] * Xdata[n * C + c] * alpha_over_size;
    }
    float accum_scale = 0.;
    for (int i = 0; i < size_ - 1; ++i) {
      accum_scale += padded_ratio_data[i];
    }
    for (int c = 0; c < C; ++c) {
      accum_scale += padded_ratio_data[c + size_ - 1];
      scale_data[n * C + c] = bias_ + accum_scale;
      accum_scale -= padded_ratio_data[c];
    }
  }

  math::Set<float, CPUContext>(
      padded_ratio.numel(), 0., padded_ratio_data, &context_);
  // the ratio 2*alpha*beta/size
  const float cache_ratio = 2. * alpha_ * beta_ / size_;
  const float* Ydata = Y.data<float>();

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  for (int n = 0; n < num_rows; ++n) {
    const int offset = n * C;
    for (int c = 0; c < C; ++c) {
      padded_ratio_data[c + pre_pad_] =
          Ydata[offset + c] * dYdata[offset + c] / scale_data[offset + c];
    }
    float accum_ratio = 0.;
    for (int c = 0; c < size_ - 1; ++c) {
      accum_ratio += padded_ratio_data[c];
    }
    for (int c = 0; c < C; ++c) {
      accum_ratio += padded_ratio_data[c + size_ - 1];
      dXdata[offset + c] =
          dYdata[offset + c] * pow(scale_data[offset + c], -beta_) -
          cache_ratio * Xdata[offset + c] * accum_ratio;
      accum_ratio -= padded_ratio_data[c];
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(LRN, LRNOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LRNGradient, LRNGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LRN)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(

`LRN` applies Local Response Normalization to an input blob. This operation performs
a kind of "lateral inhibition" by normalizing over local input regions, where
normalization is applied across channels. This operator is typically used to
normalize an unbounded activation (such as ReLU). The output shape is the same as
the input shape. The `brew` module has a wrapper for this operator for use in a
`ModelHelper` object.

The formula for LRN is as follows:

$$b_{c} = a_{c}(bias + \frac{\alpha}{n}\sum_{c'=max(0,c-n/2)}^{min(N-1,c+n/2)} a_{c'}^2 )^{-\beta}$$


Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator("LRN",
     ["X"],
     ["Y", "Y_scale"],
     size=11,
     alpha=0.001,
     beta=0.5,
     bias=2.0,
     order="NHWC"
)

workspace.FeedBlob("X", np.random.randn(1, 6, 6, 1).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
print("Y_scale:\n", workspace.FetchBlob("Y_scale"))
```

**Result**

```
X:
 [[[[ 0.72985137]
   [-0.3753357 ]
   [ 2.7344604 ]
   [-0.5937792 ]
   [ 0.38440478]
   [-2.1659644 ]]

  [[-0.92846817]
   [-0.9996144 ]
   [ 0.212943  ]
   [-1.968045  ]
   [-0.77839696]
   [ 0.45492038]]

  [[-0.11263168]
   [ 1.9901097 ]
   [ 0.19275683]
   [ 0.15630436]
   [ 0.7536298 ]
   [-0.77339894]]

  [[ 0.8353551 ]
   [-0.7784452 ]
   [ 1.779317  ]
   [ 0.22421335]
   [ 1.3846219 ]
   [-3.0546608 ]]

  [[ 0.09977621]
   [ 2.2071757 ]
   [ 0.79971045]
   [ 3.563886  ]
   [-0.7169287 ]
   [ 0.77170426]]

  [[-1.4296649 ]
   [ 0.19181213]
   [ 0.45961624]
   [-1.0201577 ]
   [ 0.62854475]
   [-0.6395456 ]]]]

Y:
 [[[[ 0.5160766 ]
   [-0.26540157]
   [ 1.9332271 ]
   [-0.41986194]
   [ 0.27181432]
   [-1.5314047 ]]

  [[-0.6565133 ]
   [-0.7068181 ]
   [ 0.15057328]
   [-1.3914955 ]
   [-0.5504022 ]
   [ 0.32167578]]

  [[-0.0796426 ]
   [ 1.4070934 ]
   [ 0.13629955]
   [ 0.11052381]
   [ 0.53288984]
   [-0.5468682 ]]

  [[ 0.5906759 ]
   [-0.5504363 ]
   [ 1.2580767 ]
   [ 0.1585426 ]
   [ 0.9790328 ]
   [-2.1595135 ]]

  [[ 0.07055242]
   [ 1.5605361 ]
   [ 0.5654725 ]
   [ 2.5193207 ]
   [-0.50693923]
   [ 0.54567   ]]

  [[-1.0108787 ]
   [ 0.13563155]
   [ 0.3249962 ]
   [-0.72134334]
   [ 0.44444424]
   [-0.45222285]]]]
Y_scale:
 [[[[2.0000484]
   [2.0000129]
   [2.0006797]
   [2.000032 ]
   [2.0000134]
   [2.0004265]]

  [[2.0000784]
   [2.0000908]
   [2.000004 ]
   [2.0003521]
   [2.000055 ]
   [2.0000188]]

  [[2.0000012]
   [2.00036  ]
   [2.0000033]
   [2.0000021]
   [2.0000517]
   [2.0000544]]

  [[2.0000634]
   [2.000055 ]
   [2.0002878]
   [2.0000045]
   [2.0001743]
   [2.0008483]]

  [[2.000001 ]
   [2.000443 ]
   [2.0000582]
   [2.0011547]
   [2.0000467]
   [2.0000541]]

  [[2.0001857]
   [2.0000033]
   [2.0000193]
   [2.0000947]
   [2.000036 ]
   [2.0000372]]]]
```

</details>

)DOC")
    .Arg(
        "size",
        "*(type: int; default: 0)* Amount of neighboring channels to sum over for normalization")
    .Arg(
        "alpha",
        "*(type: float; default: 0)* Multiplicative (scaling) factor.")
    .Arg("beta", "*(type: float; default: 0)* Exponent.")
    .Arg("bias", "*(type: float; default: 1.0)* Additive factor.")
    .Arg("order", "*(type: float; default: 'NCHW')* Order of blob dimensions.")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor (ReLU output).")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.")
    .Output(1, "Y_scale", "*(type: Tensor`<float>`)* Output scale.")
    .InheritOnnxSchema();
OPERATOR_SCHEMA(LRNGradient).NumInputs(3).NumOutputs(1);

class GetLRNGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
      "LRNGradient", "",
      vector<string>{I(0), O(0), GO(0)},
      vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LRN, GetLRNGradient);
}  // namespace caffe2
