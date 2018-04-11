#include "caffe2/operators/local_response_normalization_op.h"

namespace caffe2 {

template<>
bool LRNOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  // Note(Yangqing): this one is copied from my Caffe implementation.
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int image_size = C * H * W;
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();

  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->mutable_data<float>();
  math::Set<float, CPUContext>(X.size(), bias_, scale_data, &context_);
  TensorCPU padded_square(
      vector<TIndex>{C + size_ - 1, H, W});
  float* padded_square_data = padded_square.mutable_data<float>();
  math::Set<float, CPUContext>(padded_square.size(), 0., padded_square_data,
                               &context_);
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
      context_.Copy<float, CPUContext, CPUContext>(
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
      X.size(), scale_data, -beta_, Ydata, &context_);
  math::Mul<float, CPUContext>(X.size(), Ydata, Xdata, Ydata, &context_);
  return true;
}

template<>
bool LRNOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  // Note(Yangqing): This one is copied from my Decaf implementation. How many
  // variants have I written...?
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const int num_rows = N * H * W;
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();

  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->mutable_data<float>();

  TensorCPU padded_square(vector<TIndex>(1, C + size_ - 1));
  float* padded_square_data = padded_square.mutable_data<float>();
  math::Set<float, CPUContext>(padded_square.size(), 0., padded_square_data,
                               &context_);
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
      X.size(), scale_data, -beta_, Ydata, &context_);
  math::Mul<float, CPUContext>(X.size(), Ydata, Xdata, Ydata, &context_);
  return true;
}

template <>
bool LRNGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int image_size = C * H * W;
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.size(), Y.size());
  DCHECK_EQ(X.size(), dY.size());
  dX->ResizeLike(X);

  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->mutable_data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  TensorCPU padded_ratio(
      vector<TIndex>{C + size_ - 1, H, W});
  float* padded_ratio_data = padded_ratio.mutable_data<float>();
  // Compute scale(copied from LRNOp) - reusing padded_ratio
  math::Set<float, CPUContext>(X.size(), bias_, scale_data, &context_);
  math::Set<float, CPUContext>(padded_ratio.size(), 0., padded_ratio_data,
                               &context_);
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
      context_.Copy<float, CPUContext, CPUContext>(
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

  math::Set<float, CPUContext>(padded_ratio.size(), 0., padded_ratio_data,
                               &context_);
  TensorCPU accum_ratio(vector<TIndex>{H, W});
  float* accum_ratio_data = accum_ratio.mutable_data<float>();


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
    math::Set<float, CPUContext>(accum_ratio.size(), 0., accum_ratio_data,
                                 &context_);
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
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const int num_rows = N * H * W;
  const float* Xdata = X.data<float>();
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.size(), Y.size());
  DCHECK_EQ(X.size(), dY.size());
  dX->ResizeLike(X);
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);
  TensorCPU padded_ratio(vector<TIndex>(1, C + size_ - 1));
  float* padded_ratio_data = padded_ratio.mutable_data<float>();
  float* scale_data = scale_->mutable_data<float>();
  // Compute scale(copied from LRNOp) - reusing padded_ratio
  math::Set<float, CPUContext>(X.size(), bias_, scale_data, &context_);
  math::Set<float, CPUContext>(padded_ratio.size(), 0., padded_ratio_data,
                               &context_);
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

  math::Set<float, CPUContext>(padded_ratio.size(), 0., padded_ratio_data,
                               &context_);
  // the ratio 2*alpha*beta/size
  const float cache_ratio = 2. * alpha_ * beta_ / size_;
  const float* Ydata = Y.data<float>();

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
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

OPERATOR_SCHEMA(LRN).NumInputs(1).NumOutputs(1, 2).InheritOnnxSchema("LRN");
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
