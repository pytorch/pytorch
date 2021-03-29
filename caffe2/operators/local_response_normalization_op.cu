#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/local_response_normalization_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void LRNFillScaleNCHW(const int nthreads, const T* in,
    const int num, const int channels, const int height,
    const int width, const int size, const T alpha_over_size,
    const T bias, T* scale) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    T accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = bias + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = bias + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = bias + accum_scale * alpha_over_size;
      ++head;
    }
    // recover the pointers for the next loop.
    in -= offset;
    scale -= offset;
  }
}

template <typename T>
__global__ void LRNFillScaleNHWC(const int nthreads, const T* in,
    const int num, const int height, const int width,
    const int channels, const int size, const T alpha_over_size,
    const T bias, T* scale) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int pre_pad = (size - 1) / 2;
    scale[index] = 0;
    for (int i = 0; i < size; ++i) {
      int raw_idx = c + i - pre_pad;
      if (raw_idx >= 0 && raw_idx < channels) {
        scale[index] += in[index + i - pre_pad] * in[index + i - pre_pad];
      }
    }
    scale[index] = bias + scale[index] * alpha_over_size;
  }
}

// TODO(Yangqing): check if it would be faster to just put it into the previous
// kernel.
template <typename T>
__global__ void LRNComputeOutput(const int nthreads, const T* in,
    const T* scale, const T negative_beta, T* out) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename T>
__global__ void LRNComputeDiffNCHW(const int nthreads, const T* bottom_data,
    const T* top_data, const T* scale, const T* top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const T negative_beta,
    const T cache_ratio,
    T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    T accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // recover pointer for next iteration.
    bottom_data -= offset;
    top_data -= offset;
    scale -= offset;
    top_diff -= offset;
    bottom_diff -= offset;
  }
}

// This local response normalization gradient does one sum per output location
// and does not use the running trick for 1-d convolution: thus it might not be
// the fastest implementation.
template <typename T>
__global__ void LRNComputeDiffNHWC(const int nthreads, const T* bottom_data,
    const T* top_data, const T* scale, const T* top_diff,
    const int num, const int height, const int width, const int channels,
    const int size, const T negative_beta, const T cache_ratio,
    T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local channel offset
    int c = index % channels;
    int pre_pad = size / 2;
    T accum_ratio = 0;
    for (int i = -pre_pad; i < size - pre_pad; ++i) {
      if (c + i >= 0 && c + i < channels) {
        accum_ratio += top_diff[index + i] * top_data[index + i] /
            scale[index + i];
      }
    }
    bottom_diff[index] = top_diff[index] * pow(scale[index], negative_beta) -
                         cache_ratio * bottom_data[index] * accum_ratio;
  }
}
}  // namespace

template<>
bool LRNOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);

  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const float* Xdata = X.data<float>();
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
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

  int n_threads = N * H * W;
  LRNFillScaleNCHW<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      n_threads, Xdata, N, C, H, W, size_, alpha_ / size_, bias_, scale_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  n_threads = X.numel();
  LRNComputeOutput<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      n_threads, Xdata, scale_data, -beta_, Ydata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template<>
bool LRNOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);

  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const float* Xdata = X.data<float>();
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
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

  int n_threads = X.numel();
  LRNFillScaleNHWC<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      n_threads, Xdata, N, H, W, C, size_, alpha_ / size_, bias_, scale_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  LRNComputeOutput<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      n_threads, Xdata, scale_data, -beta_, Ydata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool LRNGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);

  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.numel(), Y.numel());
  DCHECK_EQ(X.numel(), dY.numel());
  auto* dX = Output(0, X.sizes(), at::dtype<float>());

  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();
  int n_threads = N * H * W;
  LRNFillScaleNCHW<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      n_threads, Xdata, N, C, H, W, size_, alpha_ / size_, bias_, scale_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();

  LRNComputeDiffNCHW<float><<<CAFFE_GET_BLOCKS(n_threads),
                              CAFFE_CUDA_NUM_THREADS,
                              0, context_.cuda_stream()>>>(
      n_threads, Xdata, Ydata, scale_data, dYdata, N, C, H, W, size_, -beta_,
      2.f * alpha_ * beta_ / size_, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool LRNGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);

  DCHECK_EQ(X.dim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const float* Xdata = X.data<float>();
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.numel(), Y.numel());
  DCHECK_EQ(X.numel(), dY.numel());
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);

  float* scale_data = scale_->template mutable_data<float>();
  int n_threads = X.numel();
  LRNFillScaleNHWC<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      n_threads, Xdata, N, H, W, C, size_, alpha_ / size_, bias_, scale_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  LRNComputeDiffNHWC<float>
      <<<CAFFE_GET_BLOCKS(X.numel()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          X.numel(),
          X.data<float>(),
          Y.data<float>(),
          scale_data,
          dY.data<float>(),
          X.dim32(0),
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          size_,
          -beta_,
          2.f * alpha_ * beta_ / size_,
          dX->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}


REGISTER_CUDA_OPERATOR(LRN, LRNOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(LRNGradient, LRNGradientOp<float, CUDAContext>);

}  // namespace caffe2
