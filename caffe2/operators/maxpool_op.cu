#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/maxpool_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void MaxPoolForwardNCHW(const int nthreads, const T* bottom_data,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data,
    int* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T maxval = -FLT_MAX;
    int maxidx = -1;
    const T* bdata_offset = bottom_data + n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = c * height * width + h * width + w;
        if (bdata_offset[idx] > maxval) {
          maxidx = idx;
          maxval = bdata_offset[idx];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}

template <typename T>
__global__ void MaxPoolForwardNHWC(const int nthreads, const T* bottom_data,
    const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data,
    int* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T maxval = -FLT_MAX;
    int maxidx = -1;
    const T* bdata_offset = bottom_data + n * height * width * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (h * width + w) * channels + c;
        if (bdata_offset[idx] > maxval) {
          maxidx = idx;
          maxval = bdata_offset[idx];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}

template <typename T>
__global__ void MaxPoolBackward(
    const int nthreads, const T* top_diff, const int* mask,
    const int top_offset, const int bottom_offset, T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int image_id = (index / top_offset);
    atomicAdd(bottom_diff + image_id * bottom_offset + mask[index],
              top_diff[index]);
  }
}

}  // namespace

template <>
bool MaxPoolOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto* maxid = Output(1);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim(1));
  maxid->ReshapeLike(*Y);
  int output_size = Y->size();
  MaxPoolForwardNCHW<float><<<CAFFE_GET_BLOCKS(output_size),
                              CAFFE_CUDA_NUM_THREADS,
                              0, device_context_.cuda_stream()>>>(
      output_size, X.data<float>(), X.dim(1), X.dim(2), X.dim(3),
      Y->dim(2), Y->dim(3), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, Y->mutable_data<float>(), maxid->mutable_data<int>());
  return true;
}

template <>
bool MaxPoolOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto* maxid = Output(1);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim(3));
  maxid->ReshapeLike(*Y);
  int output_size = Y->size();
  MaxPoolForwardNHWC<float><<<CAFFE_GET_BLOCKS(output_size),
                              CAFFE_CUDA_NUM_THREADS,
                              0, device_context_.cuda_stream()>>>(
      output_size, X.data<float>(), X.dim(1), X.dim(2), X.dim(3),
      Y->dim(1), Y->dim(2), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, Y->mutable_data<float>(), maxid->mutable_data<int>());
  return true;
}


template <>
bool MaxPoolGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto& maxid = Input(2);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ReshapeLike(X);
  math::Set<float, CUDAContext>(
      X.size(), 0, dX->mutable_data<float>(), &device_context_);
  MaxPoolBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                           CAFFE_CUDA_NUM_THREADS,
                           0, device_context_.cuda_stream()>>>(
      dY.size(), dY.data<float>(), maxid.data<int>(), dY.size() / dY.dim(0),
      X.size() / X.dim(0), dX->mutable_data<float>());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(MaxPool, MaxPoolOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MaxPoolGradient, MaxPoolGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
