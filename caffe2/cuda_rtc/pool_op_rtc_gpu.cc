#include <cstdio>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pool_op.h"
#include "caffe2/cuda_rtc/common_rtc.h"

namespace caffe2 {
namespace {
class AveragePool {};
class MaxPool {};
}  // namespace

namespace {

// The max pool forward function, with parameters written in const int.
const char kMaxPoolForwardNCHWSource[] = R"(
extern "C"
__global__ void %s(const float* bottom_data, float* top_data) {
  const int nthreads = %d;
  const int channels = %d;
  const int height = %d;
  const int width = %d;
  const int pooled_height = %d;
  const int pooled_width = %d;
  const int kernel_h = %d;
  const int kernel_w = %d;
  const int stride_h = %d;
  const int stride_w = %d;
  const int pad_t = %d;
  const int pad_l = %d;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < nthreads; index += blockDim.x * gridDim.x) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int c = (index / (pooled_width * pooled_height)) %% channels;
    int n = index / (pooled_width * pooled_height * channels);
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -1.0e37f;
    const float* bdata_offset = bottom_data + n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        maxval = fmaxf(
            bdata_offset[c * height * width + h * width + w], maxval);
      }
    }
    top_data[index] = maxval;
  }
}
)";

// The max pool forward function, with parameters written in const int.
const char kMaxPoolBackwardNCHWSource[] = R"(
extern "C"
__global__ void %s(
    const float* const bottom_data, const float* const top_data,
    const float* const top_diff, float* const bottom_diff) {
  const int nthreads = %d;
  const int num = %d;
  const int channels = %d;
  const int height = %d;
  const int width = %d;
  const int pooled_height = %d;
  const int pooled_width = %d;
  const int kernel_h = %d;
  const int kernel_w = %d;
  const int stride_h = %d;
  const int stride_w = %d;
  const int pad_t = %d;
  const int pad_l = %d;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < nthreads; index += blockDim.x * gridDim.x) {
    const int w = index %% width + pad_l;
    const int h = (index / width) %% height + pad_t;
    const int c = (index / width / height) %% channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int top_offset =
        (n * channels + c) * pooled_height * pooled_width;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int top_local_offset = top_offset + ph * pooled_width + pw;
        if (bottom_data[index] == top_data[top_local_offset]) {
          bottom_diff[index] += top_diff[top_local_offset];
        }
      }
    }
  }
}
)";


class MaxPoolRTCFunction : public CudaRTCFunction<MaxPoolRTCFunction> {
 public:
  MaxPoolRTCFunction() : CudaRTCFunction(), name_(GetUniqueName()) {}

  template <typename... Args>
  string KernelName(Args... /*args*/) {
    return name_;
  }

  template <typename... Args>
  string GetSource(Args... args);

 private:
  string name_;
};

class MaxPoolGradientRTCFunction
    : public CudaRTCFunction<MaxPoolGradientRTCFunction> {
 public:
  MaxPoolGradientRTCFunction() : CudaRTCFunction(), name_(GetUniqueName()) {}

  template <typename... Args>
  string KernelName(Args... /*args*/) {
    return name_;
  }

  template <typename... Args>
  string GetSource(Args... args);

 private:
  string name_;
};


template <>
string MaxPoolRTCFunction::GetSource(
    const int output_size,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l) {
  char buffer[65536];
  int nbytes = snprintf(
      buffer, 65536, kMaxPoolForwardNCHWSource, name_.c_str(), output_size,
      channels, height, width, pooled_height, pooled_width, kernel_h, kernel_w,
      stride_h, stride_w, pad_t, pad_l);
  DCHECK_GE(nbytes, 0);
  DCHECK_LT(nbytes, 65536);
  return string(buffer);
}

template <>
string MaxPoolGradientRTCFunction::GetSource(
    const int output_size,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l) {
  char buffer[65536];
  int nbytes = snprintf(
      buffer, 65536, kMaxPoolBackwardNCHWSource, name_.c_str(), output_size,
      num, channels, height, width, pooled_height, pooled_width, kernel_h,
      kernel_w, stride_h, stride_w, pad_t, pad_l);
  DCHECK_GE(nbytes, 0);
  DCHECK_LT(nbytes, 65536);
  return string(buffer);
}

}  // namespace


class MaxPoolRTCOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  MaxPoolRTCOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Currently only NCHW is supported.");
  }
  ~MaxPoolRTCOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));

    if (input_dims_ != X.sizes()) {
      // recompile
      VLOG(1) << "MaxPool RTC recompiling";
      CAFFE_ENFORCE_LT(Y->numel(), std::numeric_limits<int>::max());
      func_.Compile(
          static_cast<int>(Y->numel()),
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          Y->dim32(2),
          Y->dim32(3),
          kernel_h(),
          kernel_w(),
          stride_h(),
          stride_w(),
          pad_t(),
          pad_l());
      input_dims_ = X.sizes().vec();
    }
    // Carry out the pooling computation.
    func_.Launch(
        CAFFE_GET_BLOCKS(Y->numel()),
        1,
        1,
        CAFFE_CUDA_NUM_THREADS,
        1,
        1,
        0,
        context_.cuda_stream(),
        X.data<float>(),
        Y->mutable_data<float>());
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    LOG(FATAL) << "Not implemented.";
    return false;
  }

 private:
  MaxPoolRTCFunction func_;
  vector<int64_t> input_dims_;
};

class MaxPoolGradientRTCOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  MaxPoolGradientRTCOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Currently only NCHW is supported.");
  }
  ~MaxPoolGradientRTCOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dY = Input(2);
    CAFFE_ENFORCE_EQ(dY.dim(), 4);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(2), X.dim32(3)});
    if (input_dims_ != X.sizes()) {
      VLOG(1) << "MaxPoolGradient RTC recompiling";
      CAFFE_ENFORCE_LT(X.numel(), std::numeric_limits<int>::max());
      func_.Compile(
          static_cast<int>(X.numel()),
          X.dim32(0),
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          dY.dim32(2),
          dY.dim32(3),
          kernel_h(),
          kernel_w(),
          stride_h(),
          stride_w(),
          pad_t(),
          pad_l());
      input_dims_ = X.sizes().vec();
    }
    func_.Launch(
        CAFFE_GET_BLOCKS(X.numel()),
        1,
        1,
        CAFFE_CUDA_NUM_THREADS,
        1,
        1,
        0,
        context_.cuda_stream(),
        X.data<float>(),
        Y.data<float>(),
        dY.data<float>(),
        dX->mutable_data<float>());
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    LOG(FATAL) << "Not implemented.";
    return false;
  }

 private:
  MaxPoolGradientRTCFunction func_;
  vector<int64_t> input_dims_;
};

namespace {
REGISTER_CUDA_OPERATOR_WITH_ENGINE(MaxPool, NVRTC, MaxPoolRTCOp);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(MaxPoolGradient, NVRTC,
                                   MaxPoolGradientRTCOp);
}  // namespace
}  // namespace caffe2
