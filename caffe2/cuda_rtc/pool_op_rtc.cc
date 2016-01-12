// TODO: reduce the apparent redundancy of all the code below.
#include <cfloat>
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
const char kMaxPoolForwardNCHWSource[] = "                                   \n\
extern \"C\"                                                                 \n\
__global__ void %s(const float* bottom_data, float* top_data) {              \n\
  const int nthreads = %d;                                                   \n\
  const int channels = %d;                                                   \n\
  const int height = %d;                                                     \n\
  const int width = %d;                                                      \n\
  const int pooled_height = %d;                                              \n\
  const int pooled_width = %d;                                               \n\
  const int kernel_h = %d;                                                   \n\
  const int kernel_w = %d;                                                   \n\
  const int stride_h = %d;                                                   \n\
  const int stride_w = %d;                                                   \n\
  const int pad_t = %d;                                                      \n\
  const int pad_l = %d;                                                      \n\
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
       index < nthreads; index += blockDim.x * gridDim.x) {                  \n\
    int pw = index %% pooled_width;                                          \n\
    int ph = (index / pooled_width) %% pooled_height;                        \n\
    int c = (index / (pooled_width * pooled_height)) %% channels;            \n\
    int n = index / (pooled_width * pooled_height * channels);               \n\
    int hstart = ph * stride_h - pad_t;                                      \n\
    int wstart = pw * stride_w - pad_l;                                      \n\
    int hend = min(hstart + kernel_h, height);                               \n\
    int wend = min(wstart + kernel_w, width);                                \n\
    hstart = max(hstart, 0);                                                 \n\
    wstart = max(wstart, 0);                                                 \n\
    float maxval = -1.0e37f;                                                 \n\
    const float* bdata_offset = bottom_data + n * channels * height * width; \n\
    for (int h = hstart; h < hend; ++h) {                                    \n\
      for (int w = wstart; w < wend; ++w) {                                  \n\
        maxval = fmaxf(                                                      \n\
            bdata_offset[c * height * width + h * width + w], maxval);       \n\
      }                                                                      \n\
    }                                                                        \n\
    top_data[index] = maxval;                                                \n\
  }                                                                          \n\
}                                                                            \n\
";

// The max pool forward function, with parameters written in const int.
const char kMaxPoolBackwardNCHWSource[] = "                                   \n\
extern \"C\" \n\
__global__ void %s( \n\
    const float* const bottom_data, const float* const top_data, \n\
    const float* const top_diff, float* const bottom_diff) { \n\
  const int nthreads = %d; \n\
  const int num = %d; \n\
  const int channels = %d; \n\
  const int height = %d; \n\
  const int width = %d; \n\
  const int pooled_height = %d; \n\
  const int pooled_width = %d; \n\
  const int kernel_h = %d; \n\
  const int kernel_w = %d; \n\
  const int stride_h = %d; \n\
  const int stride_w = %d; \n\
  const int pad_t = %d; \n\
  const int pad_l = %d; \n\
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
       index < nthreads; index += blockDim.x * gridDim.x) {                  \n\
    const int w = index % width + pad_l; \n\
    const int h = (index / width) %% height + pad_t; \n\
    const int c = (index / width / height) %% channels; \n\
    const int n = index / width / height / channels; \n\
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1; \n\
    const int phend = min(h / stride_h + 1, pooled_height); \n\
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1; \n\
    const int pwend = min(w / stride_w + 1, pooled_width); \n\
    const int top_offset = \n\
        (n * channels + c) * pooled_height * pooled_width; \n\
    bottom_diff[index] = 0; \n\
    for (int ph = phstart; ph < phend; ++ph) { \n\
      for (int pw = pwstart; pw < pwend; ++pw) { \n\
        int top_local_offset = top_offset + ph * pooled_width + pw; \n\
        if (bottom_data[index] == top_data[top_local_offset]) { \n\
          bottom_diff[index] += top_diff[top_local_offset]; \n\
        } \n\
      } \n\
    } \n\
  } \n\
} \n\
";


class MaxPoolRTCFunction : public CudaRTCFunction<MaxPoolRTCFunction> {
 public:
  MaxPoolRTCFunction() : CudaRTCFunction(), name_(GetUniqueName()) {}

  template <typename... Args>
  string KernelName(Args... args) { return name_; }

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
  string KernelName(Args... args) { return name_; }

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
  sprintf(buffer, kMaxPoolForwardNCHWSource, name_.c_str(), output_size,
          channels, height, width,
          pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w,
          pad_t, pad_l);
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
  sprintf(buffer, kMaxPoolBackwardNCHWSource, name_.c_str(), output_size,
          num, channels, height, width, pooled_height, pooled_width, kernel_h,
          kernel_w, stride_h, stride_w, pad_t, pad_l);
  return string(buffer);
}

}  // namespace


class MaxPoolRTCOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  MaxPoolRTCOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws) {}
  ~MaxPoolRTCOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    ConvPoolOpBase::SetOutputSize(X, Y, X.dim(1));

    if (input_dims_ != X.dims()) {
      // recompile
      CAFFE_VLOG(1) << "MaxPool RTC recompiling";
      func_.Compile(Y->size(), X.dim(1), X.dim(2), X.dim(3), Y->dim(2), Y->dim(3),
                    kernel_h_, kernel_w_, stride_h_, stride_w_, pad_t_, pad_l_);
      input_dims_ = X.dims();
    }
    // Carry out the pooling computation.
    func_.Launch(CAFFE_GET_BLOCKS(Y->size()), 1, 1, CAFFE_CUDA_NUM_THREADS,
                 1, 1, 0, device_context_.cuda_stream(),
                 X.data<float>(), Y->mutable_data<float>());
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    CAFFE_LOG_FATAL << "Not implemented.";
    return false;
  }

 private:
  MaxPoolRTCFunction func_;
  vector<int> input_dims_;
  // Input: X
  // Output: Y
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(MaxPoolRTCOp);
};

class MaxPoolGradientRTCOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  MaxPoolGradientRTCOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws) {}
  ~MaxPoolGradientRTCOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dY = Input(2);
    CAFFE_CHECK_EQ(dY.ndim(), 4);
    auto* dX = Output(0);
    dX->ReshapeLike(X);
    ConvPoolOpBase<CUDAContext>::ComputePads(X.dim(2), X.dim(3));
    if (input_dims_ != X.dims()) {
      CAFFE_VLOG(1) << "MaxPoolGradient RTC recompiling";
      func_.Compile(X.size(), X.dim(0), X.dim(1), X.dim(2), X.dim(3), dY.dim(2),
                    dY.dim(3), kernel_h_, kernel_w_, stride_h_, stride_w_,
                    pad_t_, pad_l_);
      input_dims_ = X.dims();
    }
    func_.Launch(CAFFE_GET_BLOCKS(X.size()), 1, 1, CAFFE_CUDA_NUM_THREADS, 1, 1,
                 0, device_context_.cuda_stream(),
                 X.data<float>(), Y.data<float>(), dY.data<float>(),
                 dX->mutable_data<float>());
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    CAFFE_LOG_FATAL << "Not implemented.";
    return false;
  }

 private:
  MaxPoolGradientRTCFunction func_;
  vector<int> input_dims_;
  // Input: X, Y, dY
  // Output: dX
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(MaxPoolGradientRTCOp);
};

namespace {
REGISTER_CUDA_OPERATOR_WITH_ENGINE(MaxPool, NVRTC, MaxPoolRTCOp);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(MaxPoolGradient, NVRTC,
                                   MaxPoolGradientRTCOp);
}  // namespace
}  // namespace caffe2
