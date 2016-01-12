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
        int idx = c * height * width + h * width + w;                        \n\
        if (bdata_offset[idx] > maxval) {                                    \n\
          maxval = bdata_offset[idx];                                        \n\
        }                                                                    \n\
      }                                                                      \n\
    }                                                                        \n\
    top_data[index] = maxval;                                                \n\
  }                                                                          \n\
}                                                                            \n\
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

namespace {
REGISTER_CUDA_OPERATOR_WITH_ENGINE(MaxPool, NVRTC, MaxPoolRTCOp);
}  // namespace
}  // namespace caffe2
