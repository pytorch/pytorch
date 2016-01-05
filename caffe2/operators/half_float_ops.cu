#include <cuda_fp16.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace {
__global__ void FloatToHalfKernel(const int N, const float* X, half* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __float2half(X[i]);
  }
}

__global__ void HalfToFloatKernel(const int N, const half* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __half2float(X[i]);
  }
}
}

class FloatToHalfCUDA : public Operator<CUDAContext> {
 public:
  FloatToHalfCUDA(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws) {}
  ~FloatToHalfCUDA() {}

  bool RunOnDevice() {
    auto& X = Input(0);
    auto* Y = Output(0);
    Y->ReshapeLike(X);
    FloatToHalfKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                        0, device_context_.cuda_stream()>>>(
      X.size(), X.data<float>(),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FloatToHalfCUDA);
};

class HalfToFloatCUDA : public Operator<CUDAContext> {
 public:
  HalfToFloatCUDA(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws) {}
  ~HalfToFloatCUDA() {}

  bool RunOnDevice() {
    auto& X = Input(0);
    auto* Y = Output(0);
    Y->ReshapeLike(X);
    HalfToFloatKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                        0, device_context_.cuda_stream()>>>(
      X.size(), reinterpret_cast<const half*>(X.data<float16>()),
      Y->mutable_data<float>());
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(HalfToFloatCUDA);
};

namespace {
REGISTER_CUDA_OPERATOR(FloatToHalf, FloatToHalfCUDA);
REGISTER_CUDA_OPERATOR(HalfToFloat, HalfToFloatCUDA);

struct GetFloatToHalfGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    return SingleGradientDef(
        "HalfToFloat", "",
        vector<string>{GO(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(FloatToHalf, GetFloatToHalfGradient);

struct GetHalfToFloatGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    return SingleGradientDef(
        "FloatToHalf", "",
        vector<string>{GO(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(HalfToFloat, GetHalfToFloatGradient);

}  // namespace
}  // namespace caffe2