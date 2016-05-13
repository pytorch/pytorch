#include "caffe2/core/common_gpu.h"

#ifdef CAFFE_HAS_CUDA_FP16

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

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    Y->ReshapeLike(X);
    FloatToHalfKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      X.size(), X.data<float>(),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(FloatToHalfCUDA);
};

class HalfToFloatCUDA : public Operator<CUDAContext> {
 public:
  HalfToFloatCUDA(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws) {}
  ~HalfToFloatCUDA() {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    Y->ReshapeLike(X);
    HalfToFloatKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      X.size(), reinterpret_cast<const half*>(X.data<float16>()),
      Y->mutable_data<float>());
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(HalfToFloatCUDA);
};

namespace {
REGISTER_CUDA_OPERATOR(FloatToHalf, FloatToHalfCUDA);
REGISTER_CUDA_OPERATOR(HalfToFloat, HalfToFloatCUDA);

OPERATOR_SCHEMA(FloatToHalf).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(HalfToFloat).NumInputs(1).NumOutputs(1);

class GetFloatToHalfGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "HalfToFloat", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(FloatToHalf, GetFloatToHalfGradient);

class GetHalfToFloatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "FloatToHalf", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(HalfToFloat, GetHalfToFloatGradient);

}  // namespace
}  // namespace caffe2

#endif // CAFFE_HAS_CUDA_FP16
