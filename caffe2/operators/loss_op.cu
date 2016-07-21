#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/loss_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void ALGKernel(const int N, const T* dY, T* dX) {
  const T value = (*dY) / N; 
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = value;
  }
}
}  // namespace

class AveragedLossGradientGPUSpecialization final
    : public Operator<CUDAContext> {
 public:
  AveragedLossGradientGPUSpecialization(
      const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}
  ~AveragedLossGradientGPUSpecialization() {}
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dY = Input(1);
    DCHECK_EQ(dY.size(), 1);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    ALGKernel<float><<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                       0, context_.cuda_stream()>>>(
        X.size(), dY.data<float>(), dX->mutable_data<float>());
    return true;
  }
};

namespace {
REGISTER_CUDA_OPERATOR(AveragedLoss, AveragedLoss<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(AveragedLossGradient,
                       AveragedLossGradientGPUSpecialization);
}  // namespace
}  // namespace caffe2

