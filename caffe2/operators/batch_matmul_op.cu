#include "caffe2/operators/batch_matmul_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <>
bool BatchMatMulOp<CUDAContext, DefaultEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR(BatchMatMul, BatchMatMulOp<CUDAContext>);


#if !defined(USE_ROCM)

template <>
bool BatchMatMulOp<CUDAContext, TensorCoreEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    BatchMatMul,
    TENSORCORE,
    BatchMatMulOp<CUDAContext, TensorCoreEngine>);

#endif

} // namespace caffe2
