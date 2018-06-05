#include "caffe2/operators/ensure_cpu_output_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {
// From CUDA Context, takes either CUDA or CPU tensor as input, and produce
// TensorCPU
REGISTER_CUDA_OPERATOR(EnsureCPUOutput, EnsureCPUOutputOp<CUDAContext>);
} // namespace caffe2
