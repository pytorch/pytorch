#include "allreduce_ops.h"

#include "caffe2/core/context_gpu.h"

#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"

namespace caffe2 {
namespace gloo {

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingFull() {
  algorithm_.reset(new ::gloo::CudaAllreduceRing<T>(
      init_.context, init_.outputs, init_.size));
}

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingChunked() {
  algorithm_.reset(new ::gloo::CudaAllreduceRingChunked<T>(
      init_.context, init_.outputs, init_.size));
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    Allreduce,
    GLOO,
    AllreduceOp<float, CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
