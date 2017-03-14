#include "allreduce_ops.h"

#include "caffe2/core/context_gpu.h"

#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"

namespace caffe2 {
namespace gloo {

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingFull() {
  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  auto pointers = getPointers();
  auto size = Output(0)->size();
  algorithm_.reset(new ::gloo::CudaAllreduceRing<T>(context, pointers, size));
}

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingChunked() {
  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  auto pointers = getPointers();
  auto size = Output(0)->size();
  algorithm_.reset(
      new ::gloo::CudaAllreduceRingChunked<T>(context, pointers, size));
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    Allreduce,
    GLOO,
    AllreduceOp<float, CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
