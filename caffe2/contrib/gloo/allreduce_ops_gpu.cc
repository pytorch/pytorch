#include "allreduce_ops.h"

#include "caffe2/core/context_gpu.h"

#include <gloo/cuda_allreduce_halving_doubling.h>
#include <gloo/cuda_allreduce_ring.h>
#include <gloo/cuda_allreduce_ring_chunked.h>
#include <gloo/types.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void AllreduceOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::CudaAllreduceHalvingDoubling<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<float16>()) {
    algorithm_.reset(new ::gloo::CudaAllreduceHalvingDoubling<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeRingFull() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::CudaAllreduceRing<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<float16>()) {
    algorithm_.reset(new ::gloo::CudaAllreduceRing<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeRingChunked() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::CudaAllreduceRingChunked<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<float16>()) {
    algorithm_.reset(new ::gloo::CudaAllreduceRingChunked<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(Allreduce, GLOO, AllreduceOp<CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
