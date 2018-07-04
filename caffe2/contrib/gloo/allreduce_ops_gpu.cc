#include "allreduce_ops.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"

#include <gloo/cuda_allreduce_halving_doubling.h>
#include <gloo/cuda_allreduce_ring.h>
#include <gloo/cuda_allreduce_ring_chunked.h>
#include <gloo/types.h>

namespace caffe2 {
namespace gloo {

namespace {

// Decides on using GPUDirect based on device support.
template <template <typename T, typename W> class A, typename T>
std::unique_ptr<::gloo::Algorithm> initializeAlgorithm(
    bool gpu_direct_,
    std::shared_ptr<::gloo::Context> context,
    std::vector<T*> ptrs,
    size_t size) {
  if (gpu_direct_) {
    if (context->getDevice()->hasGPUDirect()) {
      return std::unique_ptr<::gloo::Algorithm>(
        new A<T, ::gloo::CudaDeviceWorkspace<T>>(context, ptrs, size));
    } else {
      LOG(WARNING)
        << "GPUDirect not available; "
        << "Gloo communication will go through system memory instead.";
    }
  }

  return std::unique_ptr<::gloo::Algorithm>(
    new A<T, ::gloo::CudaHostWorkspace<T>>(context, ptrs, size));
}

} // namespace

template <class Context>
void AllreduceOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceHalvingDoubling, float>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<float>(),
        init_.size);
  } else if (init_.template IsType<float16>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceHalvingDoubling, ::gloo::float16>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size);
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeRingFull() {
  if (init_.template IsType<float>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceRing, float>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<float>(),
        init_.size);
  } else if (init_.template IsType<float16>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceRing, ::gloo::float16>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size);
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeRingChunked() {
  if (init_.template IsType<float>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceRingChunked, float>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<float>(),
        init_.size);
  } else if (init_.template IsType<float16>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceRingChunked, ::gloo::float16>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size);
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(Allreduce, GLOO, AllreduceOp<CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
