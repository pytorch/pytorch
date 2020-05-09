#include "caffe2/contrib/gloo/allreduce_ops.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"

#include <gloo/cuda_allreduce_bcube.h>
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

/**
 * This is a helper function which attempts to get a base value depending on the
 * # of nodes. Larger the base the better performance (up to 4) is what we have
 * observed in gloo benchmarks. At the moment bcube works only if # nodes = base
 * ^ x. Where x is some constant. So, if # node don't match our expectation
 * simply return -1. This will indicate caller to switch to another algorithm
 * like halving-doubling.
 */
static int getAllrduceBcubeBase(int nodes) {
  auto getExponent = [](int n, int b) -> int {
    float lg2n = log2(n);
    float lg2b = log2(b);
    return ceil(lg2n / lg2b);
  };
  auto baseCheck = [&](int n, int b) -> bool {
    int e = getExponent(n, b);
    return n == pow(b, e);
  };
  for (const auto base : {6, 5, 4, 3, 2}) {
    if (baseCheck(nodes, base)) {
      return base;
    }
    /*
     * Base could work if # nodes is multiple of the base yet smaller than
     * base^2
     */
    if (nodes < base * base && 0 == nodes % base) {
      return base;
    }
  }
  return -1;
}

} // namespace

template <class Context>
void AllreduceOp<Context>::initializeBcube() {
  int base = getAllrduceBcubeBase(init_.size);
  if (-1 == base) {
    return initializeHalvingDoubling();
  }
  init_.context->base = base;
  if (init_.template IsType<float>()) {
    algorithm_ = initializeAlgorithm<::gloo::CudaAllreduceBcube, float>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<float>(),
        init_.size);
  } else if (init_.template IsType<at::Half>()) {
    algorithm_ =
        initializeAlgorithm<::gloo::CudaAllreduceBcube, ::gloo::float16>(
            gpu_direct_,
            init_.context,
            init_.template getOutputs<::gloo::float16>(),
            init_.size);
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_ =
      initializeAlgorithm<::gloo::CudaAllreduceHalvingDoubling, float>(
        gpu_direct_,
        init_.context,
        init_.template getOutputs<float>(),
        init_.size);
  } else if (init_.template IsType<at::Half>()) {
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
  } else if (init_.template IsType<at::Half>()) {
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
  } else if (init_.template IsType<at::Half>()) {
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
