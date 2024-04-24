#include "allreduce_ops.h"

#include <math.h>

#include <gloo/allreduce_bcube.h>
#include <gloo/allreduce_halving_doubling.h>
#include <gloo/allreduce_ring.h>
#include <gloo/allreduce_ring_chunked.h>
#include <gloo/types.h>

namespace {
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

namespace caffe2 {
namespace gloo {

template <class Context>
void AllreduceOp<Context>::initializeBcube() {
  int base = getAllrduceBcubeBase(init_.size);
  if (-1 == base) {
    return initializeHalvingDoubling();
  }
  init_.context->base = base;
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceBcube<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::at::Half>()) {
    algorithm_.reset(new ::gloo::AllreduceBcube<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::at::Half>()) {
    algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

// Used outside of the translation unit
template void AllreduceOp<CPUContext>::initializeHalvingDoubling();

template <class Context>
void AllreduceOp<Context>::initializeRingFull() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceRing<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::at::Half>()) {
    algorithm_.reset(new ::gloo::AllreduceRing<::gloo::float16>(
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
    algorithm_.reset(new ::gloo::AllreduceRingChunked<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::at::Half>()) {
    algorithm_.reset(new ::gloo::AllreduceRingChunked<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Allreduce, GLOO, AllreduceOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
