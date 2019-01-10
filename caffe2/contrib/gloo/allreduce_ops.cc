#include "allreduce_ops.h"

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/allreduce_ring.h>
#include <gloo/allreduce_ring_chunked.h>
#include <gloo/types.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void AllreduceOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::caffe2::float16>()) {
    algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<::gloo::float16>(
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
    algorithm_.reset(new ::gloo::AllreduceRing<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::caffe2::float16>()) {
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
  } else if (init_.template IsType<::caffe2::float16>()) {
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
