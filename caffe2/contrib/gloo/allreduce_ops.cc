#include "allreduce_ops.h"

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/allreduce_ring.h>
#include <gloo/allreduce_ring_chunked.h>

namespace caffe2 {
namespace gloo {

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeHalvingDoubling() {
  algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<T>(
      init_.context, init_.outputs, init_.size));
}

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingFull() {
  algorithm_.reset(
      new ::gloo::AllreduceRing<T>(init_.context, init_.outputs, init_.size));
}

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingChunked() {
  algorithm_.reset(new ::gloo::AllreduceRingChunked<T>(
      init_.context, init_.outputs, init_.size));
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Allreduce,
    GLOO,
    AllreduceOp<float, CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
