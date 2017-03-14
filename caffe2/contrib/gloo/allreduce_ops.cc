#include "allreduce_ops.h"

#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"

namespace caffe2 {
namespace gloo {

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingFull() {
  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  auto pointers = getPointers();
  auto size = Output(0)->size();
  algorithm_.reset(new ::gloo::AllreduceRing<T>(context, pointers, size));
}

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingChunked() {
  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  auto pointers = getPointers();
  auto size = Output(0)->size();
  algorithm_.reset(
      new ::gloo::AllreduceRingChunked<T>(context, pointers, size));
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Allreduce,
    GLOO,
    AllreduceOp<float, CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
