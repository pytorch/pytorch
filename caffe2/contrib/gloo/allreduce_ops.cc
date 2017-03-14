#include "allreduce_ops.h"

#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"

namespace caffe2 {
namespace gloo {

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingFull() {
  auto& input = Input(INPUT);
  auto* output = Output(OUTPUT);
  CAFFE_ENFORCE_EQ(input.template data<T>(), output->template data<T>());

  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  std::vector<T*> ptrs = {output->template mutable_data<T>()};
  algorithm_.reset(new ::gloo::AllreduceRing<T>(context, ptrs, output->size()));
}

template <typename T, class Context>
void AllreduceOp<T, Context>::initializeRingChunked() {
  auto& input = Input(INPUT);
  auto* output = Output(OUTPUT);
  CAFFE_ENFORCE_EQ(input.template data<T>(), output->template data<T>());

  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  std::vector<T*> ptrs = {output->template mutable_data<T>()};
  algorithm_.reset(
      new ::gloo::AllreduceRingChunked<T>(context, ptrs, output->size()));
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Allreduce,
    GLOO,
    AllreduceOp<float, CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
