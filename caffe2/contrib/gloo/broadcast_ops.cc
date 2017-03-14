#include "broadcast_ops.h"

#include "gloo/broadcast_one_to_all.h"

namespace caffe2 {
namespace gloo {

template <class Context>
void BroadcastOp<Context>::initialize() {
  auto& input = Input(INPUT);
  auto* output = Output(OUTPUT);
  CAFFE_ENFORCE_EQ(input.raw_data(), output->raw_data());

  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  if (output->template IsType<float>()) {
    auto ptr = output->template mutable_data<float>();
    algorithm_.reset(new ::gloo::BroadcastOneToAll<float>(
        context, {ptr}, output->size(), root_));
  } else if (output->template IsType<long>()) {
    auto ptr = output->template mutable_data<long>();
    algorithm_.reset(new ::gloo::BroadcastOneToAll<long>(
        context, {ptr}, output->size(), root_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", output->meta().name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Broadcast, GLOO, BroadcastOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
