#pragma once

#include <algorithm>

#include "caffe2/core/operator.h"

#include "fbcollective/broadcast_one_to_all.h"
#include "fbcollective/context.h"

namespace caffe2 {
namespace fbcollective {

template <class Context>
class BroadcastOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {}

  virtual ~BroadcastOp() {}

  bool RunOnDevice() override {
    std::call_once(once_, [&] { initialize(); });
    algorithm_->Run();
    return true;
  }

 protected:
  void initialize() {
    auto& input = Input(INPUT);
    auto* output = Output(OUTPUT);
    CAFFE_ENFORCE_EQ(input.raw_data(), output->raw_data());

    const auto& context =
        OperatorBase::Input<std::shared_ptr<::fbcollective::Context>>(COMM);
    if (output->template IsType<float>()) {
      auto ptr = output->template mutable_data<float>();
      algorithm_.reset(new ::fbcollective::BroadcastOneToAll<float>(
          context, ptr, output->size(), root_));
    } else if (output->template IsType<long>()) {
      auto ptr = output->template mutable_data<long>();
      algorithm_.reset(new ::fbcollective::BroadcastOneToAll<long>(
          context, ptr, output->size(), root_));
    } else {
      CAFFE_ENFORCE(false, "Unhandled type: ", output->meta().name());
    }
  }

  const int root_;
  std::once_flag once_;
  std::unique_ptr<::fbcollective::Algorithm> algorithm_;

  INPUT_TAGS(COMM, INPUT);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace fbcollective
} // namespace caffe2
