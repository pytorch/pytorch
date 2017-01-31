#pragma once

#include <algorithm>

#include "caffe2/core/operator.h"

#include "fbcollective/broadcast_one_to_all.h"
#include "fbcollective/context.h"

namespace caffe2 {
namespace fbcollective {

template <typename T, class Context>
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
    CAFFE_ENFORCE_EQ(input.template data<T>(), output->template data<T>());

    const auto& context =
        OperatorBase::Input<std::shared_ptr<::fbcollective::Context>>(COMM);
    T* ptr = output->template mutable_data<T>();
    algorithm_.reset(new ::fbcollective::BroadcastOneToAll<T>(
        context, ptr, output->size(), root_));
  }

  const int root_;
  std::once_flag once_;
  std::unique_ptr<::fbcollective::Algorithm> algorithm_;

  INPUT_TAGS(COMM, INPUT);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace fbcollective
} // namespace caffe2
