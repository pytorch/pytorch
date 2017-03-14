#pragma once

#include <algorithm>

#include "caffe2/core/operator.h"

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace caffe2 {
namespace gloo {

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
    algorithm_->run();
    return true;
  }

 protected:
  void initialize();

  const int root_;
  std::once_flag once_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;

  INPUT_TAGS(COMM, INPUT);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace gloo
} // namespace caffe2
