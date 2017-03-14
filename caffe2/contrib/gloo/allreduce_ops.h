#pragma once

#include <algorithm>

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace caffe2 {
namespace gloo {

template <typename T, class Context>
class AllreduceOp final : public Operator<Context> {
  enum Mode { RING_FULL, RING_CHUNKED };

 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  virtual ~AllreduceOp() {}

  bool RunOnDevice() override {
    std::call_once(once_, [&] { initialize(); });
    algorithm_->run();
    return true;
  }

 protected:
  void initialize() {
    Mode mode = RING_FULL;
    auto bytes = Input(INPUT).nbytes();
    if (bytes < 4096) {
      mode = RING_FULL;
    } else {
      mode = RING_CHUNKED;
    }

    switch (mode) {
      case RING_FULL:
        initializeRingFull();
        return;
      case RING_CHUNKED:
        initializeRingChunked();
        return;
    }

    CAFFE_ENFORCE(false, "Unreachable code");
  }

  void initializeRingFull();
  void initializeRingChunked();

  std::once_flag once_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;

  INPUT_TAGS(COMM, INPUT);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace gloo
} // namespace caffe2
