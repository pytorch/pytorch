#pragma once

#include <algorithm>

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include "fbcollective/allreduce_ring.h"
#include "fbcollective/allreduce_ring_chunked.h"
#include "fbcollective/context.h"

namespace caffe2 {
namespace fbcollective {

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
    algorithm_->Run();
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

  void initializeRingFull() {
    auto& input = Input(INPUT);
    auto* output = Output(OUTPUT);
    CAFFE_ENFORCE_EQ(input.template data<T>(), output->template data<T>());

    const auto& context =
        OperatorBase::Input<std::shared_ptr<::fbcollective::Context>>(COMM);
    std::vector<T*> ptrs = {output->template mutable_data<T>()};
    algorithm_.reset(
        new ::fbcollective::AllreduceRing<T>(context, ptrs, output->size()));
  }

  void initializeRingChunked() {
    auto& input = Input(INPUT);
    auto* output = Output(OUTPUT);
    CAFFE_ENFORCE_EQ(input.template data<T>(), output->template data<T>());

    auto& context =
        OperatorBase::Input<std::shared_ptr<::fbcollective::Context>>(COMM);
    std::vector<T*> ptrs = {output->template mutable_data<T>()};
    algorithm_.reset(new ::fbcollective::AllreduceRingChunked<T>(
        context, ptrs, output->size()));
  }

  std::once_flag once_;
  std::unique_ptr<::fbcollective::Algorithm> algorithm_;

  INPUT_TAGS(COMM, INPUT);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace fbcollective
} // namespace caffe2
