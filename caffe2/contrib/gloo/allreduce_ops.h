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

    // If any parameter has changed in between runs, the initialized
    // algorithm is invalid and cannot be used.
    update(current_);
    CAFFE_ENFORCE(current_ == init_, "Inputs/outputs have changed");

    algorithm_->run();
    return true;
  }

 protected:
  void initialize() {
    Mode mode = RING_FULL;
    auto bytes = Input(1).nbytes();

    // Pretty arbitrary threshold but seems to work well.
    // Logic for switching between algorithms in a topology
    // dependent manner will eventually move to Gloo itself.
    if (bytes < (256 * 1024)) {
      mode = RING_FULL;
    } else {
      mode = RING_CHUNKED;
    }

    // Store which inputs/outputs this instance initialized with
    update(init_);

    // Verify inputs == ouputs
    CAFFE_ENFORCE_EQ(init_.inputs.size(), init_.outputs.size());
    for (auto i = 0; i < init_.inputs.size(); i++) {
      CAFFE_ENFORCE_EQ(init_.inputs[i], init_.outputs[i]);
    }

    // Verify tensors all have same size
    size_t size = Input(1).size();
    for (auto i = 2; i < InputSize(); i++) {
      CAFFE_ENFORCE_EQ(Input(i).size(), size);
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

  // Captures the parameters passed to Gloo when first initialized.
  // An instance is updated every time this op runs and is compared
  // to the reference instance for equality. If any parameter has
  // changed from run to run, the initialized algorithm is invalid.
  struct GlooParameters {
    std::shared_ptr<::gloo::Context> context;
    std::vector<const T*> inputs;
    std::vector<T*> outputs;
    size_t size;

    bool operator==(GlooParameters const& other) const {
      return context == other.context && inputs == other.inputs &&
          outputs == other.outputs && size == other.size;
    }
  };

  void update(GlooParameters& params) {
    params.context = OperatorBase::Input<std::shared_ptr<::gloo::Context>>(0);
    params.inputs.resize(InputSize() - 1);
    params.outputs.resize(OutputSize());
    for (auto i = 0; i < params.inputs.size(); i++) {
      params.inputs[i] = Input(i + 1).template data<T>();
      params.outputs[i] = Output(i)->template mutable_data<T>();
    }
    params.size = Output(0)->size();
  }

  GlooParameters init_;
  GlooParameters current_;
};

} // namespace gloo
} // namespace caffe2
