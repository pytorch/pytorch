#pragma once

#include "rebatching_queue.h"

#include "c10/util/irange.h"

namespace caffe2 {

using RebatchingQueuePtr = std::unique_ptr<RebatchingQueue>;

class CreateRebatchingQueueOp : public Operator<CPUContext> {
 public:
  CreateRebatchingQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<RebatchingQueuePtr>(0) =
        RebatchingQueuePtr(new RebatchingQueue(
            OperatorBase::GetSingleArgument<int>("capacity", 1),
            OperatorBase::GetSingleArgument<int>("num_blobs", 1)));
    return true;
  }
};

class EnqueueRebatchingQueueOp : public Operator<CPUContext> {
 public:
  EnqueueRebatchingQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        enqueueBatch_(
            OperatorBase::GetSingleArgument<bool>("enqueue_batch", false)) {}
  bool RunOnDevice() override {
    auto& queue = Inputs()[0]->template Get<RebatchingQueuePtr>();
    CHECK(queue);
    CAFFE_ENFORCE_EQ(InputSize(), queue->numBlobs() + 1);
    std::vector<const Tensor*> inputTensors;
    inputTensors.reserve(InputSize() - 1);
    for (const auto i : c10::irange(1, InputSize())) {
      inputTensors.push_back(&Input(i));
    }

    return enqueueBatch_ ? queue->enqueueMany(context_, inputTensors)
                         : queue->enqueueOne(context_, inputTensors);
  }

 private:
  const bool enqueueBatch_;
};

class DequeueRebatchingQueueOp : public Operator<CPUContext> {
 public:
  DequeueRebatchingQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        numElements_(OperatorBase::GetSingleArgument<int>("num_elements", 1)) {}

  bool RunOnDevice() override {
    auto& queue = Inputs()[0]->template Get<RebatchingQueuePtr>();
    CHECK(queue);

    std::vector<Tensor*> outputTensors;
    outputTensors.reserve(OutputSize());
    for (const auto i : c10::irange(OutputSize())) {
      outputTensors.push_back(Output(i));
    }

    return queue->dequeue(context_, numElements_, outputTensors);
  }

 private:
  int numElements_;
};

class CloseRebatchingQueueOp : public Operator<CPUContext> {
 public:
  CloseRebatchingQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(InputSize(), 1);
    auto& queue = Inputs()[0]->template Get<RebatchingQueuePtr>();
    CAFFE_ENFORCE(queue);
    queue->close();
    return true;
  }
};
} // namespace caffe2
