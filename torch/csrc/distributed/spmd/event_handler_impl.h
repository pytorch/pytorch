#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/spmd/event_handler.h>
#include <torch/csrc/distributed/spmd/event_impl.h>


namespace torch {
namespace distributed {
namespace spmd {

// WIP DefaultTrigger
class TORCH_API DefaultTrigger : public EventHandler {
 public:
  using EventHandler::EventHandler;
  C10_NODISCARD std::vector<EventSchema> ingressEvents() const override;
  C10_NODISCARD std::vector<EventSchema> egressEvents() const override;
  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override;

 private:
  std::vector<std::shared_ptr<Future>> handlePrepareModule(
      c10::intrusive_ptr<PrepareModuleEvent> event);

  std::vector<std::shared_ptr<Future>> handlePreForward(
      c10::intrusive_ptr<PreForwardEvent> event);

  void autogradHook(size_t index);

  // keep grad accumulators alive
  std::vector<std::shared_ptr<torch::autograd::Node>> gradAccumulators_;
  std::vector<at::Tensor> params_;
  std::vector<std::shared_ptr<Future>> gradReadyFutures_;
};

// WIP DefaultBucketer
// TODO:
// 1. split this into BucketIndexer and BucketAllocator
// 2. bucketing gradients into larger tensors
class TORCH_API DefaultBucketer : public EventHandler {
 public:
  using EventHandler::EventHandler;
  // FIXME: we might need more advanced ingress/egress event specifications.
  // E.g., LOCAL_GRAD_READY -> BUCKET_READY; COMM_DONE -> GLOBAL_GRAD_READY,
  // otherwise, DefaultBucketer and AllReduceComm can form a cycle.
  C10_NODISCARD std::vector<EventSchema> ingressEvents() const override;
  C10_NODISCARD std::vector<EventSchema> egressEvents() const override;
  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override;

 private:
  std::vector<std::shared_ptr<Future>> handlePrepareModule(
      c10::intrusive_ptr<PrepareModuleEvent> event);

  std::vector<std::shared_ptr<Future>> handleLocalGradReady(
      c10::intrusive_ptr<LocalGradReadyEvent> event);

  std::vector<std::shared_ptr<Future>> handleCommDone(
      c10::intrusive_ptr<CommDoneEvent> event);

  std::vector<at::Tensor> params_;
};

// WIP AllReduceComm
// TODO:
// 1. Launch AllReduce asynchronously
class TORCH_API AllReduceComm : public EventHandler {
 public:
  explicit AllReduceComm(c10::intrusive_ptr<c10d::ProcessGroup> pg)
      : EventHandler(), pg_(std::move(pg)) {}

  C10_NODISCARD std::vector<EventSchema> ingressEvents() const override;
  C10_NODISCARD std::vector<EventSchema> egressEvents() const override;
  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override;

 private:
  std::vector<std::shared_ptr<Future>> handleBucketReady(
      c10::intrusive_ptr<BucketReadyEvent> event);

  const c10::intrusive_ptr<c10d::ProcessGroup> pg_;
};

} // namespace spmd
} // namespace distributed
} // namespace torch
