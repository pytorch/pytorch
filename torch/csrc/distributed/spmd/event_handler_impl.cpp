#include <torch/csrc/distributed/spmd/event_handler_impl.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>


namespace torch {
namespace distributed {
namespace spmd {

using c10::IValue;

namespace {

template <class T>
std::vector<std::shared_ptr<Future>> createOneFutureEvent(
    c10::intrusive_ptr<T> event) {
  auto future = std::make_shared<Future>(at::AnyClassType::get());
  future->markCompleted(
      IValue(c10::static_intrusive_pointer_cast<Event>(event)));
  std::vector<std::shared_ptr<Future>> futures;
  futures.reserve(1);
  futures.emplace_back(std::move(future));
  return futures;
}

} // namespace

/////////////////////////////////////////////////////////////////////
//                         DefaultTrigger                          //
/////////////////////////////////////////////////////////////////////

std::vector<EventSchema> DefaultTrigger::ingressEvents() const {
  return {EventType::PREPARE_MODULE, EventType::PRE_FORWARD};
}

std::vector<EventSchema> DefaultTrigger::egressEvents() const {
  return {EventType::LOCAL_GRAD_READY};
}

std::vector<std::shared_ptr<Future>> DefaultTrigger::handleEvent(
    const c10::intrusive_ptr<Event>& event) {
  switch (event->schema().type_) {
    case EventType::PREPARE_MODULE:
      return handlePrepareModule(
          c10::static_intrusive_pointer_cast<PrepareModuleEvent>(event));
    case EventType::PRE_FORWARD:
      return handlePreForward(
          c10::static_intrusive_pointer_cast<PreForwardEvent>(event));
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "DefaultTrigger retrieved an unexcepted event type ",
          event->schema().type_);
  }
}

std::vector<std::shared_ptr<Future>> DefaultTrigger::handlePrepareModule(
    c10::intrusive_ptr<PrepareModuleEvent> event) {
  params_ = event->parameters();
  for (size_t index = 0; index < params_.size(); ++index) {
    auto& param = params_[index];
    auto gradAccumulator = torch::autograd::impl::grad_accumulator(param);
    // Hook to execute after the gradient accumulator has executed.
    gradAccumulator->add_post_hook(
        torch::make_unique<torch::autograd::utils::LambdaPostHook>(
            [this, index](
                const torch::autograd::variable_list& outputs,
                const torch::autograd::variable_list& /* unused */) {
              autogradHook(index);
              return outputs;
            }));
    gradAccumulators_.push_back(std::move(gradAccumulator));
  }
  return {};
}

std::vector<std::shared_ptr<Future>> DefaultTrigger::handlePreForward(
    c10::intrusive_ptr<PreForwardEvent> /* unused */) {
  gradReadyFutures_.clear();
  gradReadyFutures_.reserve(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    gradReadyFutures_.emplace_back(
        std::make_shared<Future>(at::AnyClassType::get()));
  }
  return gradReadyFutures_;
}

void DefaultTrigger::autogradHook(size_t index) {
  TORCH_INTERNAL_ASSERT(gradReadyFutures_.size() == params_.size());
  auto lgr = c10::make_intrusive<LocalGradReadyEvent>(
      index, params_[index].mutable_grad());

  gradReadyFutures_[index]->markCompleted(
      IValue(c10::static_intrusive_pointer_cast<Event>(lgr)));
}

/////////////////////////////////////////////////////////////////////
//                        DefaultBucketer                          //
/////////////////////////////////////////////////////////////////////

// FIXME: we might need more advanced ingress/egress event specifications.
// E.g., LOCAL_GRAD_READY -> BUCKET_READY; COMM_DONE -> GLOBAL_GRAD_READY,
// otherwise, DefaultBucketer and AllReduceComm can form a cycle.
std::vector<EventSchema> DefaultBucketer::ingressEvents() const {
  // FIXME: consume PREPARE_MODULE to allocate buckets
  return {
      EventType::PREPARE_MODULE,
      EventType::LOCAL_GRAD_READY,
      EventType::COMM_DONE};
}

std::vector<EventSchema> DefaultBucketer::egressEvents() const {
  return {EventType::BUCKET_READY, EventType::GLOBAL_GRAD_READY};
}

std::vector<std::shared_ptr<Future>> DefaultBucketer::handleEvent(
    const c10::intrusive_ptr<Event>& event) {
  switch (event->schema().type_) {
    case EventType::PREPARE_MODULE: {
      return handlePrepareModule(
          c10::static_intrusive_pointer_cast<PrepareModuleEvent>(event));
    }
    case EventType::LOCAL_GRAD_READY: {
      return handleLocalGradReady(
          c10::static_intrusive_pointer_cast<LocalGradReadyEvent>(event));
    }
    case EventType::COMM_DONE: {
      return {};
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "DefaultBucketer unexcepted event type",
          event->schema().type_);
  }
}

std::vector<std::shared_ptr<Future>> DefaultBucketer::handlePrepareModule(
    c10::intrusive_ptr<PrepareModuleEvent> event) {
  params_ = event->parameters();
  return {};
}

std::vector<std::shared_ptr<Future>> DefaultBucketer::handleLocalGradReady(
    c10::intrusive_ptr<LocalGradReadyEvent> event) {
  std::vector<size_t> paramIndices;
  paramIndices.reserve(1);
  paramIndices.push_back(event->index());
  auto bucket = std::make_shared<Bucket>(
      event->index(), event->grad(), std::move(paramIndices));
  return createOneFutureEvent<BucketReadyEvent>(
      c10::make_intrusive<BucketReadyEvent>(std::move(bucket)));
}

std::vector<std::shared_ptr<Future>> DefaultBucketer::handleCommDone(
    c10::intrusive_ptr<CommDoneEvent> event) {
  const auto& bucket = event->bucket();
  auto& grad = params_[bucket->index_].mutable_grad();
  if (bucket->tensor_.data_ptr() != grad.data_ptr()) {
    grad.copy_(bucket->tensor_, /*non_blocking=*/true);
  }

  // FIXME
  TORCH_INTERNAL_ASSERT(bucket->paramIndices_.size() == 1);
  return createOneFutureEvent<GlobalGradReadyEvent>(
      c10::make_intrusive<GlobalGradReadyEvent>(bucket->paramIndices_[0]));
}

/////////////////////////////////////////////////////////////////////
//                         AllReduceComm                           //
/////////////////////////////////////////////////////////////////////

std::vector<EventSchema> AllReduceComm::ingressEvents() const {
  return {EventType::BUCKET_READY};
}

std::vector<EventSchema> AllReduceComm::egressEvents() const {
  return {EventType::COMM_DONE};
}

std::vector<std::shared_ptr<Future>> AllReduceComm::handleEvent(
    const c10::intrusive_ptr<Event>& event) {
  switch (event->schema().type_) {
    case EventType::BUCKET_READY: {
      return handleBucketReady(
          c10::static_intrusive_pointer_cast<BucketReadyEvent>(event));
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false, "AllReduceComm unexcepted event type ", event->schema().type_);
  }
}

std::vector<std::shared_ptr<Future>> AllReduceComm::handleBucketReady(
    c10::intrusive_ptr<BucketReadyEvent> event) {
  const auto& bucket = event->bucket();
  std::vector<at::Tensor> buffers;
  buffers.reserve(1);
  buffers.push_back(bucket->tensor_);
  pg_->allreduce(buffers)->wait();
  return createOneFutureEvent<CommDoneEvent>(
      c10::make_intrusive<CommDoneEvent>(bucket));
}

} // namespace spmd
} // namespace distributed
} // namespace torch
