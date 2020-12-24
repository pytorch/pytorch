#include <c10d/default_comm_hooks.hpp>

#include <c10d/comm.hpp>
#include <c10d/ProcessGroup.hpp>
#include <torch/torch.h>

namespace c10d {

c10::intrusive_ptr<c10::ivalue::Future> AllReduceCommHook::runHook(
    GradBucket& bucket) {
  auto allreduce_work = state_->allreduce(bucket.getTensorsRef());

  auto div_by_process_group_size = [allreduce_work, this]() {
    auto tensor = allreduce_work->result()[0] / state_->getSize();
    return c10::IValue(tensor);
  };

  auto fut = allreduce_work->getFuture();
  return fut->then(div_by_process_group_size, fut->elementType());
}

c10::intrusive_ptr<c10::ivalue::Future> FP16CompressCommHook::runHook(
    GradBucket& bucket) {
  auto& tensors = bucket.getTensorsRef();
  for (auto& tensor : tensors) {
    tensor.copy_(tensor.to(torch::kFloat16));
  }
  auto allreduce_work = state_->allreduce(tensors);

  auto decompress_and_div_by_process_group_size = [allreduce_work, this]() {
    auto tensor = allreduce_work->result()[0];
    tensor.copy_(tensor.to(torch::kFloat) / state_->getSize());
    return c10::IValue(tensor);
  };

  auto fut = allreduce_work->getFuture();
  return fut->then(
      decompress_and_div_by_process_group_size, fut->elementType());
}

} // namespace c10d
