#include <torch/csrc/distributed/c10d/default_comm_hooks.h>

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.h>
#include <torch/torch.h>

namespace c10d {

c10::intrusive_ptr<torch::jit::Future> allReduceHook(
    ProcessGroup* process_group,
    GradBucket& bucket) {
  auto allreduce_work = process_group->allreduce(*bucket.getTensorsRef());

  auto div_by_process_group_size = [allreduce_work, process_group]() {
    auto tensor = allreduce_work->result()[0] / process_group->getSize();
    return c10::IValue(tensor);
  };

  auto fut = allreduce_work->getFuture();
  return fut->then(div_by_process_group_size, fut->elementType());
}

c10::intrusive_ptr<torch::jit::Future> fp16CompressHook(
    ProcessGroup* process_group,
    GradBucket& bucket) {
  auto* tensors = bucket.getTensorsRef();
  for (auto& tensor : *tensors) {
    tensor.copy_(tensor.to(torch::kFloat16));
  }
  auto allreduce_work = process_group->allreduce(*tensors);

  auto decompress_and_div_by_process_group_size = [allreduce_work,
                                                   process_group]() {
    auto tensor = allreduce_work->result()[0];
    tensor.copy_(tensor.to(torch::kFloat) / process_group->getSize());
    return c10::IValue(tensor);
  };

  auto fut = allreduce_work->getFuture();
  return fut->then(
      decompress_and_div_by_process_group_size, fut->elementType());
}

} // namespace c10d
