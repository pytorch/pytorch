#include <c10d/default_comm_hooks.hpp>

#include <c10d/comm.hpp>
#include <c10d/ProcessGroup.hpp>
#include <torch/torch.h>

namespace c10d {

c10::intrusive_ptr<c10::ivalue::Future> AllReduceCommHook::runHook(
    GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getTensorRef()};
  auto allreduce_fut = state_->allreduce(tensors)->getFuture();
  auto div_by_process_group_size = [size = state_->getSize()](
        c10::ivalue::Future& allreduce_fut) {
    auto result = allreduce_fut.value();
    TORCH_INTERNAL_ASSERT(result.isTensorList(),
        "ProcessGroup::allreduce should return TensorList");
    auto tensor = result.toTensorVector()[0] / size;
    return c10::IValue(tensor);
  };

  return allreduce_fut->then(div_by_process_group_size, allreduce_fut->elementType());
}

c10::intrusive_ptr<c10::ivalue::Future> FP16CompressCommHook::runHook(
    GradBucket& bucket) {
  auto& tensor = bucket.getTensorRef();
  tensor.copy_(tensor.to(torch::kFloat16));
  std::vector<at::Tensor> tensors = {tensor};
  auto allreduce_fut = state_->allreduce(tensors)->getFuture();
  auto decompress_and_div_by_process_group_size =
      [allreduce_fut, this](c10::ivalue::Future& allreduce_fut) {
        auto result = allreduce_fut.value();
        TORCH_INTERNAL_ASSERT(result.isTensorList(),
            "ProcessGroup::allreduce should return TensorList");
        auto tensor = result.toTensorVector()[0];
        tensor.copy_(tensor.to(torch::kFloat) / state_->getSize());
        return c10::IValue(tensor);
      };

  return allreduce_fut->then(
      decompress_and_div_by_process_group_size, allreduce_fut->elementType());
}

} // namespace c10d
