#include <c10d/default_comm_hooks.hpp>

#include <c10d/ProcessGroup.hpp>
#include <c10d/comm.hpp>
#include <torch/torch.h>

namespace c10d {

c10::intrusive_ptr<c10::ivalue::Future> AllReduceCommHook::runHook(
    GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getTensorRef()};
  // Apply the division first to avoid overflow, especially for FP16.
  tensors[0] /= state_->getSize();
  return state_->allreduce(tensors)->getFuture();
}

c10::intrusive_ptr<c10::ivalue::Future> FP16CompressCommHook::runHook(
    GradBucket& bucket) {
  auto& tensor = bucket.getTensorRef();
  tensor.copy_(tensor.to(torch::kFloat16));
  std::vector<at::Tensor> tensors = {tensor};
  // Apply the division first to avoid overflow.
  tensors[0] /= state_->getSize();

  auto allreduce_fut = state_->allreduce(tensors)->getFuture();
  auto decompress = [](c10::ivalue::Future& allreduce_fut) {
    auto result = allreduce_fut.value();
    TORCH_INTERNAL_ASSERT(
        result.isTensorList(),
        "ProcessGroup::allreduce should return TensorList");
    auto reduce_tensor = result.toTensorVector()[0];
    reduce_tensor.copy_(reduce_tensor.to(torch::kFloat));
    return c10::IValue(reduce_tensor);
  };

  return allreduce_fut->then(decompress, allreduce_fut->elementType());
}

c10::intrusive_ptr<c10::ivalue::Future> _AllReduceBySumCommHook::
    runHook(GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getTensorRef()};
  return state_->allreduce(tensors)->getFuture();
}

} // namespace c10d
