#include <c10d/default_comm_hooks.hpp>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/comm.hpp>
#include <torch/torch.h>

namespace {
void convert_complex_tensors(std::vector<at::Tensor> &tensors) {
  for (auto &tensor : tensors) {
    if (tensor.is_complex()) {
      tensor = at::view_as_real(tensor);
    }
  }
}

} // namespace

namespace c10d {
c10::intrusive_ptr<c10::ivalue::Future> AllReduceCommHook::runHook(
    GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getBufferRef()};
  convert_complex_tensors(tensors);
  // Apply the division first to avoid overflow, especially for FP16.
  tensors[0] /= state_->getSize();
  return state_->allreduce(tensors)->getFuture();
}

c10::intrusive_ptr<c10::ivalue::Future> FP16CompressCommHook::runHook(
    GradBucket& bucket) {

  at::Tensor compressed_tensor;
  if (bucket.getBufferRef().is_complex()) {
    compressed_tensor = at::view_as_real(bucket.getBufferRef()).to(torch::kFloat16);
  } else {
    compressed_tensor = bucket.getBufferRef().to(torch::kFloat16);
  }

  // Apply the division first to avoid overflow.
  compressed_tensor /= state_->getSize();
  std::vector<at::Tensor> tensors = {compressed_tensor};

  auto allreduce_fut = state_->allreduce(tensors)->getFuture();
  auto decompressed_tensor = bucket.getBufferRef();
  auto decompress = [decompressed_tensor](c10::ivalue::Future& allreduce_fut) {
    auto result = allreduce_fut.value();
    TORCH_INTERNAL_ASSERT(
        result.isTensorList(),
        "ProcessGroup::allreduce should return TensorList");

    auto reduce_tensor = result.toTensorVector()[0];
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      reduce_tensor.scalar_type() == at::ScalarType::Half,
      "Expected reduced tensor to be fp16 in FP16CompressHook, but got type ",
      reduce_tensor.scalar_type()
    );
    if (decompressed_tensor.is_complex()) {
      reduce_tensor = at::view_as_complex(reduce_tensor
        .to(c10::toRealValueType(decompressed_tensor.scalar_type())));
    }
    decompressed_tensor.copy_(reduce_tensor);
    return c10::IValue(decompressed_tensor);
  };

  return allreduce_fut->then(decompress, allreduce_fut->elementType());
}

c10::intrusive_ptr<c10::ivalue::Future> _AllReduceBySumCommHook::
    runHook(GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getBufferRef()};
  convert_complex_tensors(tensors);
  return state_->allreduce(tensors)->getFuture();
}

} // namespace c10d
