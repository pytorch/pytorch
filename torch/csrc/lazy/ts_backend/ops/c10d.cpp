#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/ts_backend/ops/allreduce.h>
#include <torch/csrc/lazy/ts_backend/ops/broadcast.h>
#include <torch/library.h>

namespace torch {
namespace lazy {

at::Tensor broadcast(const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group, int64_t root_rank, int64_t root_tensor,
    int64_t timeout) {
  TORCH_LAZY_FN_COUNTER("lazy::");

  // Currently LazyTensor IR cannot interpret the eager FunctionSchema, i.e.,
  // c10::intrusive_ptr<ProcessGroup::Work> broadcast(at::TensorList tensors, const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank, int64_t root_tensor, int64_t timeout).
  // For two reasons:
  // 1: Don't understand the ProcessGroup::Work return type. LazyTensor IR expects returns types to be all Tensors.
  // 2: If we change the return types to be a TensorList, it still won't work given TensorList is not supported as well.
  // So here, we restrict the input to be one tensor and overload another broadcast FunctionSchema, i.e.,
  // at::Tensor broadcast_(const at::Tensor& tensor, const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank, int64_t root_tensor, int64_t timeout)
  // In the new schema, it basically makes the broadcast as an inplace op.
//   TORCH_CHECK(tensors.size() == 1lu, "Only one tensor input is supported for c10d::broadcast.")
//   auto& tensor = tensors[0];
//   auto input = GetLtcTensor(tensor);
//   input->SetIrValue(MakeNode<Broadcast>(input->GetIrValue(), process_group, root_rank, root_tensor, timeout,
//       std::vector<Shape>{Shape(tensor.scalar_type(), tensor.sizes().vec())}));

//   auto future = c10::make_intrusive<at::ivalue::Future>(c10::TensorType::get(),
//       std::vector<c10::Device>{tensor.device()});
//   future->markCompleted(tensor);
//   // For profilingTitle, there is no programatic way to query such value. They are hard-coded for each
//   // backend. Since c10d::ProcessGroup::Work will eventually go away, let's leave it as it is now.
//   auto work = c10::make_intrusive<c10d::ProcessGroup::Work>(process_group->getRank(), c10d::OpType::BROADCAST,
//       /*profilingTitle=*/nullptr, tensors.vec());
//   work->setFuture(std::move(future));
//   // We might want to extend Work instead of making .finish() public.
//   work->finish();
//   return work;

  // Adopts the new broadcast_ schema as AOT.
  auto input = GetLtcTensor(tensor);
  input->SetIrValue(MakeNode<Broadcast>(input->GetIrValue(), process_group, root_rank, root_tensor, timeout,
      std::vector<Shape>{Shape(tensor.scalar_type(), tensor.sizes().vec())}));
  return tensor;
}

at::Tensor allreduce(const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group, int64_t reduce_op, int64_t timeout) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto input = GetLtcTensor(tensor);
  input->SetIrValue(MakeNode<Allreduce>(input->GetIrValue(), process_group, reduce_op, timeout,
      std::vector<Shape>{Shape(tensor.scalar_type(), tensor.sizes().vec())}));
  return tensor;
}

TORCH_LIBRARY_IMPL(c10d, Lazy, m) {
    m.impl("broadcast_", broadcast);
    m.impl("allreduce_", allreduce);
}

}  // namespace lazy
}  // namespace torch
