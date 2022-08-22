#include <torch/csrc/distributed/c10d/OpsImpl.hpp>
#include <torch/library.h>

namespace c10d {
namespace ops {

// Below are ProcessGroup's corresponding ops for each backend. Ops are but
// routed through the dispatcher to be dispatched to the appropriate backend.
// Currently a no-op as the process group does not have a list of backends.
c10::intrusive_ptr<Work> send_cpu(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->send(
      tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> send_cuda(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->send(
      tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->recv(
      tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->recv(
      tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> broadcast_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->broadcast(
      tensor_vec,
      BroadcastOptions{
          root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> broadcast_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->broadcast(
      tensor_vec,
      BroadcastOptions{
          root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> allreduce_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->allreduce(
      tensor_vec,
      AllreduceOptions{
          static_cast<ReduceOp>(reduce_op),
          std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> allreduce_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->allreduce(
      tensor_vec,
      AllreduceOptions{
          static_cast<ReduceOp>(reduce_op),
          std::chrono::milliseconds(timeout)});
}

// register functions to dispatcher
namespace {
TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("send", send_cpu);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("send", send_cuda);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("recv_", recv_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("recv_", recv_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("broadcast_", broadcast_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("broadcast_", broadcast_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("allreduce_", allreduce_cpu_);
}

// TODO: The SparseCPU/SparseCUDA dispatched methods are only used to support
// sparse all_reduce in the Gloo backend
TORCH_LIBRARY_IMPL(c10d, SparseCPU, m) {
  m.impl("allreduce_", allreduce_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, SparseCUDA, m) {
  m.impl("allreduce_", allreduce_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("allreduce_", allreduce_cuda_);
}
} // namespace

} // namespace ops
} // namespace c10d
