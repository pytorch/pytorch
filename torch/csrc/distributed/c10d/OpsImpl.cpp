#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
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
  return process_group->getBackend(c10::DeviceType::CPU)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> send_cuda(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> reduce_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->reduce(
          tensor_vec,
          ReduceOptions{
              *reduce_op.get(),
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> reduce_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->reduce(
          tensor_vec,
          ReduceOptions{
              *reduce_op.get(),
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->broadcast(
              tensor_vec,
              BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->broadcast(
              tensor_vec,
              BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  std::cout << "IN ALLREDUCE CPU" << std::endl;
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->allreduce(
              tensor_vec,
              AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->allreduce(
              tensor_vec,
              AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_cpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_cuda_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_cpu_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->reduce_scatter(
              const_cast<std::vector<at::Tensor>&>(output_tensors),
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_cuda_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->reduce_scatter(
              const_cast<std::vector<at::Tensor>&>(output_tensors),
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors, work);
}

c10::intrusive_ptr<Work> gather_cpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::CPU)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          const_cast<std::vector<at::Tensor>&>(input_tensors),
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> gather_cuda_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          const_cast<std::vector<at::Tensor>&>(input_tensors),
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_cpu_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->scatter(
              const_cast<std::vector<at::Tensor>&>(output_tensors),
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_cuda_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->scatter(
              const_cast<std::vector<at::Tensor>&>(output_tensors),
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors, work);
}

c10::intrusive_ptr<Work> alltoall_cpu_(
    at::TensorList output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->alltoall(
          output_tensors_vec,
          input_tensors_vec,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> alltoall_cuda_(
    at::TensorList output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->alltoall(
          output_tensors_vec,
          input_tensors_vec,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> barrier_cpu(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::CPU)
      ->barrier(BarrierOptions{device_ids, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> barrier_cuda(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->barrier(BarrierOptions{device_ids, std::chrono::milliseconds(timeout)});
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
  m.impl("reduce_", reduce_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("reduce_", reduce_cuda_);
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

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("allgather_", allgather_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("allgather_", allgather_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("reduce_scatter_", reduce_scatter_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("reduce_scatter_", reduce_scatter_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("gather_", gather_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("gather_", gather_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("scatter_", scatter_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("scatter_", scatter_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("alltoall_", alltoall_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("alltoall_", alltoall_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("barrier", barrier_cpu);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("barrier", barrier_cuda);
}

} // namespace

} // namespace ops
} // namespace c10d
