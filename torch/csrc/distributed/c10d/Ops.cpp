#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>

namespace c10d {
namespace {

TORCH_LIBRARY(c10d, m) {
  // The following ProcessGroup, Work, and ReduceOp definitions are more like
  // declarations. They don't expose the details of the two classes into
  // TorchScript.
  m.class_<ProcessGroup>("ProcessGroup").def(torch::init<int64_t, int64_t>());
  m.class_<Work>("Work")
      .def(torch::init<>())
      .def("wait", [](const c10::intrusive_ptr<Work>& self) { self->wait(); });
  m.class_<ReduceOp>("ReduceOp").def(torch::init<>());
  m.def(
      "broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int root_tensor, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "allreduce_coalesced_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int timeout) -> (Tensor[][], __torch__.torch.classes.c10d.Work)");
  m.def(
      "_allgather_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup process_group) -> (Tensor, __torch__.torch.classes.c10d.Work)");
  m.def(
      "allgather_coalesced_(Tensor[][] output_lists, Tensor[] input_list, __torch__.torch.classes.c10d.ProcessGroup process_group) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "allgather_into_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, __torch__.torch.classes.c10d.ProcessGroup process_group) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "reduce_scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "_reduce_scatter_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> (Tensor, __torch__.torch.classes.c10d.Work)");
  m.def(
      "reduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int root_rank, int root_tensor, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "gather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] output_split_sizes, int[] input_split_sizes, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "monitored_barrier_(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, int timeout, bool wait_all_ranks) -> ()");
  m.def(
      "send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int dst, int tag) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int src, int tag) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int tag) -> __torch__.torch.classes.c10d.Work");
}
} // namespace

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

c10::intrusive_ptr<c10d::Work> send_privateuse1(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
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

c10::intrusive_ptr<c10d::Work> recv_privateuse1_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_any_source_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_any_source_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
}

c10::intrusive_ptr<c10d::Work> recv_any_source_privateuse1_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
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

c10::intrusive_ptr<c10d::Work> reduce_privateuse1_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->reduce(
          tensor_vec,
          c10d::ReduceOptions{
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

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
broadcast_privateuse1_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->broadcast(
              tensor_vec,
              c10d::BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
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

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
allreduce_privateuse1_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->allreduce(
              tensor_vec,
              c10d::AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>(
      std::move(tensor_vec), work);
}

c10::intrusive_ptr<Work> allreduce_coalesced_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);

  return process_group->getBackend(c10::DeviceType::CPU)
      ->allreduce_coalesced(tensor_vec, opts);
}

c10::intrusive_ptr<Work> allreduce_coalesced_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);

  return process_group->getBackend(c10::DeviceType::CUDA)
      ->allreduce_coalesced(tensor_vec, opts);
}

c10::intrusive_ptr<c10d::Work> allreduce_coalesced_privateuse1_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  c10d::AllreduceCoalescedOptions opts = c10d::AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);

  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->allreduce_coalesced(tensor_vec, opts);
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

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<c10d::Work>>
allgather_privateuse1_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              c10d::AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::tuple<
      std::vector<std::vector<at::Tensor>>,
      c10::intrusive_ptr<c10d::Work>>(output_tensors, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_cpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto work = process_group->getBackend(c10::DeviceType::CPU)
                  ->_allgather_base(output_tensor, input_tensor);

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_cuda_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto work = process_group->getBackend(c10::DeviceType::CUDA)
                  ->_allgather_base(output_tensor, input_tensor);

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work>>
_allgather_base_privateuse1_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  auto work = process_group->getBackend(c10::DeviceType::PrivateUse1)
                  ->_allgather_base(output_tensor, input_tensor);

  return std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work>>(
      output_tensor, work);
}

c10::intrusive_ptr<Work> allgather_coalesced_cpu_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto input_list_vec = input_list.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec);
}

c10::intrusive_ptr<Work> allgather_coalesced_cuda_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto input_list_vec = input_list.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec);
}

c10::intrusive_ptr<c10d::Work> allgather_coalesced_privateuse1_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  auto input_list_vec = input_list.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec);
}

#define IMPL_ALLGATHER_INTO_TENSOR_COALESCED(DEV)                          \
  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_##DEV##_( \
      at::TensorList outputs,                                              \
      at::TensorList inputs,                                               \
      const c10::intrusive_ptr<ProcessGroup>& process_group) {             \
    auto output_vec = outputs.vec();                                       \
    auto input_vec = inputs.vec();                                         \
    return process_group->getBackend(c10::DeviceType::DEV)                 \
        ->allgather_into_tensor_coalesced(output_vec, input_vec);          \
  }

IMPL_ALLGATHER_INTO_TENSOR_COALESCED(CPU)
IMPL_ALLGATHER_INTO_TENSOR_COALESCED(CUDA)
IMPL_ALLGATHER_INTO_TENSOR_COALESCED(PrivateUse1)

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_cpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors_vec, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_cuda_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors_vec, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
reduce_scatter_privateuse1_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              c10d::ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>(
      output_tensors_vec, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_cpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->_reduce_scatter_base(
              output_tensor,
              input_tensor,
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_cuda_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->_reduce_scatter_base(
              output_tensor,
              input_tensor,
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work>>
_reduce_scatter_base_privateuse1_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->_reduce_scatter_base(
              output_tensor,
              input_tensor,
              c10d::ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work>>(
      output_tensor, work);
}

c10::intrusive_ptr<Work> gather_cpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}
c10::intrusive_ptr<Work> gather_cuda_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<c10d::Work> gather_privateuse1_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          c10d::GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_cpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CPU)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_cuda_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::CUDA)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
scatter_privateuse1_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              c10d::ScatterOptions{
                  root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> alltoall_cpu_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::CPU)
                  ->alltoall(
                      output_tensors_vec,
                      input_tensors_vec,
                      AllToAllOptions{std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> alltoall_cuda_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::CUDA)
                  ->alltoall(
                      output_tensors_vec,
                      input_tensors_vec,
                      AllToAllOptions{std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
alltoall_privateuse1_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->alltoall(
              output_tensors_vec,
              input_tensors_vec,
              c10d::AllToAllOptions{std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>(
      std::move(output_tensors_vec), work);
}

c10::intrusive_ptr<Work> alltoall_base_cpu_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::CPU)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> alltoall_base_cuda_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<c10d::Work> alltoall_base_privateuse1_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          c10d::AllToAllOptions{std::chrono::milliseconds(timeout)});
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

c10::intrusive_ptr<c10d::Work> barrier_privateuse1(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->barrier(
          c10d::BarrierOptions{device_ids, std::chrono::milliseconds(timeout)});
}

void monitored_barrier_cpu_(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout,
    bool wait_all_ranks) {
  process_group->getBackend(c10::DeviceType::CPU)
      ->monitoredBarrier(
          BarrierOptions{device_ids, std::chrono::milliseconds(timeout)},
          wait_all_ranks);
}

// register functions to dispatcher
namespace {

#define REGISTER_C10D_OP(FUNC, DEV)  \
  TORCH_LIBRARY_IMPL(c10d, DEV, m) { \
    m.impl(#FUNC, FUNC##DEV##_);     \
  }

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("send", send_cpu);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("send", send_cuda);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("send", send_privateuse1);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("recv_", recv_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("recv_", recv_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("recv_", recv_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("recv_any_source_", recv_any_source_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("recv_any_source_", recv_any_source_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("recv_any_source_", recv_any_source_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("reduce_", reduce_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("reduce_", reduce_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("reduce_", reduce_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("broadcast_", broadcast_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("broadcast_", broadcast_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("broadcast_", broadcast_privateuse1_);
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

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("allreduce_", allreduce_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("allreduce_coalesced_", allreduce_coalesced_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("allreduce_coalesced_", allreduce_coalesced_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("allreduce_coalesced_", allreduce_coalesced_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("allgather_", allgather_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("allgather_", allgather_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("allgather_", allgather_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("_allgather_base_", _allgather_base_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("_allgather_base_", _allgather_base_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("_allgather_base_", _allgather_base_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("allgather_coalesced_", allgather_coalesced_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("allgather_coalesced_", allgather_coalesced_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("allgather_coalesced_", allgather_coalesced_privateuse1_);
}

REGISTER_C10D_OP(allgather_into_tensor_coalesced_, CPU)
REGISTER_C10D_OP(allgather_into_tensor_coalesced_, CUDA)
REGISTER_C10D_OP(allgather_into_tensor_coalesced_, PrivateUse1)

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("reduce_scatter_", reduce_scatter_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("reduce_scatter_", reduce_scatter_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("reduce_scatter_", reduce_scatter_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("gather_", gather_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("gather_", gather_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("gather_", gather_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("scatter_", scatter_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("scatter_", scatter_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("scatter_", scatter_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("alltoall_", alltoall_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("alltoall_", alltoall_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("alltoall_", alltoall_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("alltoall_base_", alltoall_base_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("alltoall_base_", alltoall_base_cuda_);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("alltoall_base_", alltoall_base_privateuse1_);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("barrier", barrier_cpu);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("barrier", barrier_cuda);
}

TORCH_LIBRARY_IMPL(c10d, PrivateUse1, m) {
  m.impl("barrier", barrier_privateuse1);
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("monitored_barrier_", monitored_barrier_cpu_);
}

} // namespace

} // namespace ops
} // namespace c10d
