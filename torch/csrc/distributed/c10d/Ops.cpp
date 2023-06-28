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
      "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? sparse_indices, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
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
      "reduce_scatter_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> __torch__.torch.classes.c10d.Work");
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

namespace {

#define IMPL_SEND(DEV)                                                        \
  c10::intrusive_ptr<Work> send##DEV(                                         \
      at::TensorList tensors,                                                 \
      const c10::intrusive_ptr<ProcessGroup>& process_group,                  \
      int64_t dstRank,                                                        \
      int64_t tag) {                                                          \
    auto tensor_vec = tensors.vec();                                          \
    return process_group->getBackend(c10::DeviceType::DEV)                    \
        ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag)); \
  }

IMPL_SEND(CPU)
IMPL_SEND(CUDA)
IMPL_SEND(PrivateUse1)

#define IMPL_RECV(DEV)                                                        \
  c10::intrusive_ptr<Work> recv_##DEV(                                        \
      at::TensorList tensors,                                                 \
      const c10::intrusive_ptr<ProcessGroup>& process_group,                  \
      int64_t srcRank,                                                        \
      int64_t tag) {                                                          \
    auto tensor_vec = tensors.vec();                                          \
    return process_group->getBackend(c10::DeviceType::DEV)                    \
        ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag)); \
  }

IMPL_RECV(CPU)
IMPL_RECV(CUDA)
IMPL_RECV(PrivateUse1)

#define IMPL_RECV_ANY_SOURCE(DEV)                            \
  c10::intrusive_ptr<Work> recv_any_source_##DEV(            \
      at::TensorList tensors,                                \
      const c10::intrusive_ptr<ProcessGroup>& process_group, \
      int64_t tag) {                                         \
    auto tensor_vec = tensors.vec();                         \
    return process_group->getBackend(c10::DeviceType::DEV)   \
        ->recvAnysource(tensor_vec, static_cast<int>(tag));  \
  }

IMPL_RECV_ANY_SOURCE(CPU)
IMPL_RECV_ANY_SOURCE(CUDA)
IMPL_RECV_ANY_SOURCE(PrivateUse1)

#define IMPL_REDUCE(DEV)                                     \
  c10::intrusive_ptr<Work> reduce_##DEV(                     \
      at::TensorList tensors,                                \
      const c10::intrusive_ptr<ProcessGroup>& process_group, \
      const c10::intrusive_ptr<ReduceOp>& reduce_op,         \
      int64_t root_rank,                                     \
      int64_t root_tensor,                                   \
      int64_t timeout) {                                     \
    auto tensor_vec = tensors.vec();                         \
    return process_group->getBackend(c10::DeviceType::DEV)   \
        ->reduce(                                            \
            tensor_vec,                                      \
            ReduceOptions{                                   \
                *reduce_op.get(),                            \
                root_rank,                                   \
                root_tensor,                                 \
                std::chrono::milliseconds(timeout)});        \
  }

IMPL_REDUCE(CPU)
IMPL_REDUCE(CUDA)
IMPL_REDUCE(PrivateUse1)

#define IMPL_BROADCAST(DEV)                                               \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>           \
      broadcast_##DEV(                                                    \
          at::TensorList tensors,                                         \
          const c10::intrusive_ptr<ProcessGroup>& process_group,          \
          int64_t root_rank,                                              \
          int64_t root_tensor,                                            \
          int64_t timeout) {                                              \
    auto tensor_vec = tensors.vec();                                      \
    auto work = process_group->getBackend(c10::DeviceType::DEV)           \
                    ->broadcast(                                          \
                        tensor_vec,                                       \
                        BroadcastOptions{                                 \
                            root_rank,                                    \
                            root_tensor,                                  \
                            std::chrono::milliseconds(timeout)});         \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>( \
        std::move(tensor_vec), work);                                     \
  }

IMPL_BROADCAST(CPU)
IMPL_BROADCAST(CUDA)
IMPL_BROADCAST(PrivateUse1)

// Return input tensors as output tensors to make inplace allreduce look like
// a functional API, so that make_fx can correctly build the dependencies in
// the graph later.
#define IMPL_ALLREDUCE(DEV)                                                 \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>             \
      allreduce_##DEV(                                                      \
          at::TensorList tensors,                                           \
          const c10::intrusive_ptr<ProcessGroup>& process_group,            \
          const c10::intrusive_ptr<ReduceOp>& reduce_op,                    \
          const c10::optional<at::Tensor>& sparse_indices,                  \
          int64_t timeout) {                                                \
    auto tensor_vec = tensors.vec();                                        \
    auto work =                                                             \
        process_group->getBackend(c10::DeviceType::DEV)                     \
            ->allreduce(                                                    \
                tensor_vec,                                                 \
                AllreduceOptions{                                           \
                    *reduce_op.get(), std::chrono::milliseconds(timeout)}); \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(   \
        std::move(tensor_vec), work);                                       \
  }

IMPL_ALLREDUCE(CPU)
IMPL_ALLREDUCE(CUDA)
IMPL_ALLREDUCE(PrivateUse1)

#define IMPL_ALLREDUCE_COALESCED(DEV)                             \
  c10::intrusive_ptr<Work> allreduce_coalesced_##DEV(             \
      at::TensorList tensors,                                     \
      const c10::intrusive_ptr<ProcessGroup>& process_group,      \
      const c10::intrusive_ptr<ReduceOp>& reduce_op,              \
      int64_t timeout) {                                          \
    auto tensor_vec = tensors.vec();                              \
    AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{}; \
    opts.reduceOp = *reduce_op.get();                             \
    opts.timeout = std::chrono::milliseconds(timeout);            \
    return process_group->getBackend(c10::DeviceType::DEV)        \
        ->allreduce_coalesced(tensor_vec, opts);                  \
  }

IMPL_ALLREDUCE_COALESCED(CPU)
IMPL_ALLREDUCE_COALESCED(CUDA)
IMPL_ALLREDUCE_COALESCED(PrivateUse1)

// Copy output tensors (not storage) so that this can be used in a functional
// manner
#define IMPL_ALLGATHER(DEV)                                                    \
  std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>   \
      allgather_##DEV(                                                         \
          const std::vector<std::vector<at::Tensor>>& output_tensors,          \
          at::TensorList input_tensors,                                        \
          const c10::intrusive_ptr<ProcessGroup>& process_group,               \
          int64_t timeout) {                                                   \
    auto input_tensors_vec = input_tensors.vec();                              \
    auto work = process_group->getBackend(c10::DeviceType::DEV)                \
                    ->allgather(                                               \
                        const_cast<std::vector<std::vector<at::Tensor>>&>(     \
                            output_tensors),                                   \
                        input_tensors_vec,                                     \
                        AllgatherOptions{std::chrono::milliseconds(timeout)}); \
    return std::                                                               \
        tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>( \
            output_tensors, work);                                             \
  }

IMPL_ALLGATHER(CPU)
IMPL_ALLGATHER(CUDA)
IMPL_ALLGATHER(PrivateUse1)

#define IMPL__ALLGATHER_BASE(DEV)                                         \
  std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_##DEV( \
      at::Tensor& output_tensor,                                          \
      at::Tensor& input_tensor,                                           \
      const c10::intrusive_ptr<ProcessGroup>& process_group) {            \
    auto work = process_group->getBackend(c10::DeviceType::DEV)           \
                    ->_allgather_base(output_tensor, input_tensor);       \
    return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(              \
        output_tensor, work);                                             \
  }

IMPL__ALLGATHER_BASE(CPU)
IMPL__ALLGATHER_BASE(CUDA)
IMPL__ALLGATHER_BASE(PrivateUse1)

#define IMPL_ALLGATHER_COALESCED(DEV)                                        \
  c10::intrusive_ptr<Work> allgather_coalesced_##DEV(                        \
      const std::vector<std::vector<at::Tensor>>& output_lists,              \
      const at::TensorList& input_list,                                      \
      const c10::intrusive_ptr<ProcessGroup>& process_group) {               \
    auto input_list_vec = input_list.vec();                                  \
    return process_group->getBackend(c10::DeviceType::DEV)                   \
        ->allgather_coalesced(                                               \
            const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists), \
            input_list_vec);                                                 \
  }

IMPL_ALLGATHER_COALESCED(CPU)
IMPL_ALLGATHER_COALESCED(CUDA)
IMPL_ALLGATHER_COALESCED(PrivateUse1)

#define IMPL_ALLGATHER_INTO_TENSOR_COALESCED(DEV)                       \
  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_##DEV( \
      at::TensorList outputs,                                           \
      at::TensorList inputs,                                            \
      const c10::intrusive_ptr<ProcessGroup>& process_group) {          \
    auto output_vec = outputs.vec();                                    \
    auto input_vec = inputs.vec();                                      \
    return process_group->getBackend(c10::DeviceType::DEV)              \
        ->allgather_into_tensor_coalesced(output_vec, input_vec);       \
  }

IMPL_ALLGATHER_INTO_TENSOR_COALESCED(CPU)
IMPL_ALLGATHER_INTO_TENSOR_COALESCED(CUDA)
IMPL_ALLGATHER_INTO_TENSOR_COALESCED(PrivateUse1)

#define IMPL_REDUCE_SCATTER(DEV)                                            \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>             \
      reduce_scatter_##DEV(                                                 \
          const at::TensorList& output_tensors,                             \
          const std::vector<std::vector<at::Tensor>>& input_tensors,        \
          const c10::intrusive_ptr<ProcessGroup>& process_group,            \
          const c10::intrusive_ptr<ReduceOp>& reduce_op,                    \
          int64_t timeout) {                                                \
    auto output_tensors_vec = output_tensors.vec();                         \
    auto work =                                                             \
        process_group->getBackend(c10::DeviceType::DEV)                     \
            ->reduce_scatter(                                               \
                output_tensors_vec,                                         \
                const_cast<std::vector<std::vector<at::Tensor>>&>(          \
                    input_tensors),                                         \
                ReduceScatterOptions{                                       \
                    *reduce_op.get(), std::chrono::milliseconds(timeout)}); \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(   \
        output_tensors_vec, work);                                          \
  }

IMPL_REDUCE_SCATTER(CPU)
IMPL_REDUCE_SCATTER(CUDA)
IMPL_REDUCE_SCATTER(PrivateUse1)

#define IMPL__REDUCE_SCATTER_BASE(DEV)                                         \
  std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_##DEV( \
      at::Tensor& output_tensor,                                               \
      at::Tensor& input_tensor,                                                \
      const c10::intrusive_ptr<ProcessGroup>& process_group,                   \
      const c10::intrusive_ptr<ReduceOp>& reduce_op,                           \
      int64_t timeout) {                                                       \
    auto work =                                                                \
        process_group->getBackend(c10::DeviceType::DEV)                        \
            ->_reduce_scatter_base(                                            \
                output_tensor,                                                 \
                input_tensor,                                                  \
                ReduceScatterOptions{                                          \
                    *reduce_op.get(), std::chrono::milliseconds(timeout)});    \
    return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(                   \
        output_tensor, work);                                                  \
  }

IMPL__REDUCE_SCATTER_BASE(CPU)
IMPL__REDUCE_SCATTER_BASE(CUDA)
IMPL__REDUCE_SCATTER_BASE(PrivateUse1)

#define IMPL_REDUCE_SCATTER_TENSOR_COALESCED(DEV)                       \
  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced_##DEV( \
      at::TensorList outputs,                                           \
      at::TensorList inputs,                                            \
      const c10::intrusive_ptr<ProcessGroup>& process_group,            \
      const c10::intrusive_ptr<ReduceOp>& reduce_op,                    \
      int64_t timeout) {                                                \
    auto output_vec = outputs.vec();                                    \
    auto input_vec = inputs.vec();                                      \
    return process_group->getBackend(c10::DeviceType::DEV)              \
        ->reduce_scatter_tensor_coalesced(                              \
            output_vec,                                                 \
            input_vec,                                                  \
            ReduceScatterOptions{                                       \
                *reduce_op.get(), std::chrono::milliseconds(timeout)}); \
  }

IMPL_REDUCE_SCATTER_TENSOR_COALESCED(CPU)
IMPL_REDUCE_SCATTER_TENSOR_COALESCED(CUDA)
IMPL_REDUCE_SCATTER_TENSOR_COALESCED(PrivateUse1)

#define IMPL_GATHER(DEV)                                                       \
  c10::intrusive_ptr<Work> gather_##DEV(                                       \
      const std::vector<std::vector<at::Tensor>>& output_tensors,              \
      const at::TensorList& input_tensors,                                     \
      const c10::intrusive_ptr<ProcessGroup>& process_group,                   \
      int64_t root_rank,                                                       \
      int64_t timeout) {                                                       \
    auto input_tensors_vec = input_tensors.vec();                              \
    return process_group->getBackend(c10::DeviceType::DEV)                     \
        ->gather(                                                              \
            const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors), \
            input_tensors_vec,                                                 \
            GatherOptions{root_rank, std::chrono::milliseconds(timeout)});     \
  }

IMPL_GATHER(CPU)
IMPL_GATHER(CUDA)
IMPL_GATHER(PrivateUse1)

#define IMPL_SCATTER(DEV)                                                      \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_##DEV( \
      const at::TensorList& output_tensors,                                    \
      const std::vector<std::vector<at::Tensor>>& input_tensors,               \
      const c10::intrusive_ptr<ProcessGroup>& process_group,                   \
      int64_t root_rank,                                                       \
      int64_t timeout) {                                                       \
    auto output_tensors_vec = output_tensors.vec();                            \
    auto work = process_group->getBackend(c10::DeviceType::DEV)                \
                    ->scatter(                                                 \
                        output_tensors_vec,                                    \
                        const_cast<std::vector<std::vector<at::Tensor>>&>(     \
                            input_tensors),                                    \
                        ScatterOptions{                                        \
                            root_rank, std::chrono::milliseconds(timeout)});   \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(      \
        std::move(output_tensors_vec), work);                                  \
  }

IMPL_SCATTER(CPU)
IMPL_SCATTER(CUDA)
IMPL_SCATTER(PrivateUse1)

#define IMPL_ALLTOALL(DEV)                                                    \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>               \
      alltoall_##DEV(                                                         \
          const at::TensorList& output_tensors,                               \
          const at::TensorList& input_tensors,                                \
          const c10::intrusive_ptr<ProcessGroup>& process_group,              \
          int64_t timeout) {                                                  \
    auto output_tensors_vec = output_tensors.vec();                           \
    auto input_tensors_vec = input_tensors.vec();                             \
    auto work = process_group->getBackend(c10::DeviceType::DEV)               \
                    ->alltoall(                                               \
                        output_tensors_vec,                                   \
                        input_tensors_vec,                                    \
                        AllToAllOptions{std::chrono::milliseconds(timeout)}); \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(     \
        std::move(output_tensors_vec), work);                                 \
  }

IMPL_ALLTOALL(CPU)
IMPL_ALLTOALL(CUDA)
IMPL_ALLTOALL(PrivateUse1)

#define IMPL_ALLTOALL_BASE(DEV)                                   \
  c10::intrusive_ptr<Work> alltoall_base_##DEV(                   \
      at::Tensor& output,                                         \
      at::Tensor& input,                                          \
      const c10::intrusive_ptr<ProcessGroup>& process_group,      \
      std::vector<int64_t> output_split_sizes,                    \
      std::vector<int64_t> input_split_sizes,                     \
      int64_t timeout) {                                          \
    return process_group->getBackend(c10::DeviceType::DEV)        \
        ->alltoall_base(                                          \
            output,                                               \
            input,                                                \
            output_split_sizes,                                   \
            input_split_sizes,                                    \
            AllToAllOptions{std::chrono::milliseconds(timeout)}); \
  }

IMPL_ALLTOALL_BASE(CPU)
IMPL_ALLTOALL_BASE(CUDA)
IMPL_ALLTOALL_BASE(PrivateUse1)

#define IMPL_BARRIER(DEV)                                                    \
  c10::intrusive_ptr<Work> barrier##DEV(                                     \
      at::Tensor /* unused */,                                               \
      const c10::intrusive_ptr<ProcessGroup>& process_group,                 \
      const std::vector<int64_t>& device_ids,                                \
      int64_t timeout) {                                                     \
    return process_group->getBackend(c10::DeviceType::DEV)                   \
        ->barrier(                                                           \
            BarrierOptions{device_ids, std::chrono::milliseconds(timeout)}); \
  }

IMPL_BARRIER(CPU)
IMPL_BARRIER(CUDA)
IMPL_BARRIER(PrivateUse1)

void monitored_barrier_CPU(
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

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
allreduce_sparse_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    const c10::optional<at::Tensor>& sparse_indices,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::CUDA)
                  ->allreduce_sparse(
                      tensor_vec,
                      AllreduceOptions{
                          *reduce_op.get(),
                          std::chrono::milliseconds(timeout),
                          sparse_indices});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}
} // namespace

// register functions to dispatcher
namespace {

// 2nd level expansion
// FUNC: op name
// DEV: device
#define REGISTER_C10D_OP1(FUNC, DEV) \
  TORCH_LIBRARY_IMPL(c10d, DEV, m) { \
    m.impl(#FUNC, FUNC##DEV);        \
  }

// 1st level expansion
#define REGISTER_C10D_OP(FUNC)  \
  REGISTER_C10D_OP1(FUNC, CPU)  \
  REGISTER_C10D_OP1(FUNC, CUDA) \
  REGISTER_C10D_OP1(FUNC, PrivateUse1)

// Now we start to register ops with the three device keys

REGISTER_C10D_OP(send)
REGISTER_C10D_OP(recv_)
REGISTER_C10D_OP(recv_any_source_)
REGISTER_C10D_OP(reduce_)
REGISTER_C10D_OP(broadcast_)
REGISTER_C10D_OP(allreduce_)
REGISTER_C10D_OP(allreduce_coalesced_)
REGISTER_C10D_OP(allgather_)
REGISTER_C10D_OP(_allgather_base_)
REGISTER_C10D_OP(allgather_coalesced_)
REGISTER_C10D_OP(allgather_into_tensor_coalesced_)
REGISTER_C10D_OP(reduce_scatter_)
REGISTER_C10D_OP(_reduce_scatter_base_)
REGISTER_C10D_OP(reduce_scatter_tensor_coalesced_)
REGISTER_C10D_OP(gather_)
REGISTER_C10D_OP(scatter_)
REGISTER_C10D_OP(alltoall_)
REGISTER_C10D_OP(alltoall_base_)
REGISTER_C10D_OP(barrier)

// The following ops are specialized, register them separately

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("monitored_barrier_", monitored_barrier_CPU);
}

// TODO: The SparseCPU/SparseCUDA dispatched methods are only used to support
// sparse all_reduce in the Gloo backend
TORCH_LIBRARY_IMPL(c10d, SparseCPU, m) {
  m.impl("allreduce_", allreduce_CPU);
}

TORCH_LIBRARY_IMPL(c10d, SparseCUDA, m) {
  m.impl("allreduce_", allreduce_sparse_cuda_);
}

} // namespace

} // namespace ops
} // namespace c10d
