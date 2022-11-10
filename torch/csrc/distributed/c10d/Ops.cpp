#include <torch/csrc/distributed/c10d/Ops.hpp>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>

namespace c10d {
namespace {

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work = process_group->broadcast(
      tensor_vec,
      BroadcastOptions{
          root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work = process_group->allreduce(
      tensor_vec,
      AllreduceOptions{*reduce_op.get(), std::chrono::milliseconds(timeout)});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

c10::intrusive_ptr<Work> allreduce_coalesced_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);

  return process_group->allreduce_coalesced(tensor_vec, opts);
}

c10::intrusive_ptr<Work> reduce_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->reduce(
      tensor_vec,
      ReduceOptions{
          *reduce_op.get(),
          root_rank,
          root_tensor,
          std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work = process_group->allgather(
      const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
      input_tensors_vec,
      AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> reduce_scatter_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work = process_group->reduce_scatter(
      const_cast<std::vector<at::Tensor>&>(output_tensors),
      const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
      ReduceScatterOptions{
          *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors, work);
}

c10::intrusive_ptr<Work> gather_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  return process_group->gather(
      const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
      const_cast<std::vector<at::Tensor>&>(input_tensors),
      GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto work = process_group->scatter(
      const_cast<std::vector<at::Tensor>&>(output_tensors),
      const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
      ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors, work);
}

c10::intrusive_ptr<Work> alltoall_(
    at::TensorList output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  return process_group->alltoall(
      output_tensors_vec,
      input_tensors_vec,
      AllToAllOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> barrier(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout) {
  return process_group->barrier(
      BarrierOptions{device_ids, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> send(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->send(
      tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->recv(
      tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

TORCH_LIBRARY(c10d, m) {
  // The following ProcessGroup, Work, and ReduceOp definitions are more like
  // declarations. They don't expose the details of the two classes into
  // TorchScript.
  m.class_<ProcessGroup>("ProcessGroup").def(torch::init<int64_t, int64_t>());
  m.class_<Work>("Work")
      .def(torch::init<>())
      .def("wait", [](const c10::intrusive_ptr<Work>& self) { self->wait(); });
  m.class_<ReduceOp>("ReduceOp").def(torch::init<>());
  // It's important to register the op to the CompositeExplicitAutograd key
  // instead of the CompositeImplicitAutograd key to enable
  // __torch_dispatch__.
  m.def(
      "broadcast_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, broadcast_));
  m.def(
      "allreduce_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, allreduce_));
  m.def(
      "allreduce_coalesced_",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, allreduce_coalesced_));
  m.def(
      "allgather_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, allgather_));
  m.def(
      "reduce_scatter_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, reduce_scatter_));
  m.def(
      "reduce_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, reduce_));
  m.def(
      "gather_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, gather_));
  m.def(
      "scatter_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, scatter_));
  m.def(
      "alltoall_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, alltoall_));
  m.def(
      "barrier",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, barrier));
  m.def("send", dispatch(c10::DispatchKey::CompositeExplicitAutograd, send));
  m.def("recv_", dispatch(c10::DispatchKey::CompositeExplicitAutograd, recv_));
}
} // namespace

namespace ops {

c10::intrusive_ptr<Work> broadcast(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const BroadcastOptions& opts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::broadcast_", "")
          .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
              at::TensorList,
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,
              int64_t,
              int64_t,
              int64_t)>();
  // It's awakward to unbox the opts here and box them again in the custom C++
  // op. But it's also complicated to make opts as a CustomClassHolder. Leave it
  // as it is now.
  return std::get<1>(op.call(
      tensors,
      process_group,
      opts.rootRank,
      opts.rootTensor,
      opts.timeout.count()));
}

c10::intrusive_ptr<Work> allreduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const AllreduceOptions& opts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::allreduce_", "")
          .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
              at::TensorList,
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,
              const c10::intrusive_ptr<::c10d::ReduceOp>&,
              int64_t)>();

  return std::get<1>(op.call(
      tensors,
      process_group,
      c10::make_intrusive<ReduceOp>(opts.reduceOp),
      opts.timeout.count()));
}

c10::intrusive_ptr<Work> allreduce_coalesced(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const AllreduceCoalescedOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::allreduce_coalesced_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           const c10::intrusive_ptr<::c10d::ReduceOp>&,
                           int64_t)>();

  return op.call(
      tensors,
      process_group,
      c10::make_intrusive<ReduceOp>(opts.reduceOp),
      opts.timeout.count());
}

c10::intrusive_ptr<Work> allgather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const AllgatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::allgather_", "")
                       .typed<std::tuple<
                           std::vector<std::vector<at::Tensor>>,
                           c10::intrusive_ptr<Work>>(
                           const std::vector<std::vector<at::Tensor>>&,
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t)>();

  return std::get<1>(op.call(
      output_tensors, input_tensors, process_group, opts.timeout.count()));
}

c10::intrusive_ptr<Work> reduce_scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::reduce_scatter_", "")
          .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
              const std::vector<at::Tensor>&,
              const std::vector<std::vector<at::Tensor>>&,
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,
              const c10::intrusive_ptr<::c10d::ReduceOp>&,
              int64_t)>();
  return std::get<1>(op.call(
      output_tensors,
      input_tensors,
      process_group,
      c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
      opts.timeout.count()));
}

c10::intrusive_ptr<Work> reduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const ReduceOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::reduce_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           const c10::intrusive_ptr<::c10d::ReduceOp>&,
                           int64_t,
                           int64_t,
                           int64_t)>();
  return op.call(
      tensors,
      process_group,
      c10::make_intrusive<ReduceOp>(opts.reduceOp),
      opts.rootRank,
      opts.rootTensor,
      opts.timeout.count());
}

c10::intrusive_ptr<Work> gather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const GatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::gather_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           const std::vector<std::vector<at::Tensor>>&,
                           const std::vector<at::Tensor>&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(
      output_tensors,
      input_tensors,
      process_group,
      opts.rootRank,
      opts.timeout.count());
}

c10::intrusive_ptr<Work> scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ScatterOptions& opts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::scatter_", "")
          .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
              const std::vector<at::Tensor>&,
              const std::vector<std::vector<at::Tensor>>&,
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,
              int64_t,
              int64_t)>();
  return std::get<1>(op.call(
      output_tensors,
      input_tensors,
      process_group,
      opts.rootRank,
      opts.timeout.count()));
}

c10::intrusive_ptr<Work> alltoall(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList output_tensors,
    at::TensorList input_tensors,
    const AllToAllOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::alltoall_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t)>();
  return op.call(
      output_tensors, input_tensors, process_group, opts.timeout.count());
}

c10::intrusive_ptr<Work> barrier(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const BarrierOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::barrier", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::Tensor,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           const std::vector<int64_t>&,
                           int64_t)>();

  // Default to using cpu implementation
  at::Tensor tensor = at::empty({0}, at::TensorOptions().device(at::kCPU));
  // if opts.device_ids or backend is nccl are specified then use cuda
  // implementation
  // TODO: getBackendName() is always "NOT DEFINED"
  if (opts.device_ids.size() > 0 || process_group->getBackendName() == "nccl") {
    // set cuda tensor
    tensor = at::empty(
        {0}, at::TensorOptions().device(at::kCUDA, opts.device_ids[0]));
  }
  return op.call(tensor, process_group, opts.device_ids, opts.timeout.count());
}

c10::intrusive_ptr<Work> send(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t dstRank,
    int64_t tag) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::send", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(tensors, process_group, dstRank, tag);
}

c10::intrusive_ptr<Work> recv(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t srcRank,
    int64_t tag) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::recv_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(tensors, process_group, srcRank, tag);
}

} // namespace ops
} // namespace c10d
