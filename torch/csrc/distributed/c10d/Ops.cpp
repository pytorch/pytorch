#include <torch/csrc/distributed/c10d/Ops.hpp>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>

namespace c10d {
namespace {

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

c10::intrusive_ptr<Work> _allgather_base_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  return process_group->_allgather_base(output_tensor, input_tensor);
}

c10::intrusive_ptr<Work> allgather_coalesced_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto input_list_vec = input_list.vec();
  return process_group->allgather_coalesced(
      const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
      input_list_vec);
}

c10::intrusive_ptr<Work> _reduce_scatter_base_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  return process_group->_reduce_scatter_base(
      output_tensor,
      input_tensor,
      ReduceScatterOptions{
          *reduce_op.get(), std::chrono::milliseconds(timeout)});
}

void monitored_barrier_(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout,
    bool wait_all_ranks) {
  process_group->monitoredBarrier(
      BarrierOptions{device_ids, std::chrono::milliseconds(timeout)},
      wait_all_ranks);
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

c10::intrusive_ptr<Work> recv_any_source_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->recvAnysource(tensor_vec, static_cast<int>(tag));
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
      "broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int root_tensor, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "allreduce_coalesced_",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, allreduce_coalesced_));
  m.def(
      "allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int timeout) -> (Tensor[][], __torch__.torch.classes.c10d.Work)");
  m.def(
      "_allgather_base_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, _allgather_base_));
  m.def(
      "allgather_coalesced_",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, allgather_coalesced_));
  m.def(
      "reduce_scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "_reduce_scatter_base_",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, _reduce_scatter_base_));
  m.def(
      "reduce_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, reduce_));
  m.def(
      "gather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
  m.def(
      "alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] output_split_sizes, int[] input_split_sizes, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, int timeout) -> __torch__.torch.classes.c10d.Work");
  m.def(
      "monitored_barrier_",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, monitored_barrier_));
  m.def("send", dispatch(c10::DispatchKey::CompositeExplicitAutograd, send));
  m.def("recv_", dispatch(c10::DispatchKey::CompositeExplicitAutograd, recv_));
  m.def(
      "recv_any_source_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, recv_any_source_));
}
} // namespace

namespace ops {

c10::intrusive_ptr<Work> broadcast(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const BroadcastOptions& opts) {
  // TODO: handles the case of using a PythonProcessGroup which is used in
  // Reducer.cpp This can be removed once
  // https://github.com/pytorch/pytorch/issues/90659 is resolved
  if (!process_group->hasBackends()) {
    auto tensor_vec = tensors.vec();
    return process_group->broadcast(tensor_vec, opts);
  }

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
  // TODO: handles the case of using a PythonProcessGroup which is used in
  // Reducer.cpp This can be removed once
  // https://github.com/pytorch/pytorch/issues/90659 is resolved
  if (!process_group->hasBackends()) {
    auto tensor_vec = tensors.vec();
    return process_group->allreduce(tensor_vec, opts);
  }

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
  // TODO: handles the case of using a PythonProcessGroup which is used in
  // Reducer.cpp This can be removed once
  // https://github.com/pytorch/pytorch/issues/90659 is resolved
  if (!process_group->hasBackends()) {
    auto input_tensors_vec = input_tensors.vec();
    return process_group->allgather(
        const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
        input_tensors_vec,
        opts);
  }

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

c10::intrusive_ptr<Work> _allgather_base(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::_allgather_base_", "")
                       .typed<c10::intrusive_ptr<Work>(
                           at::Tensor&,
                           at::Tensor&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&)>();

  return op.call(output_tensor, input_tensor, process_group);
}

c10::intrusive_ptr<Work> allgather_coalesced(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const AllgatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::allgather_coalesced_", "")
                       .typed<c10::intrusive_ptr<Work>(
                           const std::vector<std::vector<at::Tensor>>&,
                           const at::TensorList&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&)>();

  return op.call(output_lists, input_list, process_group);
}

c10::intrusive_ptr<Work> reduce_scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::reduce_scatter_", "")
          .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
              const at::TensorList&,
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

c10::intrusive_ptr<Work> _reduce_scatter_base(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const ReduceScatterOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::_reduce_scatter_base_", "")
                       .typed<c10::intrusive_ptr<Work>(
                           at::Tensor&,
                           at::Tensor&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           const c10::intrusive_ptr<::c10d::ReduceOp>&,
                           int64_t)>();
  return op.call(
      output_tensor,
      input_tensor,
      process_group,
      c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
      opts.timeout.count());
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
    const at::TensorList& input_tensors,
    const GatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::gather_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           const std::vector<std::vector<at::Tensor>>&,
                           const at::TensorList&,
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
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ScatterOptions& opts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::scatter_", "")
          .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
              const at::TensorList&,
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

c10::intrusive_ptr<Work> alltoall_base(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::Tensor& output,
    at::Tensor& input,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    const AllToAllOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::alltoall_base_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::Tensor&,
                           at::Tensor&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           std::vector<int64_t>,
                           std::vector<int64_t>,
                           int64_t)>();
  return op.call(
      output,
      input,
      process_group,
      output_split_sizes,
      input_split_sizes,
      opts.timeout.count());
}

void monitored_barrier(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const BarrierOptions& opts,
    bool wait_all_ranks) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::monitored_barrier_", "")
                       .typed<void(
                           at::Tensor,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           const std::vector<int64_t>&,
                           int64_t,
                           bool)>();
  // Default to using cpu implementation, monitored barrier is only for GLOO
  at::Tensor tensor = at::empty({0}, at::TensorOptions().device(at::kCPU));
  op.call(
      tensor,
      process_group,
      opts.device_ids,
      opts.timeout.count(),
      wait_all_ranks);
}

c10::intrusive_ptr<Work> barrier(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const BarrierOptions& opts) {
  static at::Tensor tensor;
  // TODO: if nccl was specified then use it
  if (process_group->getBackendType() ==
      c10d::ProcessGroup::BackendType::NCCL) {
    // set cuda tensor
    tensor = at::empty(
        {1}, at::TensorOptions().device(at::DeviceType::CUDA).dtype(at::kByte));
  } else {
    // Default to using cpu implementation
    tensor = at::empty(
        {1}, at::TensorOptions().device(at::DeviceType::CPU).dtype(at::kByte));
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::barrier", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::Tensor,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           const std::vector<int64_t>&,
                           int64_t)>();

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

c10::intrusive_ptr<Work> recv_any_source(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t tag) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::recv_any_source_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t)>();
  return op.call(tensors, process_group, tag);
}

} // namespace ops
} // namespace c10d
