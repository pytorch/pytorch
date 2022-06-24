#include <torch/csrc/distributed/c10d/Ops.hpp>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace c10d {
namespace {
c10::intrusive_ptr<ProcessGroup::Work> broadcast_(
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

c10::intrusive_ptr<ProcessGroup::Work> allreduce_(
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

c10::intrusive_ptr<ProcessGroup::Work> allgather_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  return process_group->allgather(
      const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
      const_cast<std::vector<at::Tensor>&>(input_tensors),
      AllgatherOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t reduce_op,
    int64_t timeout) {
  return process_group->reduce_scatter(
      const_cast<std::vector<at::Tensor>&>(output_tensors),
      const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
      ReduceScatterOptions{
          static_cast<ReduceOp>(reduce_op),
          std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<ProcessGroup::Work> reduce_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->reduce(
      tensor_vec,
      ReduceOptions{
          static_cast<ReduceOp>(reduce_op),
          root_rank,
          root_tensor,
          std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<ProcessGroup::Work> gather_(
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

c10::intrusive_ptr<ProcessGroup::Work> scatter_(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  return process_group->scatter(
      const_cast<std::vector<at::Tensor>&>(output_tensors),
      const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
      ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY(c10d, m) {
  // The following ProcessGroup and Work definations are more like declarations.
  // They don't expose the details of the two classes into TorchScript.
  m.class_<ProcessGroup>("ProcessGroup").def(torch::init<int64_t, int64_t>());
  m.class_<ProcessGroup::Work>("Work").def(torch::init<>());
  // It's important to register the op to the CompositeExplicitAutograd key to
  // enable
  // __torch_dispatch__.
  m.def(
      "broadcast_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, broadcast_));
  m.def(
      "allreduce_",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, allreduce_));
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
}
} // namespace

namespace ops {

c10::intrusive_ptr<ProcessGroup::Work> broadcast(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const BroadcastOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::broadcast_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t,
                           int64_t)>();
  // It's awakward to unbox the opts here and box them again in the custom C++
  // op. But it's also complicated to make opts as a CustomClassHolder. Leave it
  // as it is now.
  return op.call(
      tensors,
      process_group,
      opts.rootRank,
      opts.rootTensor,
      opts.timeout.count());
}

c10::intrusive_ptr<ProcessGroup::Work> allreduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const AllreduceOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::allreduce_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(
      tensors,
      process_group,
      static_cast<uint64_t>(opts.reduceOp),
      opts.timeout.count());
}

c10::intrusive_ptr<ProcessGroup::Work> allgather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const AllgatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::allgather_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
                           const std::vector<std::vector<at::Tensor>>&,
                           const std::vector<at::Tensor>&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t)>();
  return op.call(
      output_tensors, input_tensors, process_group, opts.timeout.count());
}

c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::reduce_scatter_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
                           const std::vector<at::Tensor>&,
                           const std::vector<std::vector<at::Tensor>>&,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(
      output_tensors,
      input_tensors,
      process_group,
      static_cast<uint64_t>(opts.reduceOp),
      opts.timeout.count());
}

c10::intrusive_ptr<ProcessGroup::Work> reduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const ReduceOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::reduce_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t)>();
  return op.call(
      tensors,
      process_group,
      static_cast<uint64_t>(opts.reduceOp),
      opts.rootRank,
      opts.rootTensor,
      opts.timeout.count());
}

c10::intrusive_ptr<ProcessGroup::Work> gather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const GatherOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::gather_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
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

c10::intrusive_ptr<ProcessGroup::Work> scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ScatterOptions& opts) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::scatter_", "")
                       .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(
                           const std::vector<at::Tensor>&,
                           const std::vector<std::vector<at::Tensor>>&,
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

} // namespace ops
} // namespace c10d
