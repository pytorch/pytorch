#include <ATen/ThreadLocalState.h>
#include <c10d/ProcessGroup.hpp>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Logging.h>
#include <fmt/format.h>

namespace c10d {

std::string opTypeToString(OpType opType) {
  switch (opType) {
    case OpType::BROADCAST:
      return "BROADCAST";
    case OpType::ALLREDUCE:
      return "ALLREDUCE";
    case OpType::ALLREDUCE_COALESCED:
      return "ALLREDUCE_COALESCED";
    case OpType::REDUCE:
      return "REDUCE";
    case OpType::ALLGATHER:
      return "ALLGATHER";
    case OpType::_ALLGATHER_BASE:
      return "_ALLGATHER_BASE";
    case OpType::ALLGATHER_COALESCED:
      return "ALLGATHER_COALESCED";
    case OpType::GATHER:
      return "GATHER";
    case OpType::SCATTER:
      return "SCATTER";
    case OpType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case OpType::ALLTOALL_BASE:
      return "ALLTOALL_BASE";
    case OpType::ALLTOALL:
      return "ALLTOALL";
    case OpType::SEND:
      return "SEND";
    case OpType::RECV:
      return "RECV";
    case OpType::RECVANYSOURCE:
      return "RECVANYSOURCE";
    case OpType::BARRIER:
      return "BARRIER";
    case OpType::UNKNOWN:
      return "UNKNOWN";
    case OpType::_REDUCE_SCATTER_BASE:
      return "_REDUCE_SCATTER_BASE";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown op type!");
  }
  return "UNKNOWN";
}

bool isP2POp(OpType opType, bool batchP2P /*= false*/) {
  if (batchP2P)
    return false;
  return opType == OpType::SEND || opType == OpType::RECV ||
      opType == OpType::RECVANYSOURCE;
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");

  // TODO: figure out a way to pass store into the default process group
  // c10::intrusive_ptr<TCPStore> store = c10::make_intrusive<TCPStore>(
  //     "localhost",
  //     29500,
  //     1,
  //     true,
  //     std::chrono::milliseconds(::c10d::Store::kDefaultTimeout),
  //     false);

  // set up our list of backends
  setBackend(
      c10::DeviceType::CPU,
      c10::make_intrusive<DummyProcessGroupBackend>(rank, size));
  setBackend(
      c10::DeviceType::CUDA,
      c10::make_intrusive<DummyProcessGroupBackend>(rank, size));
}

ProcessGroup::~ProcessGroup() {}

void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(
      fmt::format("c10d.process_group_{}", getBackendName()));
}

c10::intrusive_ptr<Work> _dummy_broadcast_cpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  VLOG(1) << "in _dummy_broadcast_cpu_";
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CPU)
      ->broadcast(
          tensor_vec,
          BroadcastOptions{
              root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> _dummy_broadcast_cuda_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  VLOG(1) << "in _dummy_broadcast_cuda_";
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::CUDA)
      ->broadcast(
          tensor_vec,
          BroadcastOptions{
              root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> ProcessGroup::_DummyBroadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  VLOG(1) << "in _DummyBroadcast";
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::dummy_broadcast_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t,
                           int64_t)>();
  VLOG(1) << "calling op";
  // It's awakward to unbox the opts here and box them again in the custom C++
  // op. But it's also complicated to make opts as a CustomClassHolder. Leave it
  // as it is now.
  return op.call(
      tensors,
      c10::make_intrusive<::c10d::ProcessGroup>(*this),
      opts.rootRank,
      opts.rootTensor,
      opts.timeout.count());
}

// register functions to dispatcher
namespace {
TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("dummy_broadcast_", _dummy_broadcast_cpu_);
}

TORCH_LIBRARY_IMPL(c10d, CUDA, m) {
  m.impl("dummy_broadcast_", _dummy_broadcast_cuda_);
}
} // namespace

} // namespace c10d
