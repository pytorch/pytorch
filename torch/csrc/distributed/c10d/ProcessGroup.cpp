#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>

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

c10::intrusive_ptr<Backend> ProcessGroup::getBackend(
    c10::DeviceType deviceType) const {
  // TORCH_CHECK(
  //     deviceTypeToBackends_.find(deviceType) != deviceTypeToBackends_.end(),
  //     "Backend type for device type ",
  //     deviceType,
  //     " was not found");

  // // get backend type from device type
  // ProcessGroup::BackendType backendType =
  // deviceTypeToBackendType_.at(deviceType);

  // // backendType is in map
  // if (backendTypeToBackends_.find(backendType) !=
  // backendTypeToBackends_.end()) {
  //   return backendTypeToBackends_.at(backendType);
  // }

  // if (backendType == ProcessGroup::BackendType::MPI) {
  //   auto backend = c10::make_intrusive<ProcessGroupMPI>();
  // } else if (backendType == ProcessGroup::BackendType::GLOO) {
  //   return c10::make_intrusive<ProcessGroupGloo>();
  // } else if (backendType == ProcessGroup::BackendType::NCCL) {
  //   return c10::make_intrusive<ProcessGroupNCCL>();
  // } else if (backendType == ProcessGroup::BackendType::UCC) {
  //   return c10::make_intrusive<ProcessGroupUCC>();
  // } else {
  //   TORCH_CHECK(false, "Unknown backend type: ", backendType);
  // }

  TORCH_CHECK(
      deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end(),
      "Backend type for device type ",
      deviceType,
      " was not found");

  return deviceTypeToBackend_.at(deviceType);
}

ProcessGroup::ProcessGroup(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : store_(store),
      rank_(rank),
      size_(size),
      options_(options),
      backendType_(backendTypeResolver(options->backend)),
      dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");

  std::cout << "creating process group" << std::endl;
  // parseBackendStr(options_->backend);
  std::cout << "finished parsing backend str" << std::endl;
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), backendType_(BackendType::UNDEFINED) {}

ProcessGroup::~ProcessGroup() {}

void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(
      fmt::format("c10d.process_group_{}", getBackendName()));
}
} // namespace c10d
