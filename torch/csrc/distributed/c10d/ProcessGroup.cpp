#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>

namespace c10d {

ProcessGroup::BackendType strToBackendType(std::string backend) {
  if (backend == "undefined") {
    return ProcessGroup::BackendType::UNDEFINED;
  } else if (backend == "gloo") {
    return ProcessGroup::BackendType::GLOO;
  } else if (backend == "nccl") {
    return ProcessGroup::BackendType::NCCL;
  } else if (backend == "ucc") {
    return ProcessGroup::BackendType::UCC;
  } else if (backend == "mpi") {
    return ProcessGroup::BackendType::MPI;
  } else if (backend == "tcp") {
    return ProcessGroup::BackendType::TCP;
  } else {
    return ProcessGroup::BackendType::CUSTOM;
  }
}

std::string backendTypeToStr(ProcessGroup::BackendType backendType) {
  switch (backendType) {
    case ProcessGroup::BackendType::UNDEFINED:
      return "undefined";
    case ProcessGroup::BackendType::GLOO:
      return "gloo";
    case ProcessGroup::BackendType::NCCL:
      return "nccl";
    case ProcessGroup::BackendType::UCC:
      return "ucc";
    case ProcessGroup::BackendType::MPI:
      return "mpi";
    case ProcessGroup::BackendType::TCP:
      return "tcp";
    case ProcessGroup::BackendType::CUSTOM:
      return "custom";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown backend type");
  }
}

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
    c10::DeviceType deviceType) {
  // If there is a backend associated with this device type then return it
  if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
    return deviceTypeToBackend_.at(deviceType);
  }

  // Get the backend type associated with the device
  TORCH_CHECK(
      deviceTypeToBackendType_.find(deviceType) !=
          deviceTypeToBackendType_.end(),
      "No backend type associated with device type ",
      deviceType);
  ProcessGroup::BackendType backendType =
      deviceTypeToBackendType_.at(deviceType);

  // Check if the backend has already been initialized
  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto backend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = backend;
    return backend;
  } else {
    // TODO: finish implementing this for lazy initialization
#ifdef USE_C10D_GLOO
    std::cout << "C++ initializing the backend for "
              << backendTypeToStr(backendType) << std::endl;
    // initialize the backend

    // create a separate prefix store for the newly created backend
    auto prefixStore = c10::make_intrusive<PrefixStore>(
        backendTypeToStr(backendType) + "/", store_);
    c10::intrusive_ptr<Backend> backend;

    // TODO: we should move this into its own function for each respective
    // backend if backend is gloo then initialize a gloo backend
    if (backendType == ProcessGroup::BackendType::GLOO) {
      // create ProcessGroupGloo options
      backend = ProcessGroupGloo::createProcessGroupGloo(
          prefixStore, rank_, size_, options_->timeout);
    }

    // if backend is nccl or gloo then set sequence number
    if (backendType == ProcessGroup::BackendType::NCCL ||
        backendType == ProcessGroup::BackendType::GLOO) {
      backend->setSequenceNumberForGroup();
    }

    // if using torch_distributed_debug, then wrap backend with
    // ProcessGroupWrapper
    if (dist_debug_level_ >= DebugLevel::Detail) {
      // create new prefix store for the wrapper
      auto wrapperPrefixStore = c10::make_intrusive<PrefixStore>(
          backendTypeToStr(backendType) + "/wrapper", store_);
      auto wrapperBackend = ProcessGroupGloo::createProcessGroupGloo(
          wrapperPrefixStore, rank_, size_, options_->timeout);
      backend =
          c10::make_intrusive<ProcessGroupWrapper>(backend, wrapperBackend);
    }

    // set internal state and return
    deviceTypeToBackend_[deviceType] = backend;
    backendTypeToBackend_[backendType] = backend;
    return backend;

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
#endif // USE_C10D_GLOO
  }

  TORCH_CHECK(
      false,
      "Could not retrieve or create the backend ",
      backendType,
      " for device type ",
      deviceType);
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
      backendType_(strToBackendType(options->backend)),
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
