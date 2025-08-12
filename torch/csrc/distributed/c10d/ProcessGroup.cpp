#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/RankLocal.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>

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
    case OpType::COALESCED:
      return "COALESCED";
    case OpType::_ALLREDUCE_SPARSE:
      return "_ALLREDUCE_SPARSE";
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
  ProcessGroup::BackendType backendType{ProcessGroup::BackendType::UNDEFINED};
  try {
    backendType = deviceTypeToBackendType_.at(deviceType);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(
        false, "No backend type associated with device type ", deviceType);
  }

  // Check if the backend has already been initialized
  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto backend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = backend;
    return backend;
  }

  TORCH_CHECK(
      false,
      "Could not retrieve or create the backend ",
      backendType,
      " for device type ",
      deviceType);
}

ProcessGroup::ProcessGroup(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int size)
    : Communicator(rank, size),
      store_(std::move(store)),
      rank_(rank),
      size_(size),
      backendType_(BackendType::UNDEFINED),
      dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::ProcessGroup(int rank, int size)
    : Communicator(rank, size),
      rank_(rank),
      size_(size),
      backendType_(BackendType::UNDEFINED) {}

ProcessGroup::~ProcessGroup() = default;

void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(
      fmt::format("c10d.process_group_{}", getBackendName()));
}

const std::string& ProcessGroup::getGroupName() const {
  TORCH_CHECK(!deviceTypeToBackend_.empty(), "ProcessGroup name not set");
  return deviceTypeToBackend_.begin()->second->getGroupUid();
}

void ProcessGroup::setGroupName(const std::string& name) {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupUid(name);
  }
}

const std::string& ProcessGroup::getGroupDesc() const {
  return pg_desc_;
}

void ProcessGroup::setGroupDesc(const std::string& name) {
  pg_desc_ = name;
  // Also set the group desc for all backends
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupDesc(name);
  }
}

void ProcessGroup::enableCollectivesTiming() {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->enableCollectivesTiming();
  }
}

void ProcessGroup::release_resources() {
  store_.reset();
  deviceTypeToBackend_.clear();
  backendTypeToBackend_.clear();
}

c10::intrusive_ptr<ProcessGroup> ProcessGroup::splitGroup(
    const std::vector<int>& ranks,
    const std::optional<std::chrono::milliseconds>& timeout,
    const std::optional<c10::intrusive_ptr<Backend::Options>>& opts,
    const std::optional<std::string>& name,
    const std::optional<std::string>& desc) {
  TORCH_CHECK(
      ranks.size() > 0,
      "Split ranks cannot be empty. Please provide a non-empty list of ranks to split the group.");
  TORCH_CHECK(
      ranks.size() <= static_cast<size_t>(size_),
      "the split group's size should be no larger than the world_size set by init_process_group");
  std::set<int> ranks_set(ranks.begin(), ranks.end());
  TORCH_CHECK(
      ranks_set.size() == ranks.size(),
      "Split ranks should not have duplicates. Please provide a list of unique ranks to split the group.");
  std::vector<int> sorted_ranks = ranks;
  std::sort(sorted_ranks.begin(), sorted_ranks.end());
  c10::intrusive_ptr<ProcessGroup> newGroup;
  std::string groupName = name.has_value()
      ? name.value()
      : c10::str(getGroupName(), ":split:", fmt::format("{}", sorted_ranks));
  c10::intrusive_ptr<Store> store = c10::static_intrusive_pointer_cast<Store>(
      c10::make_intrusive<PrefixStore>(
          fmt::format("{}/", groupName), store_->clone()));
  std::string groupDesc = desc.has_value()
      ? desc.value()
      : c10::str(getGroupDesc(), ":split:", incrementSplitCount());
  for (const auto& pair : deviceTypeToBackendType_) {
    c10::DeviceType deviceType = pair.first;
    BackendType backendType = pair.second;

    auto parentBackend = getBackend(deviceType);
    auto backendOpts =
        opts.has_value() ? opts.value() : parentBackend->getBackendOptions();
    backendOpts->group_name = groupName;
    backendOpts->timeout =
        timeout.has_value() ? timeout.value() : backendOpts->timeout;
    auto splitBackend = parentBackend->split(store, sorted_ranks, backendOpts);
    if (splitBackend == nullptr) {
      continue;
    }
    splitBackend->setGroupDesc(groupDesc);
    if (!newGroup) {
      newGroup = c10::make_intrusive<ProcessGroup>(
          store, splitBackend->getRank(), splitBackend->getSize());
      newGroup->setDefaultBackend(backendType_);
    }
    newGroup->setBackend(deviceType, backendType, splitBackend);
  }

  if (!newGroup) {
    return nullptr;
  }
  newGroup->setGroupName(groupName);
  newGroup->setGroupDesc(groupDesc);
  return newGroup;
}

c10::intrusive_ptr<ProcessGroup> ProcessGroup::mergeRemoteGroup(
    const c10::intrusive_ptr<Store>& store,
    const MergeOptions& opts,
    const int& size) {
  c10::intrusive_ptr<ProcessGroup> newGroup;
  // We assume rank number is within the range of int32_t, so it won't overflow.
  int rank = static_cast<int>(store->add("mergeGroupRank", 1) - 1);
  // TODO: Do we need to check all groups have same deviceTypeToBackendType_?
  std::string groupName = opts.group_name.has_value()
      ? opts.group_name.value()
      : c10::str(getGroupName(), ":merge");
  std::string groupDesc = opts.group_desc.has_value()
      ? opts.group_desc.value()
      : c10::str(getGroupDesc(), ":merge");
  for (const auto& pair : deviceTypeToBackendType_) {
    c10::DeviceType deviceType = pair.first;
    BackendType backendType = pair.second;
    auto parentBackend = getBackend(deviceType);
    auto backendOpts = parentBackend->getBackendOptions();
    backendOpts->group_name = groupName;
    backendOpts->timeout = opts.timeout;
    auto mergedBackend = parentBackend->merge(store, backendOpts, rank, size);
    mergedBackend->setGroupDesc(groupDesc);

    // Historically, we have been using one process_group to map to all
    // backends. but in our new design, we will have one process_group per
    // backend. This logic is mostly for backward compatibility.
    if (!newGroup) {
      newGroup = c10::make_intrusive<ProcessGroup>(store, rank, size);
      newGroup->setDefaultBackend(backendType_);
    }
    newGroup->setBackend(deviceType, backendType, mergedBackend);
  }

  if (!newGroup) {
    return nullptr;
  }
  newGroup->setGroupName(groupName);
  newGroup->setGroupDesc(groupDesc);
  return newGroup;
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allreduceImpl(
    std::vector<at::Tensor>& tensors,
    c10d::ReduceOp reduceOp,
    bool asyncOp,
    std::chrono::milliseconds timeout,
    std::optional<at::Tensor> sparseIndices) {
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support allreduce"));
}

c10::intrusive_ptr<ProcessGroup>& currentProcessGroup() {
  thread_local static c10::intrusive_ptr<ProcessGroup> pg = nullptr;
  return pg;
}

void setProcessGroup(c10::intrusive_ptr<ProcessGroup> pg) {
  currentProcessGroup() = std::move(pg);
}

} // namespace c10d
