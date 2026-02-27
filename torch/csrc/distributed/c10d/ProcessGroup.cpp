#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/RankLocal.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

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
    case OpType::REDUCE_SCATTER_TENSOR_COALESCED:
      return "REDUCE_SCATTER_TENSOR_COALESCED";
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
  } catch (const std::out_of_range&) {
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
    : store_(std::move(store)),
      rank_(rank),
      size_(size),
      backendType_(BackendType::UNDEFINED),
      dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), backendType_(BackendType::UNDEFINED) {}

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
      !ranks.empty(),
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
    backendOpts->group_desc = groupDesc;
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

} // namespace c10d

namespace {

class WorkRegistry {
 public:
  void register_work(
      const at::Tensor& tensor,
      const c10::intrusive_ptr<c10d::Work>& work) {
    if (!tensor.has_storage()) {
      TORCH_WARN_ONCE(
          "Registering collective work for tensor without storage is not supported. "
          "Calling c10d_functional.wait_tensor() on this tensor will not wait for the collective to complete. "
          "Unsupported tensor type: " +
          tensor.toString());
      return;
    }
    auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);

    auto it = registry_.find(storage);
    if (it == registry_.end()) {
      registry_.emplace(
          std::move(storage),
          std::vector<c10::intrusive_ptr<c10d::Work>>{work});
    } else {
      // There is no guarantee that the previous work object for this
      // tensor storage is completed before the new work object is registered.
      // Therefore we need to maintain a list of work objects for each tensor
      // storage.

      // Check if work is already in the list
      bool work_exists = false;
      for (const auto& existing_work : it->second) {
        if (existing_work == work) {
          work_exists = true;
          break;
        }
      }

      // Only append if work is not already in the list
      if (!work_exists) {
        it->second.push_back(work);
      }
    }
  }

  std::vector<c10::intrusive_ptr<c10d::Work>> pop_works(
      const at::Tensor& tensor) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto it = registry_.find(storage);
    if (it == registry_.end()) {
      return {};
    }
    auto works = it->second;
    registry_.erase(it);
    return works;
  }

  void unregister_work(const c10::intrusive_ptr<c10d::Work>& work) {
    std::unique_lock lock(lock_);
    for (auto it = registry_.begin(); it != registry_.end();) {
      std::vector<c10::intrusive_ptr<c10d::Work>> nonmatching_works;
      for (const auto& _work : it->second) {
        if (_work != work) {
          nonmatching_works.push_back(_work);
        }
      }
      if (nonmatching_works.empty()) {
        it = registry_.erase(it);
      } else {
        it->second = std::move(nonmatching_works);
        ++it;
      }
    }
  }

  size_t get_work_registry_size() {
    std::unique_lock lock(lock_);
    size_t total_size = 0;
    for (const auto& [storage, works] : registry_) {
      total_size += works.size();
    }
    return total_size;
  }

  void set_allow_inflight_collective_as_graph_input(bool value) {
    std::unique_lock lock(lock_);
    allow_inflight_collective_as_graph_input_ = value;
  }

  bool allow_inflight_collective_as_graph_input() {
    std::unique_lock lock(lock_);
    return allow_inflight_collective_as_graph_input_;
  }

  ~WorkRegistry() {
    // If there are still unwaited work objects, their corresponding process
    // groups should have already been destroyed at this stage. Any attempts to
    // wait for these work objects or to destroy them will only result in
    // confusing errors. Therefore, we simply issue a warning and intentionally
    // allow the unwaited work objects to leak.
    size_t registry_size = get_work_registry_size();
    if (registry_size > 0) {
      TORCH_WARN(
          "At the time of process termination, there are still ",
          registry_size,
          " unwaited collective calls. "
          "Please review your program to ensure that:\n"
          "1. c10d_functional.wait_tensor() is invoked on all tensors returned from c10d_functional collective,\n"
          "2. c10d_functional.wait_tensor() is invoked on all output tensors of async_op=True torch.distributed collective "
          "called under `with allow_inflight_collective_as_graph_input_ctx():`,\n"
          "before the output tensors of the collective are used.");
    }
    for (auto& it : registry_) {
      for (auto& work : it.second) {
        work.release();
      }
    }
  }

 private:
  std::unordered_map<
      c10::weak_intrusive_ptr<c10::StorageImpl>,
      std::vector<c10::intrusive_ptr<c10d::Work>>>
      registry_;
  bool allow_inflight_collective_as_graph_input_ = false;
  std::mutex lock_;
};

static WorkRegistry process_registry;

} // namespace

namespace c10d {

void register_work(
    const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::Work>& work) {
  RankLocal<WorkRegistry>::get().register_work(tensor, work);
}

at::Tensor wait_tensor(const at::Tensor& tensor) {
  // First try to find work in the current thread's registry (fast path)
  auto works = RankLocal<WorkRegistry>::get().pop_works(tensor);

  // If no work found in current thread's registry, search all registries.
  // This handles the case where wait() is called from a different thread
  // than where the collective was initiated (e.g., user-created threads).
  if (works.empty()) {
    auto result = RankLocal<WorkRegistry>::find_across_all(
        [&tensor](WorkRegistry& registry)
            -> std::optional<std::vector<c10::intrusive_ptr<c10d::Work>>> {
          auto w = registry.pop_works(tensor);
          if (!w.empty()) {
            return w;
          }
          return std::nullopt;
        });
    if (result.has_value()) {
      works = std::move(result.value());
    }
  }

  for (const auto& work : works) {
    work->wait();
  }
  return tensor;
}

void unregister_work(const c10::intrusive_ptr<c10d::Work>& work) {
  RankLocal<WorkRegistry>::get().unregister_work(work);
}

size_t get_work_registry_size() {
  return RankLocal<WorkRegistry>::get().get_work_registry_size();
}

void set_allow_inflight_collective_as_graph_input(bool value) {
  return RankLocal<WorkRegistry>::get()
      .set_allow_inflight_collective_as_graph_input(value);
}

bool allow_inflight_collective_as_graph_input() {
  return RankLocal<WorkRegistry>::get()
      .allow_inflight_collective_as_graph_input();
}

c10::intrusive_ptr<ProcessGroup>& currentProcessGroup() {
  thread_local static c10::intrusive_ptr<ProcessGroup> pg = nullptr;
  return pg;
}

void setProcessGroup(c10::intrusive_ptr<ProcessGroup> pg) {
  currentProcessGroup() = std::move(pg);
}

} // namespace c10d
