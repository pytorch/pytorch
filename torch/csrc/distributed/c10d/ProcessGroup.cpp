#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/RankLocal.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <typeinfo>
#include <unordered_set>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace c10d {

namespace {

// Options handed to a child backend must never alias the parent's (or the
// caller's) object: Backend::getBackendOptions() returns the backend's live
// options_, and split()/merge() implementations write into what they are given
// (ProcessGroupGloo::split overwrites global_ranks_in_group,
// ProcessGroupNCCL::split also sets split_from/split_color). Sharing one object
// leaves the parent describing its child's ranks, which the parent's next split
// then indexes out of bounds.
c10::intrusive_ptr<Backend::Options> cloneOptions(
    const c10::intrusive_ptr<Backend::Options>& opts) {
  auto copy = opts->clone();
  // An out-of-tree Options subclass that does not override clone() would be
  // sliced. Keep the legacy aliasing behavior for it rather than handing its
  // backend an object of the wrong type.
  if (copy == nullptr) {
    return opts;
  }
  const Backend::Options& copyRef = *copy;
  const Backend::Options& optsRef = *opts;
  if (typeid(copyRef) != typeid(optsRef)) {
    return opts;
  }
  return copy;
}

} // namespace

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
    case OpType::ALLGATHER_INTO_TENSOR_COALESCED:
      return "ALLGATHER_INTO_TENSOR_COALESCED";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown op type!");
  }
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
  if (auto it = deviceTypeToBackend_.find(deviceType);
      it != deviceTypeToBackend_.end()) {
    return it->second;
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
  if (auto it = backendTypeToBackend_.find(backendType);
      it != backendTypeToBackend_.end()) {
    auto backend = it->second;
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
    const std::optional<std::string>& desc,
    const std::optional<std::vector<c10::Device>>& devices) {
  TORCH_CHECK(
      !ranks.empty(),
      "Split ranks cannot be empty. Please provide a non-empty list of ranks to split the group.");
  TORCH_CHECK(
      ranks.size() <= static_cast<size_t>(getSize()),
      "the split group's size should be no larger than the world_size set by init_process_group");
  std::unordered_set<int> ranks_set(ranks.begin(), ranks.end());
  TORCH_CHECK(
      ranks_set.size() == ranks.size(),
      "Split ranks should not have duplicates. Please provide a list of unique ranks to split the group.");
  std::set<c10::DeviceType> deviceTypeFilter;
  for (const auto& d : devices.value_or(std::vector<c10::Device>{})) {
    deviceTypeFilter.insert(d.type());
  }
  if (!deviceTypeFilter.empty()) {
    for (const auto& deviceType : deviceTypeFilter) {
      TORCH_CHECK(
          deviceTypeToBackendType_.contains(deviceType),
          "Requested device type for splitGroup is not present in the parent process group: ",
          deviceType);
    }
    auto defaultBackendIt = std::ranges::find_if(
        deviceTypeToBackendType_,
        [&](const auto& p) { return p.second == backendType_; });
    TORCH_CHECK(
        defaultBackendIt != deviceTypeToBackendType_.end() &&
            deviceTypeFilter.contains(defaultBackendIt->first),
        "splitGroup deviceTypes filter must include the parent process group's default backend device type.");
  }
  std::unordered_set<BackendType> validatedBackendTypes;
  for (const auto& [deviceType, backendType] : deviceTypeToBackendType_) {
    if (!deviceTypeFilter.empty() && !deviceTypeFilter.contains(deviceType)) {
      continue;
    }
    if (!validatedBackendTypes.insert(backendType).second) {
      continue;
    }
    TORCH_CHECK(
        getBackend(deviceType)->isInitialized(),
        "Parent process group backend is not initialized; pass device_id "
        "when creating it or run a collective before split_group");
  }
  c10::intrusive_ptr<ProcessGroup> newGroup;
  std::string groupName = name.has_value()
      ? name.value()
      : c10::str(getGroupName(), ":split:", fmt::format("{}", ranks));
  c10::intrusive_ptr<Store> store = c10::static_intrusive_pointer_cast<Store>(
      c10::make_intrusive<PrefixStore>(
          fmt::format("{}/", groupName), store_->clone()));
  std::string groupDesc = desc.has_value()
      ? desc.value()
      : c10::str(getGroupDesc(), ":split:", incrementSplitCount());
  std::unordered_map<BackendType, c10::intrusive_ptr<Backend>> splitBackends;
  for (const auto& pair : deviceTypeToBackendType_) {
    c10::DeviceType deviceType = pair.first;
    BackendType backendType = pair.second;

    if (!deviceTypeFilter.empty() && !deviceTypeFilter.contains(deviceType)) {
      continue;
    }

    // One backend instance can serve several device types -- a gloo world maps
    // both cpu and cuda to the same ProcessGroupGloo, and setBackend() reuses
    // whatever is already registered for a backend type. Split it once: the
    // second child would be thrown away by setBackend() below, but only after
    // rendezvousing on the same store prefix as the real child, which races
    // with it and hangs.
    auto splitIt = splitBackends.find(backendType);
    if (splitIt != splitBackends.end()) {
      newGroup->setBackend(deviceType, backendType, splitIt->second);
      continue;
    }

    auto parentBackend = getBackend(deviceType);
    // `opts` describes the group's default backend, mirroring what
    // `pg_options` means for init_process_group; a backend of another type
    // would reject it (or worse, silently fall back to defaults and drop the
    // caller's timeout), so those inherit the parent's own options instead.
    auto backendOpts = cloneOptions(
        opts.has_value() && backendType == backendType_
            ? opts.value()
            : parentBackend->getBackendOptions());
    backendOpts->group_name = groupName;
    backendOpts->timeout =
        timeout.has_value() ? timeout.value() : backendOpts->timeout;
    backendOpts->group_desc = groupDesc;
    auto splitBackend = parentBackend->split(store, ranks, backendOpts);
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
    splitBackends.emplace(backendType, splitBackend);
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
  std::unordered_map<BackendType, c10::intrusive_ptr<Backend>> mergedBackends;
  for (const auto& pair : deviceTypeToBackendType_) {
    c10::DeviceType deviceType = pair.first;
    BackendType backendType = pair.second;
    // Merge each backend instance once; see the same guard in splitGroup().
    auto mergedIt = mergedBackends.find(backendType);
    if (mergedIt != mergedBackends.end()) {
      newGroup->setBackend(deviceType, backendType, mergedIt->second);
      continue;
    }
    auto parentBackend = getBackend(deviceType);
    auto backendOpts = cloneOptions(parentBackend->getBackendOptions());
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
    mergedBackends.emplace(backendType, mergedBackend);
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

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
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

namespace {

std::vector<c10::intrusive_ptr<c10d::Work>> pop_works(
    const at::Tensor& tensor) {
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
  return works;
}

} // namespace

at::Tensor wait_tensor(const at::Tensor& tensor) {
  auto works = pop_works(tensor);

  for (const auto& work : works) {
    work->wait();
  }
  return tensor;
}

std::vector<at::Tensor> wait_tensors(at::TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "wait_tensors requires at least one tensor");
  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  std::unordered_set<c10d::Work*> seen;
  for (const auto& tensor : tensors) {
    for (auto& work : pop_works(tensor)) {
      if (seen.insert(work.get()).second) {
        works.push_back(std::move(work));
      }
    }
  }
  for (const auto& work : works) {
    work->wait();
  }
  return tensors.vec();
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
