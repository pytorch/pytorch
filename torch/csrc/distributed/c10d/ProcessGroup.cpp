#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/RankLocal.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <string_view>

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
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size)
    : store_(store),
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

} // namespace c10d

namespace {

class WorkRegistry {
 public:
  void register_work(
      const at::Tensor& tensor,
      const c10::intrusive_ptr<c10d::Work>& work) {
    auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto [it, inserted] = registry_.try_emplace(std::move(storage), work);
    TORCH_CHECK(
        inserted || it->second != work,
        "The tensor storage is already associated with another work.");
  }

  c10::intrusive_ptr<c10d::Work> pop_work(const at::Tensor& tensor) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto it = registry_.find(storage);
    if (it == registry_.end()) {
      return nullptr;
    }
    auto work = it->second;
    registry_.erase(it);
    return work;
  }

  void unregister_completed_works() {
    std::unique_lock lock(lock_);
    for (auto it = registry_.begin(); it != registry_.end();) {
      if (it->second->isCompleted()) {
        it = registry_.erase(it);
      } else {
        ++it;
      }
    }
  }

  size_t get_work_registry_size() {
    std::unique_lock lock(lock_);
    return registry_.size();
  }

  ~WorkRegistry() {
    unregister_completed_works();
    // If there are still unwaited work objects, their corresponding process
    // groups should have already been destroyed at this stage. Any attempts to
    // wait for these work objects or to destroy them will only result in
    // confusing errors. Therefore, we simply issue a warning and intentionally
    // allow the unwaited work objects to leak.
    if (!registry_.empty()) {
      TORCH_WARN(
          "At the time of process termination, there are still ",
          registry_.size(),
          " unwaited c10d_functional collective calls. "
          "Please review your program to ensure c10d_functional.wait_tensor() "
          "is invoked on all tensors returned from c10d_functional collective "
          "ops before they are used.");
    }
    for (auto& it : registry_) {
      it.second.release();
    }
  }

 private:
  std::unordered_map<
      c10::weak_intrusive_ptr<c10::StorageImpl>,
      c10::intrusive_ptr<c10d::Work>>
      registry_;
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
  auto work = RankLocal<WorkRegistry>::get().pop_work(tensor);
  if (work != nullptr) {
    work->wait();
  }
  return tensor;
}

void unregister_completed_works() {
  RankLocal<WorkRegistry>::get().unregister_completed_works();
}

size_t get_work_registry_size() {
  return RankLocal<WorkRegistry>::get().get_work_registry_size();
}

} // namespace c10d
