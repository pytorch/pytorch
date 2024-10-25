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
#include <utility>

using WorkPtr = c10::intrusive_ptr<c10d::Work>;
using WeakWorkPtr = c10::weak_intrusive_ptr<c10d::Work>;

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

} // namespace c10d

namespace {

class WorkRegistry {
 public:
  template<typename PtrType>
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

    if constexpr (std::is_same_v<PtrType, WeakWorkPtr>) {
      std::cout << "Registering weak work" << std::endl;
        auto it = weak_registry_.find(storage);
        if (it == weak_registry_.end()) {
            weak_registry_.emplace(
                std::move(storage),
                std::vector<WeakWorkPtr>{WeakWorkPtr(work)});
        } else {
            it->second.push_back(WeakWorkPtr(work));
        }
    } else {
      std::cout << "Registering strong work" << std::endl;
      auto it = registry_.find(storage);
      if (it == registry_.end()) {
          registry_.emplace(
              std::move(storage),
              std::vector<WorkPtr>{work});
      } else {
          it->second.push_back(work);
      }
    }
  }

  template<typename PtrType>
  void wait_works(const at::Tensor& tensor) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);

    if constexpr (std::is_same_v<PtrType, WeakWorkPtr>) {
        auto it = weak_registry_.find(storage);
        if (it == weak_registry_.end()) {
            return;
        }
        auto works = it->second;
        weak_registry_.erase(it);
        for (const auto& work : works) {
            if (!work.expired()) {
                std::cout << "Waiting for weak work" << std::endl;
                work.lock()->wait();
            }
        }
    } else {
        auto it = registry_.find(storage);
        if (it == registry_.end()) {
            return;
        }
        auto works = it->second;
        registry_.erase(it);
        for (const auto& work : works) {
            work->wait();
        }
    }
  }

  template<typename PtrType>
  size_t get_work_registry_size() {
    std::unique_lock lock(lock_);
    size_t total_size = 0;
    if constexpr (std::is_same_v<PtrType, WeakWorkPtr>) {
      for (const auto& [storage, works] : weak_registry_) {
        total_size += works.size();
      }
    } else {
      for (const auto& [storage, works] : registry_) {
        total_size += works.size();
      }
    }
    return total_size;
  }

  ~WorkRegistry() {
    // If there are still unwaited functional collective work objects, their corresponding process
    // groups should have already been destroyed at this stage. Any attempts to
    // wait for these work objects or to destroy them will only result in
    // confusing errors. Therefore, we simply issue a warning and intentionally
    // allow the unwaited work objects to leak.
    size_t registry_size = get_work_registry_size<WorkPtr>();
    if (registry_size > 0) {
      TORCH_WARN(
          "At the time of process termination, there are still ",
          registry_size,
          " unwaited c10d_functional collective calls. "
          "Please review your program to ensure c10d_functional.wait_tensor() "
          "is invoked on all tensors returned from c10d_functional collective "
          "ops before they are used."
      );
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
      std::vector<WorkPtr>>
      registry_;
  std::unordered_map<
      c10::weak_intrusive_ptr<c10::StorageImpl>,
      std::vector<WeakWorkPtr>>
      weak_registry_;
  std::mutex lock_;
};

static WorkRegistry process_registry;

} // namespace

namespace c10d {

void register_work(
    const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::Work>& work) {
  RankLocal<WorkRegistry>::get().register_work<WorkPtr>(tensor, work);
}

void register_work_weakref(
    const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::Work>& work) {
  RankLocal<WorkRegistry>::get().register_work<WeakWorkPtr>(tensor, work);
}

at::Tensor wait_tensor(const at::Tensor& tensor) {
  RankLocal<WorkRegistry>::get().wait_works<WorkPtr>(tensor);
  RankLocal<WorkRegistry>::get().wait_works<WeakWorkPtr>(tensor);
  return tensor;
}

size_t get_work_registry_size(bool is_weakref) {
  if (is_weakref) {
    return RankLocal<WorkRegistry>::get().get_work_registry_size<WeakWorkPtr>();
  } else {
    return RankLocal<WorkRegistry>::get().get_work_registry_size<WorkPtr>();
  }
}

} // namespace c10d
