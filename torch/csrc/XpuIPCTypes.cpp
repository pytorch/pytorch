#include <torch/csrc/XpuIPCTypes.h>

#ifdef USE_XPU

#include <ATen/MapAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {

namespace {

inline constexpr int64_t XPU_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

inline int64_t AtomicLoadCounter(const int64_t* counter_ptr) {
  return __atomic_load_n(counter_ptr, __ATOMIC_ACQUIRE);
}

inline void AtomicStoreCounter(int64_t* counter_ptr, int64_t value) {
  __atomic_store_n(counter_ptr, value, __ATOMIC_RELEASE);
}

inline int64_t AtomicDecrementCounter(int64_t* counter_ptr) {
  return __atomic_fetch_sub(counter_ptr, static_cast<int64_t>(1), __ATOMIC_ACQ_REL);
}

struct XpuIPCRefCountersFile final {
  XpuIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  int64_t* counter_ptr() {
    return static_cast<int64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(int64_t value) {
    AtomicStoreCounter(counter_ptr(), value);
  }

  bool have_offsets() const {
    return next_offset_ < size_;
  }

  bool offsets_in_use() const {
    return used_slots_ > 0;
  }

  uint64_t get_offset() const {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset() {
    used_slots_--;
  }

  const std::string& handle() const {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

class XpuIPCSentData final {
 public:
  XpuIPCSentData(
      std::string handle,
      uint64_t offset,
      int64_t* counter_ptr)
      : handle_(std::move(handle)),
        offset_(offset),
        counter_ptr_(counter_ptr) {}

  ~XpuIPCSentData();

  int64_t counter_value() const {
    return AtomicLoadCounter(counter_ptr_);
  }

  const std::string& handle() const {
    return handle_;
  }

  uint64_t offset() const {
    return offset_;
  }

  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }

  void set_export_handle_owner(std::shared_ptr<void> handle_owner) {
    export_handle_owner_ = std::move(handle_owner);
  }

 private:
  std::string handle_;
  uint64_t offset_;
  int64_t* counter_ptr_;
  at::DataPtr original_ptr_;
  std::shared_ptr<void> export_handle_owner_;
};

struct XpuIPCSentDataLimbo final {
  bool collect() {
    bool freed_memory = false;
    std::vector<std::unique_ptr<XpuIPCSentData>> reset_blocks;
    {
      std::lock_guard<std::mutex> lock(limbo_mutex_);
      std::vector<std::unique_ptr<XpuIPCSentData>> kept_blocks;
      kept_blocks.reserve(shared_blocks_.size());
      for (auto& sd : shared_blocks_) {
        if (sd->counter_value() > 0) {
          kept_blocks.push_back(std::move(sd));
        } else {
          freed_memory = true;
          reset_blocks.push_back(std::move(sd));
        }
      }
      shared_blocks_ = std::move(kept_blocks);
    }
    for (auto& sd : reset_blocks) {
      sd.reset();
    }
    return freed_memory;
  }

  void add(std::unique_ptr<XpuIPCSentData> shared_block) {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    shared_blocks_.push_back(std::move(shared_block));
    if (shared_blocks_.size() > XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO) {
      TORCH_WARN_ONCE(
          "XPU IPC tensors waiting on refcount release exceeded ",
          XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO,
          ". Consider ensuring consumers release shared tensors promptly.");
    }
  }

  uint64_t size() {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    return shared_blocks_.size();
  }

 private:
  std::vector<std::unique_ptr<XpuIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct XpuIPCGlobalEntities final {
  XpuIPCGlobalEntities() {
    alive = true;
  }

  ~XpuIPCGlobalEntities() {
    alive = false;
    limbo_.collect();
    safe_clean_current_file();
  }

  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }

  static std::atomic<bool> alive;
  std::mutex ref_counters_mutex_;
  std::unordered_map<std::string, std::shared_ptr<XpuIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<XpuIPCRefCountersFile> next_available_ref_counters_file_;
  XpuIPCSentDataLimbo limbo_;
};

std::atomic<bool> XpuIPCGlobalEntities::alive{false};
XpuIPCGlobalEntities xpu_ipc_global_entities;

void ReturnXpuRefCounter(const std::string& handle, uint64_t offset) {
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);
  auto& map = xpu_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset();
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

XpuIPCSentData::~XpuIPCSentData() {
  if (!XpuIPCGlobalEntities::alive) {
    original_ptr_.release_context();
  }
  ReturnXpuRefCounter(handle_, offset_);
}

void XpuIPCSentDataDelete(void* ptr) {
  std::unique_ptr<XpuIPCSentData> sent_data(static_cast<XpuIPCSentData*>(ptr));
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    xpu_ipc_global_entities.limbo_.add(std::move(sent_data));
  }
  xpu_ipc_global_entities.limbo_.collect();
}

at::DataPtr GetNewRefCountedXpuSentData(void* data, at::Device device) {
  std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);

  if (!xpu_ipc_global_entities.next_available_ref_counters_file_) {
    std::string ref_counter_handle = at::NewProcessWideShmHandle();
    int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
    at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(),
        flags,
        sizeof(int64_t) * XPU_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    auto rc = std::make_shared<XpuIPCRefCountersFile>(
        ref_counter_handle,
        XPU_IPC_REF_COUNTER_FILE_SIZE,
        std::move(sptr));
    xpu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
    xpu_ipc_global_entities.next_available_ref_counters_file_ = rc;
  }

  auto& file_ref = xpu_ipc_global_entities.next_available_ref_counters_file_;
  file_ref->set_counter(1);
  auto sent_data = std::make_unique<XpuIPCSentData>(
      file_ref->handle(),
      file_ref->get_offset(),
      file_ref->counter_ptr());

  file_ref->rotate_offset();
  if (!file_ref->have_offsets()) {
    file_ref.reset();
  }
  return at::DataPtr(data, sent_data.release(), XpuIPCSentDataDelete, device);
}

} // namespace

bool IsImportedXpuStorage(const c10::StorageImpl& storage) {
  return storage.received_ipc();
}

XpuSharedStorage ShareXpuStorage(const at::Storage& storage) {
  XpuSharedStorage shared;
  shared.device = storage.device().index();
  shared.size_bytes = storage.nbytes();

  if (!storage.data()) {
    return shared;
  }

  auto share_handle =
      c10::xpu::XPUCachingAllocator::shareIpcHandle(storage.mutable_data());

  shared.handle = share_handle.handle;
  shared.offset_bytes = share_handle.offset;

  at::DataPtr sent_data_ptr =
      GetNewRefCountedXpuSentData(storage.mutable_data(), storage.device());
  auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
  auto sent_data = static_cast<XpuIPCSentData*>(storage.data_ptr().get_context());
  sent_data->set_original_ptr(std::move(old_data_ptr));
  if (share_handle.owner.has_value()) {
    sent_data->set_export_handle_owner(std::move(share_handle.owner.value()));
  }

  shared.ref_counter_handle = sent_data->handle();
  shared.ref_counter_offset = sent_data->offset();
  return shared;
}

bool XpuIPCCollect() {
  if (!XpuIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = xpu_ipc_global_entities.limbo_.collect();
  if (xpu_ipc_global_entities.limbo_.size() == 0) {
    xpu_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

void ReleaseXpuIPCRefCounter(const std::string& handle, uint64_t offset) {
  if (handle.empty()) {
    return;
  }
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = at::RefcountedMapAllocator::makeDataPtr(
        handle.c_str(),
        flags,
        sizeof(int64_t) * XPU_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    auto* counter_ptr = static_cast<int64_t*>(sptr.get()) + offset;
    AtomicDecrementCounter(counter_ptr);
  } catch (c10::Error&) {
  }
}

c10::intrusive_ptr<at::StorageImpl> NewStorageFromXpuShared(
    const XpuSharedStorage& shared) {
  auto base_ptr =
      c10::xpu::XPUCachingAllocator::getIpcDevPtr(shared.handle, shared.device);

  struct XpuIpcDeleterContext {
    std::shared_ptr<void> base_ptr;
    std::string ref_counter_handle;
    uint64_t ref_counter_offset{0};
  };

  auto ctx = std::make_unique<XpuIpcDeleterContext>();
  ctx->base_ptr = std::move(base_ptr);
  ctx->ref_counter_handle = shared.ref_counter_handle;
  ctx->ref_counter_offset = shared.ref_counter_offset;

  void* dev_ptr = ctx->base_ptr.get();
  dev_ptr = static_cast<char*>(dev_ptr) + shared.offset_bytes;

  c10::DataPtr data_ptr(
      dev_ptr,
      ctx.release(),
      +[](void* ctx_) {
        std::unique_ptr<XpuIpcDeleterContext> ctx(
            static_cast<XpuIpcDeleterContext*>(ctx_));
        ctx->base_ptr.reset();
        ReleaseXpuIPCRefCounter(ctx->ref_counter_handle, ctx->ref_counter_offset);
      },
      at::Device(at::DeviceType::XPU, shared.device));

  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      shared.size_bytes,
      std::move(data_ptr),
      nullptr,
      false);
  storage->set_received_ipc(true);
  return storage;
}

} // namespace torch

namespace c10::xpu::XPUCachingAllocator {
namespace {
REGISTER_FREE_MEMORY_CALLBACK_XPU("xpu_ipc_collect", XpuIPCCollectCallback)
} // namespace
} // namespace c10::xpu::XPUCachingAllocator

#endif
