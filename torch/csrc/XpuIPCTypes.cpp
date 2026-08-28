#include <torch/csrc/XpuIPCTypes.h>

#ifdef USE_XPU

#include <ATen/MapAllocator.h>
#include <ATen/StorageUtils.h>
#include <ATen/detail/XPUHooksInterface.h>

#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {

namespace {

inline constexpr int64_t XPU_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

class XpuIpcEvent;

struct XpuIPCRefCountersFile final {
  XpuIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  int64_t* counter_ptr() {
    return static_cast<int64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(int64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() const {
    return next_offset_ < size_;
  }

  bool offsets_in_use() const {
    return used_slots_;
  }

  uint64_t get_offset() const {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset) {
    (void)offset;
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
      int64_t* counter_ptr,
      at::Device device)
      : handle_(std::move(handle)),
        offset_(offset),
        counter_ptr_(counter_ptr),
        device_(device) {}

  ~XpuIPCSentData();

  int64_t counter_value() const {
    return *counter_ptr_;
  }

  const std::string& handle() const {
    return handle_;
  }

  uint64_t offset() const {
    return offset_;
  }

  at::Device device() const {
    return device_;
  }

  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }

  void set_ipc_event(std::shared_ptr<XpuIpcEvent> ipc_event) {
    ipc_event_ = std::move(ipc_event);
  }

  void set_export_handle_owner(std::shared_ptr<void> handle_owner) {
    export_handle_owner_ = std::move(handle_owner);
  }

 private:
  std::string handle_;
  uint64_t offset_;
  int64_t* counter_ptr_;
  at::DataPtr original_ptr_;
  at::Device device_;
  std::shared_ptr<XpuIpcEvent> ipc_event_;
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
    it->second->return_offset(offset);
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
      file_ref->counter_ptr(),
      device);

  file_ref->rotate_offset();
  if (!file_ref->have_offsets()) {
    file_ref.reset();
  }
  return at::DataPtr(data, sent_data.release(), XpuIPCSentDataDelete, device);
}

class XpuIpcEvent {
 public:
  static XpuIpcEvent create(c10::DeviceIndex device) {
    return XpuIpcEvent(device, false, std::nullopt);
  }

  static XpuIpcEvent open(
      c10::DeviceIndex device,
      const std::string& ipc_pool_handle) {
    return XpuIpcEvent(device, true, ipc_pool_handle);
  }

  XpuIpcEvent(const XpuIpcEvent&) = delete;
  XpuIpcEvent& operator=(const XpuIpcEvent&) = delete;

  XpuIpcEvent(XpuIpcEvent&& other) noexcept
      : pool_(other.pool_),
        event_(other.event_),
        opened_ipc_pool_(other.opened_ipc_pool_) {
    other.release();
  }

  XpuIpcEvent& operator=(XpuIpcEvent&& other) noexcept {
    if (this != &other) {
      cleanup();
      pool_ = other.pool_;
      event_ = other.event_;
      opened_ipc_pool_ = other.opened_ipc_pool_;
      other.release();
    }
    return *this;
  }

  ~XpuIpcEvent() {
    cleanup();
  }

  std::string export_handle() const {
#ifndef _WIN32
    ze_ipc_event_pool_handle_t ipc_handle{};
    const auto& ze = at::detail::getXPUHooks().level_zero();
    TORCH_CHECK(pool_, "XPU IPC event pool is not initialized before export");
    TORCH_CHECK(
        ze.zeEventPoolGetIpcHandle(pool_, &ipc_handle) == ZE_RESULT_SUCCESS,
        "Failed to export XPU IPC event pool handle");
    return std::string(
        reinterpret_cast<const char*>(&ipc_handle), sizeof(ipc_handle));
#else
    return {};
#endif
  }

  void signal() const {
#ifndef _WIN32
    const auto& ze = at::detail::getXPUHooks().level_zero();
    TORCH_CHECK(event_, "XPU IPC event is not initialized");
    TORCH_CHECK(
        ze.zeEventHostSignal(event_) == ZE_RESULT_SUCCESS,
        "Failed to signal XPU IPC event");
#endif
  }

  void wait() const {
#ifndef _WIN32
    const auto& ze = at::detail::getXPUHooks().level_zero();
    TORCH_CHECK(event_, "XPU IPC event is not initialized");
    TORCH_CHECK(
        ze.zeEventHostSynchronize(event_, UINT64_MAX) == ZE_RESULT_SUCCESS,
        "Failed to wait on XPU IPC event");
#endif
  }

 private:
  void cleanup() {
#ifndef _WIN32
    if (!XpuIPCGlobalEntities::alive) {
      release();
      return;
    }

    const auto& ze = at::detail::getXPUHooks().level_zero();
    if (event_) {
      ze.zeEventDestroy(event_);
    }
    if (pool_) {
      if (opened_ipc_pool_) {
        ze.zeEventPoolCloseIpcHandle(pool_);
      } else {
        ze.zeEventPoolDestroy(pool_);
      }
    }
#endif
  }

  void release() noexcept {
    pool_ = nullptr;
    event_ = nullptr;
    opened_ipc_pool_ = false;
  }

  XpuIpcEvent(
      c10::DeviceIndex device,
      bool open_from_ipc,
      std::optional<std::string> ipc_pool_handle) {
#ifndef _WIN32
    const auto& ze = at::detail::getXPUHooks().level_zero();
    auto& sycl_device = c10::xpu::get_raw_device(device);
    auto& sycl_context = c10::xpu::get_device_context();
    auto l0_device =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
    auto l0_context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);

    try {
      if (open_from_ipc) {
        TORCH_CHECK(ipc_pool_handle.has_value(), "Missing XPU IPC pool handle");
        TORCH_CHECK(
            ipc_pool_handle->size() == sizeof(ze_ipc_event_pool_handle_t),
            "Invalid XPU IPC event pool handle size");
        ze_ipc_event_pool_handle_t ipc_handle{};
        std::memcpy(
            &ipc_handle,
            ipc_pool_handle->data(),
            sizeof(ze_ipc_event_pool_handle_t));
        TORCH_CHECK(
            ze.zeEventPoolOpenIpcHandle(l0_context, ipc_handle, &pool_) ==
                ZE_RESULT_SUCCESS,
            "Failed to open XPU IPC event pool handle");
        opened_ipc_pool_ = true;
      } else {
        ze_event_pool_desc_t pool_desc{};
        pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC;
        pool_desc.count = 1;
        TORCH_CHECK(
            ze.zeEventPoolCreate(l0_context, &pool_desc, 1, &l0_device, &pool_) ==
                ZE_RESULT_SUCCESS,
            "Failed to create XPU IPC event pool");
      }

      ze_event_desc_t event_desc{};
      event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
      event_desc.index = 0;
      event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
      event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
      TORCH_CHECK(
          ze.zeEventCreate(pool_, &event_desc, &event_) == ZE_RESULT_SUCCESS,
          "Failed to create XPU IPC event");
    } catch (...) {
      if (pool_) {
        if (opened_ipc_pool_) {
          ze.zeEventPoolCloseIpcHandle(pool_);
        } else {
          ze.zeEventPoolDestroy(pool_);
        }
        pool_ = nullptr;
      }
      throw;
    }
#else
    (void)device;
    (void)open_from_ipc;
    (void)ipc_pool_handle;
#endif
  }

  ze_event_pool_handle_t pool_{nullptr};
  ze_event_handle_t event_{nullptr};
  bool opened_ipc_pool_{false};
};

} // namespace

bool IsImportedXpuStorage(const c10::StorageImpl& storage) {
  return storage.received_xpu();
}

XpuSharedStorage ShareXpuStorage(const at::Storage& storage) {
  XpuSharedStorage shared;
  shared.device = storage.device().index();
  shared.size_bytes = storage.nbytes();

  if (!storage.data()) {
    return shared;
  }

  auto shandle =
      c10::xpu::XPUCachingAllocator::shareIpcHandle(storage.mutable_data());
  auto ipc_event =
      std::make_shared<XpuIpcEvent>(XpuIpcEvent::create(storage.device().index()));

  shared.handle = shandle.handle;
  shared.offset_bytes = shandle.offset;
  shared.event = ipc_event->export_handle();

  c10::xpu::getCurrentXPUStream(storage.device().index()).synchronize();
  ipc_event->signal();

  at::DataPtr sent_data_ptr =
      GetNewRefCountedXpuSentData(storage.mutable_data(), storage.device());
  auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
  auto sent_data = static_cast<XpuIPCSentData*>(storage.data_ptr().get_context());
  sent_data->set_original_ptr(std::move(old_data_ptr));
  sent_data->set_ipc_event(std::move(ipc_event));
  sent_data->set_export_handle_owner(std::move(shandle.handle_owner));

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
    *(static_cast<int64_t*>(sptr.get()) + offset) -= 1;
  } catch (c10::Error&) {
  }
}

c10::intrusive_ptr<at::StorageImpl> NewStorageFromXpuShared(
    const XpuSharedStorage& shared) {
  if (!shared.event.empty()) {
    XpuIpcEvent event = XpuIpcEvent::open(shared.device, shared.event);
    event.wait();
  }

  auto base_ptr =
      c10::xpu::XPUCachingAllocator::getIpcDevPtr(shared.handle, shared.device);

  struct XpuIpcDeleterContext {
    std::shared_ptr<void> base_ptr;
    std::string ref_counter_handle;
    uint64_t ref_counter_offset{0};
    c10::DeviceIndex device{-1};
  };

  auto ctx = std::make_unique<XpuIpcDeleterContext>();
  ctx->base_ptr = std::move(base_ptr);
  ctx->ref_counter_handle = shared.ref_counter_handle;
  ctx->ref_counter_offset = shared.ref_counter_offset;
  ctx->device = shared.device;

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
  storage->set_received_xpu(true);
  return storage;
}

} // namespace torch

namespace c10::xpu::XPUCachingAllocator {
namespace {
REGISTER_FREE_MEMORY_CALLBACK_XPU("xpu_ipc_collect", XpuIPCCollectCallback)
} // namespace
} // namespace c10::xpu::XPUCachingAllocator

#endif
