// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/NanCheckHook.hpp>

#include <mutex>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/NanCheck.hpp>

namespace c10d {

namespace {

// Hook ids must not collide with user-registered hooks; carve out a range
// far above small hand-picked ids.
std::atomic<int64_t> next_hook_id{0x4e414e43 /* 'NANC' */};

// Checks attached by attachToggleable(), keyed by the backend that toggles
// them. Weak: the group owns the check and drops it in release_resources()
// before its backends, so an entry expires before its address can be reused.
// Holding the backend itself, even weakly, would defer its destructor.
std::mutex toggleable_mutex;
std::unordered_map<const Backend*, std::weak_ptr<NanCheckHook>> toggleable;

// Caller holds toggleable_mutex.
void purgeDeadToggleable() {
  std::erase_if(
      toggleable, [](const auto& entry) { return entry.second.expired(); });
}

} // namespace

std::shared_ptr<NanCheckHook> NanCheckHook::attach(
    const c10::intrusive_ptr<ProcessGroup>& pg) {
  auto hook = std::shared_ptr<NanCheckHook>(
      new NanCheckHook(pg, std::nullopt, /*enabled=*/true));
  // The lambda owns the hook, so the check's lifetime is the process group's:
  // it goes away with the group (or with an explicit remove()), and the caller
  // does not have to keep the returned handle alive.
  pg->registerPreHook(
      hook->hook_id_, [hook](const PreHookArgs& args) { hook->onPre(args); });
  return hook;
}

std::shared_ptr<NanCheckHook> NanCheckHook::attachToggleable(
    const c10::intrusive_ptr<ProcessGroup>& pg,
    const c10::intrusive_ptr<Backend>& backend,
    c10::DeviceType device_type,
    bool enabled) {
  TORCH_CHECK(backend, "NanCheckHook: null backend");
  auto hook =
      std::shared_ptr<NanCheckHook>(new NanCheckHook(pg, device_type, enabled));
  pg->registerPreHook(
      hook->hook_id_, [hook](const PreHookArgs& args) { hook->onPre(args); });
  std::lock_guard<std::mutex> lock(toggleable_mutex);
  purgeDeadToggleable();
  toggleable.insert_or_assign(backend.get(), hook);
  return hook;
}

void NanCheckHook::setEnabled(const Backend* backend, bool enable) {
  std::shared_ptr<NanCheckHook> hook;
  {
    std::lock_guard<std::mutex> lock(toggleable_mutex);
    purgeDeadToggleable();
    if (auto it = toggleable.find(backend); it != toggleable.end()) {
      hook = it->second.lock();
    }
  }
  TORCH_CHECK(
      hook,
      "No NaN check is attached for this backend; it is attached when the "
      "process group is created through torch.distributed");
  hook->enabled_.store(enable, std::memory_order_relaxed);
}

NanCheckHook::NanCheckHook(
    const c10::intrusive_ptr<ProcessGroup>& pg,
    std::optional<c10::DeviceType> device_type,
    bool enabled)
    : pg_(pg),
      hook_id_(next_hook_id++),
      device_type_(device_type),
      enabled_(enabled) {
  TORCH_CHECK(pg, "NanCheckHook: null process group");
}

void NanCheckHook::remove() {
  // Unregistering drops the group's reference, which is normally the only one
  // keeping this hook alive.
  auto keepalive = weak_from_this().lock();
  if (auto pg = pg_.lock()) {
    pg->unregisterPreHook(hook_id_);
  }
  pg_.reset();
}

void NanCheckHook::onPre(const PreHookArgs& args) const {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  // Only send buffers are checked -- receive buffers legitimately hold
  // uninitialized data. For broadcast and scatter the in-place / input buffer
  // is a send buffer on the root only, matching ProcessGroupNCCL's native
  // rank filtering. The rank is read per call: reconfigure can change it.
  if (args.name == HookOpName::BROADCAST || args.name == HookOpName::SCATTER) {
    auto pg = pg_.lock();
    if (!pg || args.root != pg->getRank()) {
      return;
    }
  }
  for (const auto& tensor : args.input_tensors) {
    // The kernel reads numel() contiguous elements; leave other layouts to the
    // backend's own validation.
    if (tensor.is_floating_point() && tensor.is_non_overlapping_and_dense() &&
        (!device_type_ || tensor.device().type() == *device_type_)) {
      checkForNan(tensor);
    }
  }
}

} // namespace c10d
