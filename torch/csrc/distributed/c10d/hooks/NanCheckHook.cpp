// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/NanCheckHook.hpp>

#include <atomic>

#include <torch/csrc/distributed/c10d/NanCheck.hpp>

namespace c10d {

namespace {

// Hook ids must not collide with user-registered hooks; carve out a range
// far above small hand-picked ids.
std::atomic<int64_t> next_hook_id{0x4e414e43 /* 'NANC' */};

} // namespace

std::shared_ptr<NanCheckHook> NanCheckHook::attach(
    const c10::intrusive_ptr<ProcessGroup>& pg) {
  auto hook = std::shared_ptr<NanCheckHook>(new NanCheckHook(pg));
  // The lambda owns the hook, so the check's lifetime is the process group's:
  // it goes away with the group (or with an explicit remove()), and the caller
  // does not have to keep the returned handle alive.
  pg->registerPreHook(
      hook->hook_id_, [hook](const PreHookArgs& args) { hook->onPre(args); });
  return hook;
}

NanCheckHook::NanCheckHook(const c10::intrusive_ptr<ProcessGroup>& pg)
    : pg_(pg), rank_(pg->getRank()), hook_id_(next_hook_id++) {
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
  // Only send buffers are checked -- receive buffers legitimately hold
  // uninitialized data. For broadcast and scatter the in-place / input buffer
  // is a send buffer on the root only, matching ProcessGroupNCCL's native
  // rank filtering.
  if ((args.name == HookOpName::BROADCAST ||
       args.name == HookOpName::SCATTER) &&
      args.root != rank_) {
    return;
  }
  for (const auto& tensor : args.input_tensors) {
    if (tensor.is_floating_point()) {
      checkForNan(tensor);
    }
  }
}

} // namespace c10d
