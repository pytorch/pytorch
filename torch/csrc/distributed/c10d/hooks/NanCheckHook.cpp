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
    c10::intrusive_ptr<ProcessGroup> pg) {
  auto hook = std::shared_ptr<NanCheckHook>(new NanCheckHook(std::move(pg)));
  // The lambda holds a weak_ptr so the hook -> pg -> hook cycle is broken:
  // when the caller drops the returned handle, the hook destructor
  // unregisters from the process group.
  std::weak_ptr<NanCheckHook> weak = hook;
  hook->pg_->registerPreHook(hook->hook_id_, [weak](const PreHookArgs& args) {
    if (auto self = weak.lock()) {
      self->onPre(args);
    }
  });
  return hook;
}

NanCheckHook::NanCheckHook(c10::intrusive_ptr<ProcessGroup> pg)
    : pg_(std::move(pg)), hook_id_(next_hook_id++) {
  TORCH_CHECK(pg_, "NanCheckHook: null process group");
}

NanCheckHook::~NanCheckHook() {
  remove();
}

void NanCheckHook::remove() {
  if (pg_) {
    pg_->unregisterPreHook(hook_id_);
    pg_.reset();
  }
}

void NanCheckHook::onPre(const PreHookArgs& args) {
  // Only send buffers are checked -- receive buffers legitimately hold
  // uninitialized data. For broadcast and scatter the in-place / input buffer
  // is a send buffer on the root only, matching ProcessGroupNCCL's native
  // rank filtering.
  if ((args.name == HookOpName::BROADCAST ||
       args.name == HookOpName::SCATTER) &&
      args.root != pg_->getRank()) {
    return;
  }
  for (const auto& tensor : args.input_tensors) {
    if (tensor.is_floating_point()) {
      checkForNan(tensor);
    }
  }
}

} // namespace c10d
