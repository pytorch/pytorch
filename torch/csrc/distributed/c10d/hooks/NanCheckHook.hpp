// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// NanCheckHook: backend-agnostic NaN checking built on the ProcessGroup
// pre/post collective hooks (Hooks.hpp). ProcessGroupNCCL has a native NaN
// checker (TORCH_NCCL_NAN_CHECK); because the hooks fire from the dispatcher
// kernels in Ops.cpp, this hook brings the same debug feature to any backend
// routed through the c10d ops -- nccl2, nccl-lazy, gloo, custom backends.

#pragma once

#include <memory>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

class TORCH_API NanCheckHook
    : public std::enable_shared_from_this<NanCheckHook> {
 public:
  // Attaches the check to the process group, which owns it from then on: the
  // check lives as long as the group's hook map. The returned handle only needs
  // to be kept if the caller wants to remove() the check early.
  static std::shared_ptr<NanCheckHook> attach(
      const c10::intrusive_ptr<ProcessGroup>& pg);

  NanCheckHook(const NanCheckHook&) = delete;
  NanCheckHook(NanCheckHook&&) = delete;
  NanCheckHook& operator=(const NanCheckHook&) = delete;
  NanCheckHook& operator=(NanCheckHook&&) = delete;

  // Detach from the process group. Idempotent, and a no-op once the group is
  // gone.
  void remove();

 private:
  explicit NanCheckHook(const c10::intrusive_ptr<ProcessGroup>& pg);
  void onPre(const PreHookArgs& args) const;

  // Weak: the group owns the hook, so an owning pointer back would keep the
  // group alive forever.
  c10::weak_intrusive_ptr<ProcessGroup> pg_;
  int rank_;
  int64_t hook_id_;
};

} // namespace c10d
