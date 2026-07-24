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

class TORCH_API NanCheckHook {
 public:
  // Attaches a hook to the process group and returns it. The hook stays
  // attached until remove() is called or the returned handle is destroyed.
  static std::shared_ptr<NanCheckHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg);

  ~NanCheckHook();

  NanCheckHook(const NanCheckHook&) = delete;
  NanCheckHook(NanCheckHook&&) = delete;
  NanCheckHook& operator=(const NanCheckHook&) = delete;
  NanCheckHook& operator=(NanCheckHook&&) = delete;

  // Detach from the process group. Idempotent.
  void remove();

 private:
  explicit NanCheckHook(c10::intrusive_ptr<ProcessGroup> pg);
  void onPre(const PreHookArgs& args);

  c10::intrusive_ptr<ProcessGroup> pg_;
  int64_t hook_id_;
};

} // namespace c10d
