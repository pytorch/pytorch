#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>

namespace c10d {

enum class BuiltinCommHookType {
  ALLREDUCE = 1,
  FP16_COMPRESS = 2,
};

class AllReduceCommHook : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>> {
 public:
  explicit AllReduceCommHook(const c10::intrusive_ptr<ProcessGroup>& state)
      : CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>>(state) {}

  ~AllReduceCommHook() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

class FP16CompressCommHook : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>> {
 public:
  explicit FP16CompressCommHook(const c10::intrusive_ptr<ProcessGroup>& state)
      : CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>>(state) {}

  ~FP16CompressCommHook() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

// Almost same as AllReduceCommHook, but without division inside the hook.
// This enables the optimization of fusing copy and division and saves one scan
// over all the input parameters, when no communication hook is provided by the user.
// Only used internally and not released as a public built-in communication hook.
class _AllReduceBySumCommHook
    : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>> {
 public:
  explicit _AllReduceBySumCommHook(const c10::intrusive_ptr<ProcessGroup>& state)
      : CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>>(state) {}

  ~_AllReduceBySumCommHook() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
