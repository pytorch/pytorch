#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/comm.hpp>

namespace c10d {

enum class BuiltinCommHookType {
  ALLREDUCE = 1,
  FP16_COMPRESS = 2,
};

class AllReduceCommHook : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit AllReduceCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~AllReduceCommHook() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

class FP16CompressCommHook : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit FP16CompressCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~FP16CompressCommHook() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

// Almost same as AllReduceCommHook, but without division inside the hook.
// This enables the optimization of fusing copy and division and saves one scan
// over all the input parameters, when no communication hook is provided by the user.
// Only used internally and not released as a public built-in communication hook.
class _AllReduceBySumCommHook
    : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit _AllReduceBySumCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~_AllReduceBySumCommHook() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
