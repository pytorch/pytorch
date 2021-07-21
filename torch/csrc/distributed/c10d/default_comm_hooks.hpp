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

struct _AllReduceCommHookWithDivFactorState {
  _AllReduceCommHookWithDivFactorState(ProcessGroup* pg, int div_factor)
      : pg(pg), div_factor(div_factor) {}

  ProcessGroup* pg;
  // Should be equal to the process group size, with the exception of unevent
  // input.
  int div_factor;
};

// Almost same as AllReduceCommHook, but requires an additional ``div_factor``
// as the state for handling unevent input. Only used internally and not
// released as a public built-in communication hook.
class _AllReduceCommHookWithDivFactor
    : public CppCommHookInterface<_AllReduceCommHookWithDivFactorState> {
 public:
  using CppCommHookInterface::CppCommHookInterface;

  ~_AllReduceCommHookWithDivFactor() override = default;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
