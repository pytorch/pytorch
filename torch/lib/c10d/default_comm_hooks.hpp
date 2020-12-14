#pragma once

#include <c10d/comm.hpp>
#include <c10d/ProcessGroup.hpp>

namespace c10d {

enum class BuiltinCommHookType {
  ALLREDUCE = 1,
  FP16_COMPRESS = 2,
};

class AllReduceCommHook : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit AllReduceCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~AllReduceCommHook() override {}

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

class FP16CompressCommHook : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit FP16CompressCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~FP16CompressCommHook() override {}

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
