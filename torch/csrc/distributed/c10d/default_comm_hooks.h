#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.h>

namespace c10d {

enum class BuiltinCommHookType {
  NONE = 0,
  ALLREDUCE = 1,
  FP16_COMPRESS = 2,
};

class AllReduceCommHook : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit AllReduceCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~AllReduceCommHook() override {}

  c10::intrusive_ptr<torch::jit::Future> runHook(GradBucket& bucket) override;
};

class FP16CompressCommHook : public CppCommHookInterface<ProcessGroup*> {
 public:
  explicit FP16CompressCommHook(ProcessGroup* state)
      : CppCommHookInterface<ProcessGroup*>(state) {}

  ~FP16CompressCommHook() override {}

  c10::intrusive_ptr<torch::jit::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
