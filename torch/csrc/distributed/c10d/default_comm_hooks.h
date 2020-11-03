#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.h>

namespace c10d {

class AllReduceCommHook : public CppCommHookInterface<ProcessGroup*> {
  ~AllReduceCommHook() override {}

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

class FP16CompressCommHook : public CppCommHookInterface<ProcessGroup*> {
  ~FP16CompressCommHook() override {}

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
