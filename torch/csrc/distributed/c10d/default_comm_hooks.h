#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.h>

namespace c10d {

c10::intrusive_ptr<torch::jit::Future> allReduceHook(
    ProcessGroup* process_group,
    GradBucket& bucket);

c10::intrusive_ptr<torch::jit::Future> fp16CompressHook(
    ProcessGroup* process_group,
    GradBucket& bucket);

} // namespace c10d
