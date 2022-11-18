#pragma once

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

c10::intrusive_ptr<Work> sparse_allreduce(
    ProcessGroup* pg,
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts,
    bool desyncDebug);

} // namespace c10d

#endif // USE_C10D_NCCL
