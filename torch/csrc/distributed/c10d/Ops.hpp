#pragma once

#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>

namespace c10d {
namespace ops {

// Below are essentially ProcessGroup's corresponding ops but routed to the
// dispatcher. To be noted, it's a convention to use at::TensorList to represent
// const std::vector<at::Tensor>&. However, const std::vector<at::Tensor>& is
// used whenever the API accepts std::vector<std::vector<at::Tensor>>& to keep
// consistency.
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> broadcast(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const BroadcastOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> allreduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const AllreduceOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> allgather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const AllgatherOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> reduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const ReduceOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> gather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const GatherOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ScatterOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> alltoall(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList output_tensors,
    at::TensorList input_tensors,
    const AllToAllOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> barrier(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const BarrierOptions& opts = {});

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> send(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t dstRank,
    int64_t tag);

TORCH_API c10::intrusive_ptr<ProcessGroup::Work> recv(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t srcRank,
    int64_t tag);

// note [Why declaring w/o NCCL guards]
// As `c10d/Ops.cpp` is included in `torch_cpu`,
// certain NCCL related macros such as `USE_NCCL`
// and `USE_C10D_NCCL` are not available in this header.
TORCH_API c10::intrusive_ptr<ProcessGroup::Work>
nccl_premulsum_allreduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, const AllreduceOptions& opts);

} // namespace ops
} // namespace c10d
