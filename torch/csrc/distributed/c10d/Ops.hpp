#pragma once

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {
namespace ops {

// Below are essentially ProcessGroup's corresponding ops but routed to the
// dispatcher. To be noted, it's a convention to use at::TensorList to represent
// const std::vector<at::Tensor>&. However, const std::vector<at::Tensor>& is
// used whenever the API accepts std::vector<std::vector<at::Tensor>>& to keep
// consistency.
TORCH_API c10::intrusive_ptr<Work> broadcast(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const BroadcastOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> allreduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const AllreduceOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> allreduce_coalesced(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const AllreduceCoalescedOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> allgather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const AllgatherOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> _allgather_base(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> reduce_scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> reduce(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    const ReduceOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> gather(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const std::vector<at::Tensor>& input_tensors,
    const GatherOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> scatter(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ScatterOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> alltoall(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList output_tensors,
    at::TensorList input_tensors,
    const AllToAllOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> barrier(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const BarrierOptions& opts = {});

TORCH_API c10::intrusive_ptr<Work> send(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t dstRank,
    int64_t tag);

TORCH_API c10::intrusive_ptr<Work> recv(
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors,
    int64_t srcRank,
    int64_t tag);

} // namespace ops
} // namespace c10d
