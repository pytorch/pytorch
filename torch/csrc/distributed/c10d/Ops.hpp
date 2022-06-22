#pragma once

#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>

namespace c10d {
namespace ops {

// Below are essentially ProcessGroup's corresponding ops but routed to the dispatcher.
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> broadcast(const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, const BroadcastOptions& opts = {});
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> allreduce(const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, const AllreduceOptions& opts = {});
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> allgather(const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& output_tensors, const std::vector<at::Tensor>& input_tensors,
    const AllgatherOptions& opts = {});
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts = {});
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> reduce(const c10::intrusive_ptr<ProcessGroup>& process_group,
      at::TensorList tensors,
      const ReduceOptions& opts = {});
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> gather(const c10::intrusive_ptr<ProcessGroup>& process_group,
      const std::vector<std::vector<at::Tensor>>& output_tensors,
      const std::vector<at::Tensor>& input_tensors,
      const GatherOptions& opts = {});

} // namespace ops
} // namespace c10d
